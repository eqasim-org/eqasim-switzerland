"""Household-disjoint fitting, survey weighting and best-checkpoint selection."""

import copy
import logging

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .runtime import torch, nn, F, to_tensors
from .model import EmbeddingGRUModel
from .losses import MatchingLoss

logger = logging.getLogger("synpp")


def split_training_rows(source, fraction, random_seed):
    """Keep household members together; small fixtures use all rows for fitting."""
    if not 0 <= fraction < 1:
        raise ValueError("matching_embedding_validation_fraction must be in [0, 1)")
    groups = source.get("household_id", source["mz_id"]).fillna(source["mz_id"])
    if fraction == 0 or len(source) < 10 or groups.nunique() < 3:
        return np.arange(len(source)), np.array([], dtype=int)
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=random_seed)
    return next(splitter.split(source, groups=groups))


class EmbeddingTrainer:
    """Own optimizer, early stopping and best-model state for one fitted encoder."""

    def __init__(self, **options):
        self.options = dict(options)

    def fit(self, inputs, targets, train_rows, validation_rows):
        self.model_, self.model_kwargs_ = self._fit_model(
            inputs, targets, train_rows, validation_rows, **self.options)
        self.report_ = self.model_.training_report
        return self

    @staticmethod
    def _fit_model(x, y, train_indices, validation_indices, *, random_seed,
                              epochs, batch_size, lr, dropout, weight_decay, loss_weights,
                              proj_dims, use_class_weighting, latent_noise_std,
                              lr_decay_step, lr_decay_gamma, patience=12):
        if torch is None:
            raise RuntimeError("PyTorch is required for matched_v2")
        rng = np.random.RandomState(random_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = dict(cont_dim=x["cont"].shape[1], cardinalities=x["cardinalities"],
                      n_activity=y["num_activity_classes"], n_mode=y["num_mode_classes"],
                      dropout=dropout, **{key + "_proj_dim": value for key, value in proj_dims.items()})
        history, best_state, best_loss, stale = [], None, np.inf, 0
        # Restore the caller's torch RNG states when training finishes.
        devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(random_seed)
            model = EmbeddingGRUModel(**kwargs).to(device)
            model.latent_noise_std = latent_noise_std
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
            class_weights = {}
            if use_class_weighting:
                for name, key, count in (("activity", "act", y["num_activity_classes"]),
                                          ("mode", "mode", y["num_mode_classes"])):
                    weights = MatchingLoss.class_weights(y[key + "_tgt"][train_indices],
                                             y["sample_weight"][train_indices], count)
                    class_weights[name] = torch.as_tensor(weights, device=device)
            for epoch in range(epochs):
                model.train()
                train_total, train_weight = 0.0, 0.0
                for start in range(0, len(train_indices), batch_size):
                    # Permute once per epoch, without mutating global NumPy state.
                    if start == 0:
                        permutation = rng.permutation(train_indices)
                    idx = permutation[start:start + batch_size]
                    output = model(to_tensors(x, idx, device), y["act_tgt"].shape[1])
                    per_person, _ = MatchingLoss.per_person(output, y, idx, device, loss_weights, class_weights)
                    weights = torch.as_tensor(y["sample_weight"][idx], device=device)
                    # Global normalization avoids cancelling survey weights in singleton batches.
                    loss = (per_person * weights).mean()
                    spread = MatchingLoss.latent_spread(output["latent"])
                    loss = loss + loss_weights.get("latent_spread", 0) * spread
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite V2 embedding training loss")
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    train_total += float((per_person.detach() * weights).sum().item())
                    train_weight += float(weights.sum().item())
                scheduler.step()
                model.eval()
                total, weight_sum = 0.0, 0.0
                components = {}
                with torch.no_grad():
                    for start in range(0, len(validation_indices), batch_size):
                        idx = validation_indices[start:start + batch_size]
                        output = model(to_tensors(x, idx, device), y["act_tgt"].shape[1])
                        # Select on behavioural prediction, without class reweighting or regularizers.
                        evaluation_weights = {key: value for key, value in loss_weights.items()
                                              if key not in ("reconstruction", "latent_norm", "latent_spread")}
                        per_person, parts = MatchingLoss.per_person(output, y, idx, device, evaluation_weights, {})
                        weights = torch.as_tensor(y["sample_weight"][idx], device=device)
                        total += float((per_person * weights).sum().item())
                        weight_sum += float(weights.sum().item())
                        for key, value in parts.items():
                            components[key] = components.get(key, 0) + float((value * weights).sum().item())
                score = total / weight_sum if weight_sum else train_total / train_weight
                if not np.isfinite(score):
                    raise ValueError("Non-finite V2 validation loss")
                history.append({"epoch": epoch + 1, "train_loss": train_total / train_weight,
                                "validation_loss": score if weight_sum else None,
                                **{"validation_" + k: v / weight_sum for k, v in components.items()}})
                if score < best_loss - 1e-6:
                    best_loss, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
                    best_epoch = epoch + 1
                else:
                    stale += 1
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    logger.info("V2 epoch %d/%d: train %.4f, held-out %s", epoch + 1, epochs,
                                train_total / train_weight, f"{score:.4f}" if weight_sum else "unavailable")
                if len(validation_indices) and stale >= patience:
                    break
            model.load_state_dict(best_state)
        model.eval()
        model.training_report = {"best_epoch": best_epoch, "train_persons": len(train_indices),
                                 "validation_persons": len(validation_indices), "history": history}
        return model, kwargs


def train_embedding_model(inputs, targets, train_rows, validation_rows, **options):
    """Compatibility facade for earlier callers; all fitting belongs to EmbeddingTrainer."""
    trainer = EmbeddingTrainer(**options).fit(inputs, targets, train_rows, validation_rows)
    return trainer.model_, trainer.model_kwargs_


_class_weights = MatchingLoss.class_weights
