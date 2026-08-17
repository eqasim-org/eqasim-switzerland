import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, json
import matplotlib.pyplot as plt
logger = logging.getLogger("synpp: choice_model")

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_rate=0.15, apply_residual=True):
        super().__init__()
        self.apply_residual = apply_residual
        self.layer1 = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out),
            nn.SiLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.LayerNorm(dim_out)
        )
        if apply_residual:
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        if self.apply_residual:
            return self.act(x1 + x2)
        else:
            return self.act(x2)
    

class NeuralChoiceModel(nn.Module):
    def __init__(self, person_input_dim, candidate_input_dim, person_hidden_dim=32, hidden_dim=64, dropout_rate=0.15, mask_threshold=1e-4):
        super().__init__()
        self.person_input_dim = int(person_input_dim)
        self.candidate_input_dim = int(candidate_input_dim)
        self.person_hidden_dim = int(person_hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout_rate)
        self.mask_threshold = float(mask_threshold)
        self._device = torch.device("cpu")

        self.person_tower = ResidualBlock(person_input_dim, person_hidden_dim, dropout_rate=dropout_rate)
        self.utility_tower = ResidualBlock(person_hidden_dim + candidate_input_dim, hidden_dim, 
                                           dropout_rate=dropout_rate, apply_residual=False)

        # Small mask head from the last hidden dimension: predicts candidate plausibility.
        self.mask_layer = nn.Linear(hidden_dim, 1)
        self.utility_head = nn.Linear(hidden_dim, 1)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.constant_(self.mask_layer.bias, 3.0)
        nn.init.normal_(self.mask_layer.weight, std=0.01)
        nn.init.normal_(self.utility_head.weight, std=0.1)
        nn.init.zeros_(self.utility_head.bias)

    def forward(self, person_static, person_dynamic, candidate_static, candidate_dynamic):
        if person_static.ndim == 1:   person_static   = person_static.unsqueeze(0)
        if person_dynamic.ndim == 1:  person_dynamic  = person_dynamic.unsqueeze(0)
        if candidate_static.ndim == 2:  candidate_static  = candidate_static.unsqueeze(0)
        if candidate_dynamic.ndim == 2: candidate_dynamic = candidate_dynamic.unsqueeze(0)
        # Reconstruct combined tensors preserving the [dynamic|static] training layout.
        # candidate_static may be [1, N, C] (broadcast) — expand to match dynamic's batch dim.
        if candidate_static.shape[0] == 1 and candidate_dynamic.shape[0] > 1:
            candidate_static = candidate_static.expand(candidate_dynamic.shape[0], -1, -1)
        person_x    = torch.cat([person_static, person_dynamic], dim=-1)          # [B, person_input_dim]
        candidate_x = torch.cat([candidate_dynamic, candidate_static], dim=-1)    # [B, N, candidate_input_dim]
        if candidate_x.shape[-1] != self.candidate_input_dim:
            raise RuntimeError(
                "NeuralChoiceModel expected candidate feature width "
                f"{self.candidate_input_dim}, got {candidate_x.shape[-1]}. "
                "Check candidate_input_dim at model construction."
            )

        batch_size, n_candidates, _ = candidate_x.shape
        person_embedding = self.person_tower(person_x)
        person_embedding = person_embedding[:, None, :].expand(batch_size, n_candidates, self.person_hidden_dim)

        fused = torch.cat([candidate_x, person_embedding], dim=-1)
        fused = fused.reshape(-1, fused.shape[-1])

        hidden = self.utility_tower(fused)
        base_utilities = self.utility_head(hidden).reshape(batch_size, n_candidates)

        mask_logits = self.mask_layer(hidden).reshape(batch_size, n_candidates)
        mask_prob = torch.sigmoid(mask_logits)

        # Soft penalty in utility space; low-probability candidates are down-weighted.
        # Keep this always-on so the mask head can still shape utilities during training.
        utilities = base_utilities + torch.log(mask_prob.clamp_min(1e-8))

        return utilities


def train_choice_model(model, person_static_x, person_dynamic_x, candidate_static_x, candidate_dynamic_x, y,
                       valid_mask, epochs=50, batch_size=256, lr=1e-3, weight_decay=4e-3, num_threads=None,
                       logger_instance=None, weights=None, grad_clip=5.0, lr_step_size=10, lr_gamma=0.5,
                       path=None, n_val_folds=5, val_every=5,
                       validation_groups=None, rotate_validation=True,
                       early_stopping_patience=None, early_stopping_min_delta=0.0,
                       restore_best_weights=False,
                       distance_candidates=None, distance_targets=None,
                       distance_candidates_home=None, distance_targets_home=None,
                       distance_loss_weight=0.0, distance_loss_short_floor_m=100.0):
    """Train the choice model with rotating or fixed-fold validation.

    ``validation_groups`` keeps related samples (for example, trips belonging to
    one person) in the same fold. Early stopping requires a fixed validation fold,
    because losses from different rotating folds are not directly comparable.

    If ``path`` is given, loss history is saved to ``path/loss_history.json``
    and a plot to ``path/lossevolution.png``.
    """

    lr_step_size = int(np.clip(lr_step_size, max(1, epochs // 6), max(1, epochs // 3)))
    if num_threads is not None:
        torch.set_num_threads(max(1, int(num_threads)))

    device = model._device
    model.to(device)

    person_static_tensor    = torch.as_tensor(person_static_x,    dtype=torch.float32, device=device)
    person_dynamic_tensor   = torch.as_tensor(person_dynamic_x,   dtype=torch.float32, device=device)
    candidate_static_tensor = torch.as_tensor(candidate_static_x, dtype=torch.float32, device=device)
    candidate_dynamic_tensor= torch.as_tensor(candidate_dynamic_x,dtype=torch.float32, device=device)
    y_tensor      = torch.as_tensor(y,          dtype=torch.long,  device=device)
    mask_tensor   = torch.as_tensor(valid_mask, dtype=torch.bool,  device=device)
    weights_tensor = (torch.as_tensor(weights, dtype=torch.float32, device=device)
                      if weights is not None
                      else torch.ones(person_static_tensor.shape[0], device=device))

    use_distance_last_loss = (
        float(distance_loss_weight) > 0.0
        and distance_candidates is not None
        and distance_targets is not None
    )
    use_distance_home_loss = (
        float(distance_loss_weight) > 0.0
        and distance_candidates_home is not None
        and distance_targets_home is not None
    )
    if use_distance_last_loss:
        distance_candidates_tensor = torch.as_tensor(distance_candidates, dtype=torch.float32, device=device)
        distance_targets_tensor = torch.as_tensor(distance_targets, dtype=torch.float32, device=device)
        if distance_candidates_tensor.ndim != 2:
            raise ValueError("distance_candidates must be a 2D array [n_samples, n_candidates]")
        if distance_targets_tensor.ndim != 1:
            raise ValueError("distance_targets must be a 1D array [n_samples]")
        if distance_candidates_tensor.shape[0] != person_static_tensor.shape[0]:
            raise ValueError("distance_candidates row count must match number of samples")
        if distance_targets_tensor.shape[0] != person_static_tensor.shape[0]:
            raise ValueError("distance_targets length must match number of samples")
    else:
        distance_candidates_tensor = None
        distance_targets_tensor = None
    if use_distance_home_loss:
        distance_candidates_home_tensor = torch.as_tensor(distance_candidates_home, dtype=torch.float32, device=device)
        distance_targets_home_tensor = torch.as_tensor(distance_targets_home, dtype=torch.float32, device=device)
        if distance_candidates_home_tensor.ndim != 2:
            raise ValueError("distance_candidates_home must be a 2D array [n_samples, n_candidates]")
        if distance_targets_home_tensor.ndim != 1:
            raise ValueError("distance_targets_home must be a 1D array [n_samples]")
        if distance_candidates_home_tensor.shape[0] != person_static_tensor.shape[0]:
            raise ValueError("distance_candidates_home row count must match number of samples")
        if distance_targets_home_tensor.shape[0] != person_static_tensor.shape[0]:
            raise ValueError("distance_targets_home length must match number of samples")
    else:
        distance_candidates_home_tensor = None
        distance_targets_home_tensor = None
    short_floor = float(max(1e-6, distance_loss_short_floor_m))

    n_samples   = person_static_tensor.shape[0]
    n_val_folds = max(2, int(n_val_folds))
    val_every   = max(1, int(val_every))
    rotate_validation = bool(rotate_validation)
    restore_best_weights = bool(restore_best_weights)
    if rotate_validation and restore_best_weights:
        raise ValueError("Best-weight restoration requires rotate_validation=False")
    if early_stopping_patience is not None:
        early_stopping_patience = max(1, int(early_stopping_patience))
        if rotate_validation:
            raise ValueError("Early stopping requires rotate_validation=False")
    early_stopping_min_delta = max(0.0, float(early_stopping_min_delta))

    # Build folds once. With groups, split unique groups first so related
    # observations cannot leak between training and validation.
    if validation_groups is None:
        perm = np.random.permutation(n_samples)
        fold_arrays = np.array_split(perm, n_val_folds)
    else:
        validation_groups = np.asarray(validation_groups)
        if validation_groups.ndim != 1 or len(validation_groups) != n_samples:
            raise ValueError("validation_groups must be a 1D array with one value per sample")
        unique_groups = np.unique(validation_groups)
        if len(unique_groups) < 2:
            raise ValueError("validation_groups must contain at least two distinct groups")
        n_val_folds = min(n_val_folds, len(unique_groups))
        group_folds = np.array_split(np.random.permutation(unique_groups), n_val_folds)
        fold_arrays = [np.flatnonzero(np.isin(validation_groups, groups)) for groups in group_folds]

    fold_indices = [
        torch.tensor(chunk.copy(), dtype=torch.long, device=device)
        for chunk in fold_arrays
    ]
    if any(len(indices) == 0 for indices in fold_indices):
        raise ValueError("Validation split produced an empty fold")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    local_logger = logger_instance or logger
    validation_mode = "rotating" if rotate_validation else "fixed"
    validation_description = (
        f"rotating {n_val_folds}-fold"
        if rotate_validation
        else f"fixed 1/{n_val_folds} holdout"
    )
    group_label = "grouped" if validation_groups is not None else "sample-level"
    local_logger.info(
        "\tStarting training (%s, %s, validation every %d epochs)",
        validation_description, group_label, val_every,
    )
    if float(distance_loss_weight) > 0.0:
        if use_distance_last_loss or use_distance_home_loss:
            local_logger.info(
                "\tDistance regularization enabled: weight=%.4f, short_floor_m=%.1f, use_last=%s, use_home=%s",
                float(distance_loss_weight),
                float(short_floor),
                str(use_distance_last_loss),
                str(use_distance_home_loss),
            )
        else:
            local_logger.warning("\tDistance regularization requested but distance tensors were not provided; disabling it")
    t0 = time.time()

    train_losses: list[float] = []
    val_losses:   list[float] = []
    val_epochs:   list[int]   = []
    best_val_loss = np.inf
    best_epoch = None
    best_state = None
    checks_without_improvement = 0
    stopped_early = False

    for epoch in range(int(epochs)):
        # Which fold is held out this epoch?
        val_fold_idx = (epoch // val_every) % n_val_folds if rotate_validation else 0
        val_idx      = fold_indices[val_fold_idx]
        train_idx    = torch.cat([fold_indices[i] for i in range(n_val_folds) if i != val_fold_idx])

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        epoch_perm = train_idx[torch.randperm(len(train_idx), device=device)]

        total_weighted_loss = 0.0
        total_weight        = 0.0

        for start in range(0, len(epoch_perm), int(batch_size)):
            end = min(start + int(batch_size), len(epoch_perm))
            idx = epoch_perm[start:end]

            batch_person_static    = person_static_tensor[idx]
            batch_person_dynamic   = person_dynamic_tensor[idx]
            batch_candidate_static = (candidate_static_tensor
                                      if candidate_static_tensor.shape[0] == 1
                                      else candidate_static_tensor[idx])
            batch_candidate_dynamic = candidate_dynamic_tensor[idx]
            batch_y    = y_tensor[idx]
            batch_mask = mask_tensor[idx]
            batch_w    = weights_tensor[idx]

            optimizer.zero_grad(set_to_none=True)
            utilities = model(batch_person_static, batch_person_dynamic,
                              batch_candidate_static, batch_candidate_dynamic)
            utilities = utilities.masked_fill(~batch_mask, -1e9)

            ce_loss_per_sample = F.cross_entropy(utilities, batch_y, reduction="none")
            weighted_ce_loss = ce_loss_per_sample * batch_w
            ce_loss = weighted_ce_loss.sum() / batch_w.sum().clamp_min(1e-12)

            if use_distance_last_loss:
                batch_dist_candidates = distance_candidates_tensor[idx]
                batch_target_distance = distance_targets_tensor[idx]
                probs = torch.softmax(utilities, dim=1)
                expected_distance = (probs * batch_dist_candidates).sum(dim=1)
                normalizer = torch.clamp(batch_target_distance + short_floor, min=short_floor)
                relative_error = (expected_distance - batch_target_distance) / normalizer
                distance_loss_per_sample = F.smooth_l1_loss(
                    relative_error,
                    torch.zeros_like(relative_error),
                    reduction="none",
                )
                weighted_distance_loss = distance_loss_per_sample * batch_w
                distance_last_loss = weighted_distance_loss.sum() / batch_w.sum().clamp_min(1e-12)
            else:
                distance_last_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

            if use_distance_home_loss:
                batch_dist_home_candidates = distance_candidates_home_tensor[idx]
                batch_target_home_distance = distance_targets_home_tensor[idx]
                probs = torch.softmax(utilities, dim=1)
                expected_home_distance = (probs * batch_dist_home_candidates).sum(dim=1)
                normalizer_home = torch.clamp(batch_target_home_distance + short_floor, min=short_floor)
                relative_home_error = (expected_home_distance - batch_target_home_distance) / normalizer_home
                home_loss_per_sample = F.smooth_l1_loss(
                    relative_home_error,
                    torch.zeros_like(relative_home_error),
                    reduction="none",
                )
                weighted_home_loss = home_loss_per_sample * batch_w
                distance_home_loss = weighted_home_loss.sum() / batch_w.sum().clamp_min(1e-12)
            else:
                distance_home_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

            distance_loss = distance_last_loss + distance_home_loss

            loss = ce_loss + float(distance_loss_weight) * distance_loss

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            total_weighted_loss += (ce_loss + float(distance_loss_weight) * distance_loss).item() * batch_w.sum().item()
            total_weight        += batch_w.sum().item()

        train_loss = total_weighted_loss / max(total_weight, 1e-12)
        train_losses.append(float(train_loss))

        # ── Validation (at last epoch of each fold window) ───────────────────
        if (epoch + 1) % val_every == 0:
            model.eval()
            with torch.inference_mode():
                v_ps = person_static_tensor[val_idx]
                v_pd = person_dynamic_tensor[val_idx]
                v_cs = (candidate_static_tensor
                        if candidate_static_tensor.shape[0] == 1
                        else candidate_static_tensor[val_idx])
                v_cd = candidate_dynamic_tensor[val_idx]
                v_y  = y_tensor[val_idx]
                v_m  = mask_tensor[val_idx]
                v_w  = weights_tensor[val_idx]

                v_utils = model(v_ps, v_pd, v_cs, v_cd)
                v_utils = v_utils.masked_fill(~v_m, -1e9)
                v_ce_loss = ((F.cross_entropy(v_utils, v_y, reduction="none") * v_w).sum()
                             / v_w.sum().clamp_min(1e-12))
                if use_distance_last_loss:
                    v_dist_candidates = distance_candidates_tensor[val_idx]
                    v_target_distance = distance_targets_tensor[val_idx]
                    v_probs = torch.softmax(v_utils, dim=1)
                    v_expected_distance = (v_probs * v_dist_candidates).sum(dim=1)
                    v_normalizer = torch.clamp(v_target_distance + short_floor, min=short_floor)
                    v_relative_error = (v_expected_distance - v_target_distance) / v_normalizer
                    v_distance_last_loss = (
                        F.smooth_l1_loss(v_relative_error, torch.zeros_like(v_relative_error), reduction="none") * v_w
                    ).sum() / v_w.sum().clamp_min(1e-12)
                else:
                    v_distance_last_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

                if use_distance_home_loss:
                    v_dist_home_candidates = distance_candidates_home_tensor[val_idx]
                    v_target_home_distance = distance_targets_home_tensor[val_idx]
                    v_probs = torch.softmax(v_utils, dim=1)
                    v_expected_home_distance = (v_probs * v_dist_home_candidates).sum(dim=1)
                    v_normalizer_home = torch.clamp(v_target_home_distance + short_floor, min=short_floor)
                    v_relative_home_error = (v_expected_home_distance - v_target_home_distance) / v_normalizer_home
                    v_distance_home_loss = (
                        F.smooth_l1_loss(v_relative_home_error, torch.zeros_like(v_relative_home_error), reduction="none") * v_w
                    ).sum() / v_w.sum().clamp_min(1e-12)
                else:
                    v_distance_home_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

                v_distance_loss = v_distance_last_loss + v_distance_home_loss

                val_loss = float((v_ce_loss + float(distance_loss_weight) * v_distance_loss).item())

            val_losses.append(val_loss)
            val_epochs.append(epoch + 1)
            # A best epoch is meaningful only when every check uses the same
            # untouched validation fold.
            improved = (
                not rotate_validation
                and val_loss < (best_val_loss - early_stopping_min_delta)
            )
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                checks_without_improvement = 0
                if restore_best_weights or early_stopping_patience is not None:
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            elif not rotate_validation:
                checks_without_improvement += 1

            local_logger.info(
                "\tEpoch %s/%s | Train: %.4f | Val (fold %d): %.4f%s | Time: %.1fmin",
                epoch + 1, epochs, train_loss, val_fold_idx, val_loss,
                " | best" if improved else "", (time.time() - t0) / 60.0,
            )
        else:
            local_logger.info(
                "\tEpoch %s/%s | Train: %.4f | Time: %.1fmin",
                epoch + 1, epochs, train_loss, (time.time() - t0) / 60.0,
            )

        scheduler.step()

        if (
            early_stopping_patience is not None
            and checks_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            local_logger.info(
                "\tEarly stopping at epoch %d after %d validation checks without improvement",
                epoch + 1, checks_without_improvement,
            )
            break

    if restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        local_logger.info(
            "\tRestored best model weights from epoch %d (validation loss %.4f)",
            best_epoch, best_val_loss,
        )

    # ── Save history & plot ───────────────────────────────────────────────────
    if path is not None:
        history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_epochs": val_epochs,
            "validation_mode": validation_mode,
            "validation_fraction": 1.0 / n_val_folds,
            "validation_grouped": validation_groups is not None,
            "validation_every": val_every,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "restore_best_weights": restore_best_weights,
            "dropout_rate": model.dropout_rate,
            "weight_decay": float(weight_decay),
            "best_epoch": best_epoch,
            "best_val_loss": None if best_epoch is None else best_val_loss,
            "stopped_early": stopped_early,
            "epochs_completed": len(train_losses),
        }
        with open(os.path.join(path, "loss_history.json"), "w") as f:
            json.dump(history, f, indent=2)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(1, len(train_losses) + 1), train_losses,
                    label="Train loss", color="steelblue", linewidth=1.5)
            if val_epochs:
                ax.plot(val_epochs, val_losses,
                        label=f"Val loss ({validation_description})",
                        color="darkorange", linewidth=1.5, marker="o", markersize=4)
            if best_epoch is not None:
                ax.axvline(best_epoch, color="forestgreen", linestyle="--", linewidth=1.2,
                           label=f"Best epoch ({best_epoch})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weighted training objective")
            ax.set_title("Training and validation loss evolution")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(path, "lossevolution.png"), dpi=150)
            plt.close(fig)
            local_logger.info("\tLoss plot saved to %s", os.path.join(path, "lossevolution.png"))


def predict_choice_proba(model, person_static, person_dynamic, candidate_static, candidate_dynamic, valid_mask_tensor=None):
    with torch.inference_mode():
        dev = model._device
        ps = torch.as_tensor(person_static,   dtype=torch.float32, device=dev)
        pd = torch.as_tensor(person_dynamic,  dtype=torch.float32, device=dev)
        cs = torch.as_tensor(candidate_static,  dtype=torch.float32, device=dev)
        cd = torch.as_tensor(candidate_dynamic, dtype=torch.float32, device=dev)
        utilities = model(ps, pd, cs, cd)
        if valid_mask_tensor is not None:
            valid_mask_tensor = torch.as_tensor(valid_mask_tensor, dtype=torch.bool, device=dev)
            if valid_mask_tensor.ndim == 1:
                valid_mask_tensor = valid_mask_tensor.unsqueeze(0).expand(utilities.shape[0], -1)
            utilities.masked_fill_(~valid_mask_tensor, -1e9)
        return torch.softmax(utilities, dim=1)
