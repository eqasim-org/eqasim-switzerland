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
                       distance_candidates=None, distance_targets=None,
                       distance_candidates_home=None, distance_targets_home=None,
                       distance_loss_weight=0.0, distance_loss_short_floor_m=100.0):
    """Train the choice model with rotating-fold validation.

    Every ``val_every`` epochs the current hold-out fold rotates to the next one,
    so every sample spends roughly (n_val_folds-1)/n_val_folds of the time in
    training and 1/n_val_folds in validation — no data is permanently discarded.

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

    # Build rotating folds once, with a fixed random permutation.
    perm = np.random.permutation(n_samples)
    fold_indices = [
        torch.tensor(chunk.copy(), dtype=torch.long, device=device)
        for chunk in np.array_split(perm, n_val_folds)
    ]

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    local_logger = logger_instance or logger
    local_logger.info("\tStarting training (rotating %d-fold validation every %d epochs)", n_val_folds, val_every)
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

    for epoch in range(int(epochs)):
        # Which fold is held out this epoch?
        val_fold_idx = (epoch // val_every) % n_val_folds
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
            local_logger.info(
                "\tEpoch %s/%s | Train: %.4f | Val (fold %d): %.4f | Time: %.1fmin",
                epoch + 1, epochs, train_loss, val_fold_idx, val_loss, (time.time() - t0) / 60.0,
            )
        else:
            local_logger.info(
                "\tEpoch %s/%s | Train: %.4f | Time: %.1fmin",
                epoch + 1, epochs, train_loss, (time.time() - t0) / 60.0,
            )

        scheduler.step()

    # ── Save history & plot ───────────────────────────────────────────────────
    if path is not None:
        history = {"train_losses": train_losses, "val_losses": val_losses, "val_epochs": val_epochs}
        with open(os.path.join(path, "loss_history.json"), "w") as f:
            json.dump(history, f, indent=2)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(1, epochs + 1), train_losses,
                    label="Train loss", color="steelblue", linewidth=1.5)
            if val_epochs:
                ax.plot(val_epochs, val_losses,
                        label=f"Val loss (rotating {n_val_folds}-fold)",
                        color="darkorange", linewidth=1.5, marker="o", markersize=4)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weighted cross-entropy")
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