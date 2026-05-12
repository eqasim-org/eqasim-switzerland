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
                       path=None, n_val_folds=5, val_every=5):
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

            loss_per_sample = F.cross_entropy(utilities, batch_y, reduction="none")
            weighted_loss   = loss_per_sample * batch_w
            loss = weighted_loss.sum() / batch_w.sum()

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            total_weighted_loss += weighted_loss.sum().item()
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
                v_loss  = ((F.cross_entropy(v_utils, v_y, reduction="none") * v_w).sum()
                           / v_w.sum().clamp_min(1e-12))
                val_loss = float(v_loss.item())

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