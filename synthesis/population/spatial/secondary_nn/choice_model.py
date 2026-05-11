import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger("synpp: choice_model")

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_rate=0.1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.LayerNorm(dim_out)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        return self.act(x1 + x2)
    

class NeuralChoiceModel(nn.Module):
    def __init__(self, person_input_dim, candidate_input_dim, person_hidden_dim=32, hidden_dim=64, dropout_rate=0.1, mask_threshold=1e-4):
        super().__init__()
        self.person_input_dim = int(person_input_dim)
        self.candidate_input_dim = int(candidate_input_dim)
        self.person_hidden_dim = int(person_hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout_rate)
        self.mask_threshold = float(mask_threshold)
        self.device = torch.device("cpu")

        self.person_tower = ResidualBlock(person_input_dim, person_hidden_dim, dropout_rate=dropout_rate)
        self.utility_tower = ResidualBlock(person_hidden_dim + candidate_input_dim, hidden_dim, dropout_rate=dropout_rate)

        # Small mask head from the last hidden dimension: predicts candidate plausibility.
        self.mask_layer = nn.Linear(hidden_dim, 1)
        self.utility_head = nn.Linear(hidden_dim, 1)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.constant_(self.mask_layer.bias, 3.0)
        nn.init.normal_(self.mask_layer.weight, std=0.01)
        nn.init.normal_(self.utility_head.weight, std=0.05)
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

        # Apply internal hard mask only at inference time
        if not self.training:
            utilities = utilities.masked_fill(mask_prob < self.mask_threshold, -1e9)

        return utilities


def train_choice_model(model, person_static_x, person_dynamic_x, candidate_static_x, candidate_dynamic_x, y, valid_mask, epochs=50, batch_size=256, lr=1e-3, weight_decay=2e-3, num_threads=None, logger_instance=None, weights=None, grad_clip=5.0, lr_step_size=10, lr_gamma=0.5):
    lr_step_size = int(np.clip(lr_step_size, max(1, epochs // 6), max(1, epochs // 3)))
    if num_threads is not None:
        torch.set_num_threads(max(1, int(num_threads)))
    
    device = model.device
    model.to(device)

    person_static_tensor   = torch.as_tensor(person_static_x,   dtype=torch.float32, device=device)
    person_dynamic_tensor  = torch.as_tensor(person_dynamic_x,  dtype=torch.float32, device=device)
    candidate_static_tensor  = torch.as_tensor(candidate_static_x,  dtype=torch.float32, device=device)
    candidate_dynamic_tensor = torch.as_tensor(candidate_dynamic_x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
    mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device) if weights is not None else torch.ones(person_static_tensor.shape[0], device=device)

    n_samples = person_static_tensor.shape[0]
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    local_logger = logger_instance or logger
    local_logger.info("\tStarting training")
    t0 = time.time()

    for epoch in range(int(epochs)):
        model.train()
        permutation = torch.randperm(n_samples, device=device)

        total_weighted_loss = 0.0
        total_weight = 0.0

        for start in range(0, n_samples, int(batch_size)):
            end = min(start + int(batch_size), n_samples)
            idx = permutation[start:end]

            batch_person_static   = person_static_tensor[idx]
            batch_person_dynamic  = person_dynamic_tensor[idx]
            batch_candidate_static  = candidate_static_tensor if candidate_static_tensor.shape[0] == 1 else candidate_static_tensor[idx]
            batch_candidate_dynamic = candidate_dynamic_tensor[idx]
            batch_y = y_tensor[idx]
            batch_mask = mask_tensor[idx]
            batch_w = weights_tensor[idx]

            optimizer.zero_grad(set_to_none=True)
            utilities = model(batch_person_static, batch_person_dynamic, batch_candidate_static, batch_candidate_dynamic)
            utilities = utilities.masked_fill(~batch_mask, -1e9)

            loss_per_sample = F.cross_entropy(utilities, batch_y, reduction="none")
            weighted_loss = loss_per_sample * batch_w
            loss = weighted_loss.sum() / batch_w.sum()

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            total_weighted_loss += weighted_loss.sum().item()
            total_weight += batch_w.sum().item()

        train_loss = total_weighted_loss / max(total_weight, 1e-12)
        local_logger.info("\tEpoch %s/%s | Loss: %.4f | Time: %.1fmin", epoch + 1, epochs, train_loss, (time.time() - t0) / 60.0)
        scheduler.step()


def predict_choice_proba(model, person_static, person_dynamic, candidate_static, candidate_dynamic, valid_mask_tensor=None):
    with torch.inference_mode():
        dev = model.device
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