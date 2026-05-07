import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger("synpp: two_input_nn")


class TwoInputChoiceModel(nn.Module):
    def __init__(self, person_input_dim, candidate_input_dim, person_hidden_dim=32, hidden_dims=(64, 32), dropout_rate=0.1, mask_threshold=1e-3):
        super().__init__()
        self.person_input_dim = int(person_input_dim)
        self.candidate_input_dim = int(candidate_input_dim)
        self.person_hidden_dim = int(person_hidden_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.dropout_rate = float(dropout_rate)
        self.mask_threshold = float(mask_threshold)
        self.device = torch.device("cpu")

        if len(self.hidden_dims) == 0:
            raise ValueError("TwoInputChoiceModel requires at least one hidden layer")

        self.person_tower = nn.Sequential(
            nn.Linear(self.person_input_dim, self.person_hidden_dim),
            nn.LayerNorm(self.person_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity(),
        )

        utility_layers = []
        prev_dim = self.candidate_input_dim + self.person_hidden_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            utility_layers.append(nn.Linear(prev_dim, hidden_dim))
            utility_layers.append(nn.LayerNorm(hidden_dim))
            utility_layers.append(nn.ReLU())
            if i == 0 and self.dropout_rate > 0.0:
                utility_layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        self.utility_tower = nn.Sequential(*utility_layers)

        # Small mask head from the last hidden dimension: predicts candidate plausibility.
        self.mask_layer = nn.Linear(self.hidden_dims[-1], 1)
        self.utility_head = nn.Linear(self.hidden_dims[-1], 1)

    def forward(self, person_x, candidate_x):#, static_x=None):
        if person_x.ndim == 1:
            person_x = person_x.unsqueeze(0)
        if candidate_x.ndim == 2:
            candidate_x = candidate_x.unsqueeze(0)

        # if static_x is not None:
        #     if static_x.ndim == 2:
        #         static_x = static_x.unsqueeze(0)
        #     if static_x.shape[0] != candidate_x.shape[0] or static_x.shape[1] != candidate_x.shape[1]:
        #         raise RuntimeError(
        #             "TwoInputChoiceModel expected static_x batch/candidate dimensions to match candidate_x. "
        #             f"Got static_x {tuple(static_x.shape)} and candidate_x {tuple(candidate_x.shape)}."
        #         )
        #     # Keep candidate feature contract as [dynamic, static].
        #     candidate_x = torch.cat([candidate_x, static_x], dim=-1)

        if candidate_x.shape[-1] != self.candidate_input_dim:
            raise RuntimeError(
                "TwoInputChoiceModel expected candidate feature width "
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

        # Soft penalty in utility space; low-probability candidates are strongly down-weighted.
        utilities = base_utilities + torch.log(mask_prob.clamp_min(1e-8))

        # Hard mask for highly implausible candidates (numerical -inf equivalent for softmax).
        utilities = utilities.masked_fill(mask_prob < self.mask_threshold, -1e9)
        return utilities


def train_two_input_with_mask(model, person_x, candidate_x, y, valid_mask, epochs=50, batch_size=256, lr=1e-3, weight_decay=2e-3, num_threads=None, logger_instance=None, weights=None, grad_clip=5.0, lr_step_size=10, lr_gamma=0.5):
    lr_step_size = int(np.clip(lr_step_size, max(1, epochs // 6), max(1, epochs // 3)))
    if num_threads is not None:
        torch.set_num_threads(max(1, int(num_threads)))
    
    device = model.device
    model.to(device)

    person_tensor = torch.as_tensor(person_x, dtype=torch.float32, device=device)
    candidate_tensor = torch.as_tensor(candidate_x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
    mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device) if weights is not None else torch.ones(len(person_x), device=device)

    n_samples = person_tensor.shape[0]
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    local_logger = logger_instance or logger
    local_logger.info("\tStarting two-input training")
    t0 = time.time()

    for epoch in range(int(epochs)):
        model.train()
        permutation = torch.randperm(n_samples, device=device)

        total_weighted_loss = 0.0
        total_weight = 0.0

        for start in range(0, n_samples, int(batch_size)):
            end = min(start + int(batch_size), n_samples)
            idx = permutation[start:end]

            batch_person = person_tensor[idx]
            batch_candidate = candidate_tensor[idx]
            batch_y = y_tensor[idx]
            batch_mask = mask_tensor[idx]
            batch_w = weights_tensor[idx]

            optimizer.zero_grad(set_to_none=True)
            utilities = model(batch_person, batch_candidate)
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


# def predict_two_input_proba(model, person_x, candidate_x, valid_mask=None, static_x=None):
#     device = next(model.parameters()).device
#     person_tensor = torch.as_tensor(person_x, dtype=torch.float32, device=device)
#     candidate_tensor = torch.as_tensor(candidate_x, dtype=torch.float32, device=device)
#     static_tensor = None if static_x is None else torch.as_tensor(static_x, dtype=torch.float32, device=device)

#     model.eval()
#     with torch.inference_mode():
#         utilities = model(person_tensor, candidate_tensor, static_x=static_tensor)
#         if valid_mask is not None:
#             mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
#             utilities = utilities.masked_fill(~mask_tensor, -1e9)
#         probs = torch.softmax(utilities - utilities.max(dim=1, keepdim=True).values, dim=1)
#     return probs.cpu().numpy()

def predict_two_input_proba(model, person_tensor, candidate_tensor, valid_mask_tensor=None,static_tensor=None):
    with torch.inference_mode():        
        person_tensor = torch.as_tensor(person_tensor, dtype=torch.float32, device=model.device)
        candidate_tensor = torch.as_tensor(candidate_tensor, dtype=torch.float32, device=model.device)
        utilities = model(person_tensor,candidate_tensor)

        if valid_mask_tensor is not None:
            valid_mask_tensor = torch.as_tensor(valid_mask_tensor, dtype=torch.bool, device=model.device)
            if valid_mask_tensor.ndim == 1:
                valid_mask_tensor = valid_mask_tensor.unsqueeze(0).expand(utilities.shape[0], -1)
            utilities.masked_fill_(~valid_mask_tensor, -1e9)

        return torch.softmax(utilities, dim=1)