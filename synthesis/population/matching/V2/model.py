"""Person encoder and autonomous activity-sequence decoder for matching V2."""

import numpy as np

from .features import CATEGORICAL_GROUPS
from .runtime import torch, nn, F, to_tensors


if torch is not None:
    class EmbeddingGRUModel(nn.Module):
        def __init__(self, cont_dim, cardinalities, n_activity, n_mode, dropout=0.2,
                     geo_proj_dim=12, cars_proj_dim=6, emp_proj_dim=10, misc_proj_dim=8):
            super().__init__()
            self.card = cardinalities

            geo_dim, cars_dim, emp_dim, misc_dim = (
                sum(self.card[col] for col in CATEGORICAL_GROUPS[group])
                for group in ("geo", "cars", "emp", "misc"))

            self.geo_proj = nn.Sequential(nn.Linear(geo_dim, geo_proj_dim), nn.LayerNorm(geo_proj_dim), nn.SiLU())
            self.cars_proj = nn.Sequential(nn.Linear(cars_dim, cars_proj_dim), nn.LayerNorm(cars_proj_dim), nn.SiLU())
            self.emp_proj = nn.Sequential(nn.Linear(emp_dim, emp_proj_dim), nn.LayerNorm(emp_proj_dim), nn.SiLU())
            self.misc_proj = nn.Sequential(nn.Linear(misc_dim, misc_proj_dim), nn.LayerNorm(misc_proj_dim), nn.SiLU())

            encoder_input_dim = cont_dim + geo_proj_dim + cars_proj_dim + emp_proj_dim + misc_proj_dim
            self.encoder = nn.Sequential(
                nn.Linear(encoder_input_dim, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 16),
                nn.LayerNorm(16),
            )
            # Autonomous decoder: recurrent dynamics start from h0 only.
            self.gru = nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)

            self.activity_head = nn.Sequential(nn.Linear(16, 8), nn.LayerNorm(8), nn.SiLU(), nn.Linear(8, n_activity))
            self.mode_head = nn.Sequential(nn.Linear(16, 8), nn.LayerNorm(8), nn.SiLU(), nn.Linear(8, n_mode))
            self.departure_head = nn.Sequential(nn.Linear(16, 8), nn.LayerNorm(8), nn.SiLU(), nn.Linear(8, 1), nn.Sigmoid())
            self.distance_head = nn.Sequential(nn.Linear(16, 8), nn.LayerNorm(8), nn.SiLU(), nn.Linear(8, 1), nn.Softplus())
            self.duration_head = nn.Linear(16, 1)
            self.activity_duration_head = nn.Linear(16, 1)
            self.trip_count_head = nn.Sequential(nn.Linear(16, 8), nn.LayerNorm(8), nn.SiLU(), nn.Linear(8, 1), nn.Softplus())

            # Reconstruction head to keep latent informative (autoencoder-like regularization).
            self.reconstruction = nn.Sequential(
                nn.Linear(16, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(64, encoder_input_dim),
            )

        def _one_hot(self, idx, depth):
            return F.one_hot(idx.clamp(min=0, max=depth - 1), num_classes=depth).float()

        def _build_encoder_input(self, x):
            groups = {}
            for name, columns in CATEGORICAL_GROUPS.items():
                values = torch.cat([self._one_hot(x[col], self.card[col]) for col in columns], dim=1)
                groups[name] = getattr(self, name + "_proj")(values)
            return torch.cat([x["cont"]] + [groups[name] for name in ("geo", "cars", "emp", "misc")], dim=1)

        def encode(self, x):
            x_all = self._build_encoder_input(x)
            return self.encoder(x_all)

        def forward(self, x, seq_len):
            x_all = self._build_encoder_input(x)
            z = self.encoder(x_all)
            z_train = z
            if self.training and getattr(self, "latent_noise_std", 0.0) > 0.0:
                z_train = z + torch.randn_like(z) * float(self.latent_noise_std)
            h0 = z_train.unsqueeze(0)
            seq_in = torch.zeros((z.shape[0], int(seq_len), 1), dtype=z.dtype, device=z.device)
            h, _ = self.gru(seq_in, h0)
            act_logits = self.activity_head(h)
            mode_logits = self.mode_head(h)
            dep_pred = self.departure_head(h).squeeze(-1)
            dist_pred = self.distance_head(h).squeeze(-1)
            trip_pred = self.trip_count_head(z_train).squeeze(-1)
            x_recon = self.reconstruction(z_train)
            return {
                "latent": z_train, "activity": act_logits, "mode": mode_logits,
                "departure": dep_pred, "distance": dist_pred, "trip_count": trip_pred,
                "duration": F.softplus(self.duration_head(h).squeeze(-1)),
                "activity_duration": F.softplus(self.activity_duration_head(h).squeeze(-1)),
                "reconstruction": x_recon, "encoder_input": x_all,
            }
else:
    class EmbeddingGRUModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for matched_v2. Install torch in this environment.")



def encode(model, x, batch_size=4096):
    device = next(model.parameters()).device
    n = len(x["cont"])
    if n == 0:
        return np.empty((0, 16), dtype=np.float32)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = slice(start, start + batch_size)
            out.append(model.encode(to_tensors(x, idx, device)).cpu().numpy())
    embedding = np.vstack(out)
    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite V2 embeddings")
    return embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
