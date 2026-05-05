import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import pickle
import torch.nn.functional as F
import time

logger = logging.getLogger("synpp: NNModel")


def _torch_load_checkpoint(path, map_location=None):
    # PyTorch >=2.6 defaults to weights_only=True which blocks pickled sklearn objects.
    # Our artifacts are trusted local files and intentionally include scalers/transformers.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Backward compatibility with older torch versions without weights_only arg.
        return torch.load(path, map_location=map_location)

# Model
class MNLModel(nn.Module):
    def __init__(self, input_dim, num_h3, hidden_dims=(64, 32), dropout_rate=0.1):
        super().__init__()

        # Store config explicitly (IMPORTANT for saving)
        self.input_dim = input_dim
        self.num_h3 = num_h3
        self.hidden_dims = tuple(hidden_dims)
        self.dropout_rate = dropout_rate

        layers = []
        prev = input_dim

        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())

            # optional: only first layer dropout (as you had)
            if i == 0 and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev = h

        layers.append(nn.Linear(prev, 1))  # Utility output

        self.net = nn.Sequential(*layers)
        # Learned gating branch: predicts a soft feasibility mask in (0, 1).
        self.mask_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, num_h3, input_dim)
        batch, num_h3, _ = x.shape

        x = x.reshape(-1, x.shape[-1])          # (batch*num_h3, input_dim)
        utilities = self.net(x).reshape(batch, num_h3)

        # Convert learned mask to additive utility penalty.
        # This is equivalent to multiplying exp(utility) by mask_prob in softmax space.
        mask_prob = torch.sigmoid(self.mask_net(x).reshape(batch, num_h3))
        utilities = utilities + torch.log(mask_prob.clamp_min(1e-6))

        return utilities



def prune(model, threshold=1e-3):
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Skip biases & normalization layers
            if param.requires_grad and 'bias' not in name.lower() and 'norm' not in name.lower():
                param.data[param.data.abs() < threshold] = 0.0

# Training
def train_model(model, X, y, epochs=50, batch_size=512, lr=5e-3, weight_decay=1e-2,num_threads=None,
                weights=None, val_data=None, val_weights=None, patience=5, grad_clip=5.0,
                lr_step_size=10, lr_gamma=0.5):
    lr_step_size = int(np.clip(lr_step_size, max(1, epochs//6), epochs//3))  # Adjust step size based on total epochs

    if num_threads is not None:
        torch.set_num_threads(max(1, int(num_threads)))

    device = next(model.parameters()).device

    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)

    weights_tensor = (
        torch.as_tensor(weights, dtype=torch.float32, device=device)
        if weights is not None else torch.ones(len(X), device=device)
    )

    n_samples = X_tensor.shape[0]

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    logger.info("\tStarting training:")
    to = time.time()

    for epoch in range(epochs):
        model.train()

        permutation = torch.randperm(n_samples, device=device)

        total_weighted_loss = 0.0
        total_weight = 0.0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = permutation[start:end]

            batch_X = X_tensor[idx]
            batch_y = y_tensor[idx]
            batch_w = weights_tensor[idx]

            optimizer.zero_grad(set_to_none=True)

            utilities = model(batch_X)

            loss_per_sample = F.cross_entropy(
                utilities, batch_y, reduction="none"
            )

            # ---- CORRECT WEIGHTED LOSS ----
            weighted_loss = loss_per_sample * batch_w

            loss = weighted_loss.sum() / batch_w.sum()

            loss.backward()

            # ---- gradient clipping (important for stability) ----
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # ---- accumulate TRUE dataset loss ----
            total_weighted_loss += weighted_loss.sum().item()
            total_weight += batch_w.sum().item()

        train_loss = total_weighted_loss / total_weight

        # ---- optional: prune AFTER epoch
        if epoch%3 == 0:  # Prune every 3 epochs
            prune(model) 

        # ================= VALIDATION =================
        if val_data is not None:
            X_val, y_val = val_data

            val_loss = evaluate(
                model,
                X_val,
                y_val,
                weights=val_weights
            )

            logger.info(
                f"\tEpoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {(time.time() - to)/60:.1f}min"
            )

            # ---- early stopping ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("\tEarly stopping triggered.")
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break
        else:
            logger.info(f"\tEpoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Time: {(time.time() - to)/60:.1f}min")

        scheduler.step()


def train_with_mask(
    model,
    X,
    y,
    valid_mask,
    epochs=50,
    batch_size=512,
    lr=1e-3,
    weight_decay=2e-3,
    num_threads=None,
    logger_instance=None,
    weights=None,
    grad_clip=5.0,
    lr_step_size=10,
    lr_gamma=0.5,
):
    lr_step_size = int(np.clip(lr_step_size, max(1, epochs//6), epochs//3))  # Adjust step size based on total epochs
    if num_threads is not None:
        torch.set_num_threads(max(1, int(num_threads)))

    device = next(model.parameters()).device
    model.to(device)

    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
    mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    weights_tensor = (
        torch.as_tensor(weights, dtype=torch.float32, device=device)
        if weights is not None else torch.ones(len(X), device=device)
    )

    n_samples = X_tensor.shape[0]
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    local_logger = logger_instance or logger
    local_logger.info("\tStarting masked training:")
    to = time.time()

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_samples, device=device)

        total_weighted_loss = 0.0
        total_weight = 0.0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = permutation[start:end]

            batch_X = X_tensor[idx]
            batch_y = y_tensor[idx]
            batch_mask = mask_tensor[idx]
            batch_w = weights_tensor[idx]

            optimizer.zero_grad(set_to_none=True)
            utilities = model(batch_X)
            masked_utilities = utilities.masked_fill(~batch_mask, -1e9)

            loss_per_sample = F.cross_entropy(masked_utilities, batch_y, reduction="none")
            weighted_loss = loss_per_sample * batch_w
            loss = weighted_loss.sum() / batch_w.sum()

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_weighted_loss += weighted_loss.sum().item()
            total_weight += batch_w.sum().item()

        train_loss = total_weighted_loss / total_weight
        if epoch%3 == 0:  # Prune every 3 epochs
            prune(model) 

        local_logger.info(
            f"\tEpoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Time: {(time.time() - to)/60:.1f}min"
        )

        scheduler.step()



def evaluate(model, X, y, weights=None):
    device = next(model.parameters()).device

    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)

    model.eval()
    with torch.inference_mode():
        utilities = model(X_tensor)
        loss_per_sample = F.cross_entropy(utilities, y_tensor, reduction='none')

        if weights is not None:
            weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
            loss = (loss_per_sample * weights_tensor).sum() / weights_tensor.sum()
        else:
            loss = loss_per_sample.mean()

    return loss.item()



# Prediction
def predict_proba(model, X):
    device = next(model.parameters()).device
    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)

    model.eval()
    with torch.inference_mode():
        utilities = model(X_tensor)
        probs = torch.softmax(utilities - utilities.max(dim=1, keepdim=True).values, dim=1)

    return probs.cpu().numpy()



# Wrapper
class MNLWrapper:
    def __init__(self, model, scaler, numerical_cols, features, all_h3):
        self.model = model
        self.scaler = scaler
        self.numerical_cols = numerical_cols
        self.features = features
        self.all_h3 = all_h3

    def predict_utilities(self, df, already_scaled=False):
        df_scaled = df.copy()

        # Recreate missing one-hot purpose columns
        if "purpose" in df_scaled.columns:
            for col in self.features:
                if col.startswith("purpose_") and col not in df_scaled.columns:
                    val = col[len("purpose_"):]
                    df_scaled[col] = (df_scaled["purpose"].astype(str) == val).astype(np.float32)

        # Fill missing features
        for col in self.features:
            if col not in df_scaled.columns:
                df_scaled[col] = 0.0

        # Scale
        if not already_scaled:
            df_scaled[self.numerical_cols] = self.scaler.transform(df_scaled[self.numerical_cols])

        X = df_scaled[self.features].values.reshape(-1, self.model.num_h3, len(self.features))

        return predict_proba(self.model, X)

    def predict(self, df, already_scaled=False, max_utility = False):
        probs = self.predict_utilities(df, already_scaled)
        indices = np.array([np.random.choice(len(self.all_h3), p=probs[i]) for i in range(probs.shape[0])])
        return [self.all_h3[i] for i in indices]

    def predict_from_X(self, X, max_utility = False, rng=None, candidate_mask=None):
        probs = predict_proba(self.model, X)
        if candidate_mask is not None:
            mask = np.asarray(candidate_mask, dtype=bool)
            if mask.ndim == 1:
                mask = np.broadcast_to(mask[None, :], probs.shape)
            elif mask.shape != probs.shape:
                raise ValueError("candidate_mask shape must match probs or be 1D with num_h3 length")

            masked_probs = probs * mask
            row_sums = masked_probs.sum(axis=1, keepdims=True)
            empty_rows = row_sums[:, 0] <= 0.0
            if np.any(empty_rows):
                masked_probs[empty_rows] = probs[empty_rows]
                row_sums = masked_probs.sum(axis=1, keepdims=True)
            probs = masked_probs / np.clip(row_sums, 1e-12, None)

        chooser = np.random.choice if rng is None else rng.choice
        indices = np.array([chooser(len(self.all_h3), p=probs[i]) for i in range(probs.shape[0])])
        return [self.all_h3[i] for i in indices]

    def predict_frompandas(self, df, already_scaled=False, rng=None):
        probs = self.predict_utilities(df, already_scaled=already_scaled)
        if rng is None:
            indices = np.array([np.random.choice(len(self.all_h3), p=probs[i]) for i in range(probs.shape[0])])
        else:
            indices = np.array([rng.choice(len(self.all_h3), p=probs[i]) for i in range(probs.shape[0])])
        return [self.all_h3[i] for i in indices]

    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "model_config": {
                "input_dim": self.model.input_dim,
                "num_h3": self.model.num_h3,
                "hidden_dims": self.model.hidden_dims,
                "dropout_rate": self.model.dropout_rate,
            },
            "scaler": pickle.dumps(self.scaler),
            "numerical_cols": self.numerical_cols,
            "features": self.features,
            "all_h3": self.all_h3,
        }, path)

    @classmethod
    def load(cls, path, map_location=None):
        state = _torch_load_checkpoint(path, map_location=map_location)

        cfg = state["model_config"]

        model = MNLModel(
            input_dim=cfg["input_dim"],
            num_h3=cfg["num_h3"],
            hidden_dims=cfg["hidden_dims"],
            dropout_rate=cfg["dropout_rate"],
        )

        model.load_state_dict(state["model_state"])
        model.eval()

        scaler = pickle.loads(state["scaler"])

        return cls(
            model,
            scaler,
            state["numerical_cols"],
            state["features"],
            state["all_h3"],
        )


class MediumLevel1Wrapper:
    def __init__(
        self,
        model,
        scaler,
        numerical_cols,
        features,
        children_by_level0,
        level1_candidate_attributes_by_level0,
        purpose_categories,
    ):
        self.model = model
        self.scaler = scaler
        self.numerical_cols = numerical_cols
        self.features = features
        self.children_by_level0 = children_by_level0
        self.level1_candidate_attributes_by_level0 = level1_candidate_attributes_by_level0
        self.purpose_categories = [str(p) for p in purpose_categories]
        self.purpose_to_idx = {p: i for i, p in enumerate(self.purpose_categories)}

    def save(self, path):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "scaler": self.scaler,
                "numerical_cols": self.numerical_cols,
                "features": self.features,
                "input_dim": self.model.input_dim,
                "num_h3": self.model.num_h3,
                "hidden_dims": list(self.model.hidden_dims),
                "dropout_rate": self.model.dropout_rate,
                "children_by_level0": self.children_by_level0,
                "level1_candidate_attributes_by_level0": self.level1_candidate_attributes_by_level0,
                "purpose_categories": self.purpose_categories,
            },
            path,
        )

    def predict_from_X(self, X, valid_mask, max_utility=False, rng=None, return_probabilities=False):
        device = next(self.model.parameters()).device
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)

        self.model.eval()
        with torch.inference_mode():
            utilities = self.model(X_tensor)
            utilities = utilities.masked_fill(~mask_tensor, -1e9)
            probs = torch.softmax(utilities, dim=1).cpu().numpy()

        if rng is None:
            indices = np.array([np.random.choice(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])])
        else:
            indices = np.array([rng.choice(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])])

        if return_probabilities:
            return indices, probs
        return indices

    def list_level1_candidates(self, level0_h3):
        return list(self.children_by_level0.get(level0_h3, []))

    def predict_frompandas(self, df, rng=None):
        required = [
            "level0_h3", "home_x", "home_y", "work_x", "work_y", "origin_x", "origin_y", "age",
            "daily_longest_distance_from_home", "daily_crowfly_total", "crowfly_consumed_before_trip",
            "trip_position_class",
            "sex", "employed", "car_availability", "income_class", "purpose",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Missing required columns for MediumLevel1Wrapper.predict_frompandas: " + ", ".join(missing))

        out = []
        for row in df.itertuples(index=False):
            result = self.predict_level1(
                level0_h3=getattr(row, "level0_h3"),
                home_x=getattr(row, "home_x"),
                home_y=getattr(row, "home_y"),
                work_x=getattr(row, "work_x"),
                work_y=getattr(row, "work_y"),
                origin_x=getattr(row, "origin_x"),
                origin_y=getattr(row, "origin_y"),
                age=getattr(row, "age"),
                daily_longest_distance_from_home=getattr(row, "daily_longest_distance_from_home"),
                daily_crowfly_total=getattr(row, "daily_crowfly_total"),
                crowfly_consumed_before_trip=getattr(row, "crowfly_consumed_before_trip"),
                trip_position_class=getattr(row, "trip_position_class"),
                sex=getattr(row, "sex"),
                employed=getattr(row, "employed"),
                car_availability=getattr(row, "car_availability"),
                income_class=getattr(row, "income_class"),
                purpose=getattr(row, "purpose"),
                max_utility=False,
                rng=rng,
            )
            out.append(result["level1_h3"])
        return out

    def _purpose_one_hot(self, purpose):
        vec = np.zeros(len(self.purpose_categories), dtype=np.float32)
        idx = self.purpose_to_idx.get(str(purpose), 0)
        vec[idx] = 1.0
        return vec

    def predict_level1(
        self,
        level0_h3,
        home_x,
        home_y,
        work_x,
        work_y,
        origin_x,
        origin_y,
        age,
        daily_longest_distance_from_home,
        daily_crowfly_total,
        crowfly_consumed_before_trip,
        trip_position_class,
        sex,
        employed,
        car_availability,
        income_class,
        purpose,
        max_utility=False,
        rng=None,
        **kwargs,
    ):
        attrs = self.level1_candidate_attributes_by_level0.get(level0_h3)
        if attrs is None:
            raise KeyError(f"Unknown level0_h3 '{level0_h3}' in medium model candidates")

        children = attrs["children"]
        n_children = len(children)
        if n_children == 0:
            raise RuntimeError(f"No level1 candidates available for level0_h3 '{level0_h3}'")

        has_work = np.isfinite(work_x) and np.isfinite(work_y)
        safe_work_x = float(work_x) if has_work else 0.0
        safe_work_y = float(work_y) if has_work else 0.0
        safe_daily_longest = float(daily_longest_distance_from_home) if np.isfinite(daily_longest_distance_from_home) and float(daily_longest_distance_from_home) >= 0.0 else 0.0
        safe_daily_total = float(daily_crowfly_total) if np.isfinite(daily_crowfly_total) and float(daily_crowfly_total) >= 0.0 else 0.0
        safe_consumed = float(crowfly_consumed_before_trip) if np.isfinite(crowfly_consumed_before_trip) and float(crowfly_consumed_before_trip) >= 0.0 else 0.0
        safe_trip_position = float(trip_position_class) if np.isfinite(trip_position_class) else 2.0

        numerical = np.empty((n_children, len(self.numerical_cols)), dtype=np.float64)
        for j in range(n_children):
            cx = attrs["x"][j]
            cy = attrs["y"][j]

            dist_home = np.sqrt((cx - float(home_x)) ** 2 + (cy - float(home_y)) ** 2)
            dist_work = np.sqrt((cx - safe_work_x) ** 2 + (cy - safe_work_y) ** 2) if has_work else 0.0
            dist_last = np.sqrt((cx - float(origin_x)) ** 2 + (cy - float(origin_y)) ** 2)

            numerical[j, 0] = dist_home
            numerical[j, 1] = dist_work
            numerical[j, 2] = dist_last
            numerical[j, 3] = float(age)
            numerical[j, 4] = attrs["num_statent"][j]
            numerical[j, 5] = attrs["employees"][j]
            numerical[j, 6] = attrs["urban_core"][j]
            numerical[j, 7] = attrs["urban"][j]
            numerical[j, 8] = attrs["education"][j]
            numerical[j, 9] = attrs["shop"][j]
            numerical[j, 10] = attrs["leisure"][j]
            numerical[j, 11] = attrs["ovgk_share_a"][j]
            numerical[j, 12] = attrs["ovgk_share_b"][j]
            numerical[j, 13] = attrs["ovgk_share_c"][j]
            numerical[j, 14] = attrs["ovgk_share_d"][j]
            numerical[j, 15] = attrs["ovgk_share_none"][j]
            numerical[j, 16] = attrs["outside_fraction"][j]
            numerical[j, 17] = safe_daily_longest
            numerical[j, 18] = safe_daily_total
            numerical[j, 19] = safe_consumed
            numerical[j, 20] = safe_trip_position
            numerical[j, 21] = float(income_class)

        scaled_numerical = self.scaler.transform(numerical).astype(np.float32)

        X = np.zeros((1, self.model.num_h3, len(self.features)), dtype=np.float32)
        valid_mask = np.zeros((1, self.model.num_h3), dtype=bool)
        valid_mask[0, :n_children] = True

        X[0, :n_children, :len(self.numerical_cols)] = scaled_numerical
        X[0, :n_children, len(self.numerical_cols)] = float(sex)
        X[0, :n_children, len(self.numerical_cols) + 1] = float(employed)
        X[0, :n_children, len(self.numerical_cols) + 2] = float(car_availability)

        purpose_hot = self._purpose_one_hot(purpose)
        X[0, :n_children, len(self.numerical_cols) + 3:] = purpose_hot[None, :]

        device = next(self.model.parameters()).device
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)

        self.model.eval()
        with torch.inference_mode():
            utilities = self.model(X_tensor)
            utilities = utilities.masked_fill(~mask_tensor, -1e9)
            probs = torch.softmax(utilities, dim=1).cpu().numpy()[0, :n_children]

        company_mask = np.asarray(attrs["num_statent"], dtype=np.float64)[:n_children] > 0.0
        if np.any(company_mask):
            probs = probs * company_mask.astype(np.float64)
            probs = probs / np.clip(probs.sum(), 1e-12, None)

        if rng is None:
            chosen_idx = int(np.random.choice(n_children, p=probs))
        else:
            chosen_idx = int(rng.choice(n_children, p=probs))

        return {
            "level1_h3": children[chosen_idx],
            "choice_index": chosen_idx,
            "candidates": children,
            "probabilities": probs,
        }

    @classmethod
    def load(cls, path, map_location=None):
        state = _torch_load_checkpoint(path, map_location=map_location)

        required_keys = [
            "model_state",
            "scaler",
            "numerical_cols",
            "features",
            "input_dim",
            "num_h3",
            "hidden_dims",
            "dropout_rate",
            "children_by_level0",
            "level1_candidate_attributes_by_level0",
            "purpose_categories",
        ]
        missing = [k for k in required_keys if k not in state]
        if missing:
            raise RuntimeError(
                "Medium model artifact is missing required inference keys: " + ", ".join(missing)
            )

        model = MNLModel(
            input_dim=state["input_dim"],
            num_h3=state["num_h3"],
            hidden_dims=tuple(state["hidden_dims"]),
            dropout_rate=state["dropout_rate"],
        )
        model.load_state_dict(state["model_state"])
        model.eval()

        return cls(
            model=model,
            scaler=state["scaler"],
            numerical_cols=state["numerical_cols"],
            features=state["features"],
            children_by_level0=state["children_by_level0"],
            level1_candidate_attributes_by_level0=state["level1_candidate_attributes_by_level0"],
            purpose_categories=state["purpose_categories"],
        )


class DetailedLevel2Wrapper:
    def __init__(
        self,
        model,
        scaler,
        numerical_cols,
        features,
        children_by_level1,
        level2_candidate_attributes_by_level1,
        purpose_categories,
    ):
        self.model = model
        self.scaler = scaler
        self.numerical_cols = numerical_cols
        self.features = features
        self.children_by_level1 = children_by_level1
        self.level2_candidate_attributes_by_level1 = level2_candidate_attributes_by_level1
        self.purpose_categories = [str(p) for p in purpose_categories]
        self.purpose_to_idx = {p: i for i, p in enumerate(self.purpose_categories)}

    def save(self, path):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "scaler": self.scaler,
                "numerical_cols": self.numerical_cols,
                "features": self.features,
                "input_dim": self.model.input_dim,
                "num_h3": self.model.num_h3,
                "hidden_dims": list(self.model.hidden_dims),
                "dropout_rate": self.model.dropout_rate,
                "children_by_level1": self.children_by_level1,
                "level2_candidate_attributes_by_level1": self.level2_candidate_attributes_by_level1,
                "purpose_categories": self.purpose_categories,
            },
            path,
        )

    def predict_from_X(self, X, valid_mask, max_utility=False, rng=None, return_probabilities=False):
        device = next(self.model.parameters()).device
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)

        self.model.eval()
        with torch.inference_mode():
            utilities = self.model(X_tensor)
            utilities = utilities.masked_fill(~mask_tensor, -1e9)
            probs = torch.softmax(utilities, dim=1).cpu().numpy()

        if rng is None:
            indices = np.array([np.random.choice(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])])
        else:
            indices = np.array([rng.choice(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])])

        if return_probabilities:
            return indices, probs
        return indices

    def _purpose_one_hot(self, purpose):
        vec = np.zeros(len(self.purpose_categories), dtype=np.float32)
        idx = self.purpose_to_idx.get(str(purpose), 0)
        vec[idx] = 1.0
        return vec

    def predict_level2(
        self,
        level0_h3,
        level1_h3,
        home_x,
        home_y,
        work_x,
        work_y,
        origin_x,
        origin_y,
        age,
        daily_longest_distance_from_home,
        daily_crowfly_total,
        crowfly_consumed_before_trip,
        trip_position_class,
        sex,
        employed,
        car_availability,
        income_class,
        purpose,
        max_utility=False,
        rng=None,
        **kwargs,
    ):
        key = (level0_h3, level1_h3)
        attrs = self.level2_candidate_attributes_by_level1.get(key)
        if attrs is None:
            raise KeyError(f"Unknown key {key} in detailed model candidates")

        children = attrs["children"]
        n_children = len(children)
        if n_children == 0:
            raise RuntimeError(f"No level2 candidates available for key {key}")

        has_work = np.isfinite(work_x) and np.isfinite(work_y)
        safe_work_x = float(work_x) if has_work else 0.0
        safe_work_y = float(work_y) if has_work else 0.0
        safe_daily_longest = float(daily_longest_distance_from_home) if np.isfinite(daily_longest_distance_from_home) and float(daily_longest_distance_from_home) >= 0.0 else 0.0
        safe_daily_total = float(daily_crowfly_total) if np.isfinite(daily_crowfly_total) and float(daily_crowfly_total) >= 0.0 else 0.0
        safe_consumed = float(crowfly_consumed_before_trip) if np.isfinite(crowfly_consumed_before_trip) and float(crowfly_consumed_before_trip) >= 0.0 else 0.0
        safe_trip_position = float(trip_position_class) if np.isfinite(trip_position_class) else 2.0

        numerical = np.empty((n_children, len(self.numerical_cols)), dtype=np.float64)
        for j in range(n_children):
            cx = attrs["x"][j]
            cy = attrs["y"][j]

            dist_home = np.sqrt((cx - float(home_x)) ** 2 + (cy - float(home_y)) ** 2)
            dist_work = np.sqrt((cx - safe_work_x) ** 2 + (cy - safe_work_y) ** 2) if has_work else 0.0
            dist_last = np.sqrt((cx - float(origin_x)) ** 2 + (cy - float(origin_y)) ** 2)

            numerical[j, 0] = dist_home
            numerical[j, 1] = dist_work
            numerical[j, 2] = dist_last
            numerical[j, 3] = float(age)
            numerical[j, 4] = attrs["num_statent"][j]
            numerical[j, 5] = attrs["employees"][j]
            numerical[j, 6] = attrs["urban_core"][j]
            numerical[j, 7] = attrs["urban"][j]
            numerical[j, 8] = attrs["education"][j]
            numerical[j, 9] = attrs["shop"][j]
            numerical[j, 10] = attrs["leisure"][j]
            numerical[j, 11] = attrs["ovgk_share_a"][j]
            numerical[j, 12] = attrs["ovgk_share_b"][j]
            numerical[j, 13] = attrs["ovgk_share_c"][j]
            numerical[j, 14] = attrs["ovgk_share_d"][j]
            numerical[j, 15] = attrs["ovgk_share_none"][j]
            numerical[j, 16] = attrs["outside_fraction"][j]
            numerical[j, 17] = safe_daily_longest
            numerical[j, 18] = safe_daily_total
            numerical[j, 19] = safe_consumed
            numerical[j, 20] = safe_trip_position
            numerical[j, 21] = float(income_class)

        scaled_numerical = self.scaler.transform(numerical).astype(np.float32)
        X = np.zeros((1, self.model.num_h3, len(self.features)), dtype=np.float32)
        valid_mask = np.zeros((1, self.model.num_h3), dtype=bool)
        valid_mask[0, :n_children] = True

        X[0, :n_children, :len(self.numerical_cols)] = scaled_numerical
        X[0, :n_children, len(self.numerical_cols)] = float(sex)
        X[0, :n_children, len(self.numerical_cols) + 1] = float(employed)
        X[0, :n_children, len(self.numerical_cols) + 2] = float(car_availability)

        purpose_hot = self._purpose_one_hot(purpose)
        X[0, :n_children, len(self.numerical_cols) + 3:] = purpose_hot[None, :]

        indices, probs = self.predict_from_X(X, valid_mask, max_utility=max_utility, rng=rng, return_probabilities=True)
        company_mask = np.asarray(attrs["num_statent"], dtype=np.float64)[:n_children] > 0.0
        if np.any(company_mask):
            p = probs[0, :n_children] * company_mask.astype(np.float64)
            p = p / np.clip(p.sum(), 1e-12, None)
            probs[0, :n_children] = p
            if rng is None:
                chosen_idx = int(np.random.choice(n_children, p=p))
            else:
                chosen_idx = int(rng.choice(n_children, p=p))
        else:
            chosen_idx = int(indices[0])
        return {
            "level2_h3": children[chosen_idx],
            "choice_index": chosen_idx,
            "candidates": children,
            "probabilities": probs[0, :n_children],
        }

    def predict_frompandas(self, df, rng=None):
        required = [
            "level0_h3", "level1_h3", "home_x", "home_y", "work_x", "work_y", "origin_x", "origin_y", "age",
            "daily_longest_distance_from_home", "daily_crowfly_total", "crowfly_consumed_before_trip",
            "trip_position_class",
            "sex", "employed", "car_availability", "income_class", "purpose",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Missing required columns for DetailedLevel2Wrapper.predict_frompandas: " + ", ".join(missing))

        out = []
        for row in df.itertuples(index=False):
            result = self.predict_level2(
                level0_h3=getattr(row, "level0_h3"),
                level1_h3=getattr(row, "level1_h3"),
                home_x=getattr(row, "home_x"),
                home_y=getattr(row, "home_y"),
                work_x=getattr(row, "work_x"),
                work_y=getattr(row, "work_y"),
                origin_x=getattr(row, "origin_x"),
                origin_y=getattr(row, "origin_y"),
                age=getattr(row, "age"),
                daily_longest_distance_from_home=getattr(row, "daily_longest_distance_from_home"),
                daily_crowfly_total=getattr(row, "daily_crowfly_total"),
                crowfly_consumed_before_trip=getattr(row, "crowfly_consumed_before_trip"),
                trip_position_class=getattr(row, "trip_position_class"),
                sex=getattr(row, "sex"),
                employed=getattr(row, "employed"),
                car_availability=getattr(row, "car_availability"),
                income_class=getattr(row, "income_class"),
                purpose=getattr(row, "purpose"),
                max_utility=False,
                rng=rng,
            )
            out.append(result["level2_h3"])
        return out

    @classmethod
    def load(cls, path, map_location=None):
        state = _torch_load_checkpoint(path, map_location=map_location)

        required_keys = [
            "model_state",
            "scaler",
            "numerical_cols",
            "features",
            "input_dim",
            "num_h3",
            "hidden_dims",
            "dropout_rate",
            "children_by_level1",
            "purpose_categories",
        ]
        missing = [k for k in required_keys if k not in state]
        if missing:
            raise RuntimeError(
                "Detailed model artifact is missing required inference keys: " + ", ".join(missing)
            )

        model = MNLModel(
            input_dim=state["input_dim"],
            num_h3=state["num_h3"],
            hidden_dims=tuple(state["hidden_dims"]),
            dropout_rate=state["dropout_rate"],
        )
        model.load_state_dict(state["model_state"])
        model.eval()

        return cls(
            model=model,
            scaler=state["scaler"],
            numerical_cols=state["numerical_cols"],
            features=state["features"],
            children_by_level1=state["children_by_level1"],
            level2_candidate_attributes_by_level1=state.get("level2_candidate_attributes_by_level1", {}),
            purpose_categories=state["purpose_categories"],
        )