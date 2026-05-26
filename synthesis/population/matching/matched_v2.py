import logging
import importlib
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
	torch = importlib.import_module("torch")
	nn = torch.nn
	F = torch.nn.functional
except Exception:
	torch = None
	nn = None
	F = None

logger = logging.getLogger("synpp")


def configure(context):
	context.config("random_seed")
	context.config("matching_embedding_top_k", 15)
	context.config("matching_embedding_temperature", 0.3)
	context.config("matching_embedding_epochs", 100)
	context.config("matching_embedding_batch_size", 64)
	context.config("matching_embedding_lr", 1.0e-3)
	context.config("matching_embedding_dropout", 0.1)
	context.config("matching_embedding_weight_decay", 3.0e-3)
	context.config("matching_embedding_max_sequence_len", 12)
	context.config("matching_loss_weight_activity", 1.0)
	context.config("matching_loss_weight_trip_count", 1.0)
	context.config("matching_loss_weight_mode", 0.2)
	context.config("matching_loss_weight_departure", 0.2)
	context.config("matching_loss_weight_reconstruction", 0.3)
	context.config("matching_loss_weight_latent_norm", 0.05)
	context.config("matching_loss_weight_latent_spread", 0.15)
	context.config("matching_embedding_geo_proj_dim", 8)
	context.config("matching_embedding_cars_proj_dim", 4)
	context.config("matching_embedding_emp_proj_dim", 4)
	context.config("matching_embedding_misc_proj_dim", 8)
	context.config("matching_use_class_weighting", True)
	context.config("matching_trip_count_scale", 5.0)
	context.config("specific_day_scenario", default="workday")

	context.stage("data.microcensus.persons")
	context.stage("data.microcensus.trips")
	context.stage("data.microcensus.activity_chains")
	context.stage("synthesis.population.sampled")
	context.stage("synthesis.population.spatial.primary.work.work_locations", alias="work_locations")
	context.stage("data.constants")


def _l2_normalize(x):
	norms = np.linalg.norm(x, axis=1, keepdims=True)
	return x / np.maximum(norms, 1e-12)


def _fit_feature_processors(df_source, df_population):
	cont_cols = ["age", "N_children_under_12", "commute_distance", "household_size"]
	cat_cols = [
		"canton_id", "sp_region", "municipality_type", "ovgk",
		"number_of_cars_class", "car_availability",
		"employed", "employment_status", "income_class", "sex",
		"work_location_type", "marital_status", "number_of_bikes_class"
	]

	scaler = StandardScaler()
	scaler.fit(df_source[cont_cols].astype(float).values)

	mappings = {}
	for col in cat_cols:
		merged = pd.concat([df_source[col], df_population[col]], axis=0)
		vals = merged.astype(str).fillna("<na>").unique().tolist()
		vals = sorted(vals)
		mappings[col] = {v: i + 1 for i, v in enumerate(vals)}

	return scaler, mappings, cont_cols


def _transform_model_inputs(df, scaler, mappings, cont_cols):
	out = {
		"cont": scaler.transform(df[cont_cols].astype(float).values).astype(np.float32),
	}

	for col, mapping in mappings.items():
		s = df[col].astype(str).fillna("<na>")
		out[col] = s.map(lambda v: mapping.get(v, 0)).astype(np.int64).values

	out["cardinalities"] = {k: len(v) + 1 for k, v in mappings.items()}  # +1 for unknown index 0
	return out


def _sample_knn(src_emb, src_w, tgt_emb, top_k, temperature, rng):
	n_neighbors = min(top_k, len(src_emb))
	if n_neighbors == 0:
		return np.full((len(tgt_emb),), -1, dtype=np.int64)

	knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="auto")
	knn.fit(src_emb)
	dists, idx = knn.kneighbors(tgt_emb, return_distance=True)

	out = np.empty((len(tgt_emb),), dtype=np.int64)
	tau = max(float(temperature), 1e-3)

	for i in range(len(tgt_emb)):
		sims = 1.0 - dists[i]
		cand = idx[i]
		logits = (sims - np.max(sims)) / tau
		probs = np.exp(logits) * np.maximum(src_w[cand], 1e-12)
		total = probs.sum()

		if not np.isfinite(total) or total <= 0.0:
			out[i] = cand[0]
			continue

		cdf = np.cumsum(probs / total)
		out[i] = cand[np.searchsorted(cdf, rng.rand(), side="right")]

	return out


if torch is not None:
	class EmbeddingGRUModel(nn.Module):
		def __init__(self, cont_dim, cardinalities, n_activity, n_mode, dropout=0.2,
					 geo_proj_dim=10, cars_proj_dim=4, emp_proj_dim=4, misc_proj_dim=8):
			super().__init__()
			self.card = cardinalities

			geo_dim = self.card["canton_id"] + self.card["sp_region"] + self.card["municipality_type"] + self.card["ovgk"]
			cars_dim = self.card["number_of_cars_class"] + self.card["car_availability"]
			emp_dim = self.card["employed"] + self.card["employment_status"] + self.card["work_location_type"]
			misc_dim = self.card["income_class"] + self.card["sex"] + self.card["marital_status"] + self.card["number_of_bikes_class"]

			self.geo_proj = nn.Sequential(nn.Linear(geo_dim, geo_proj_dim), nn.LayerNorm(geo_proj_dim), nn.ReLU(), nn.Dropout(dropout))
			self.cars_proj = nn.Sequential(nn.Linear(cars_dim, cars_proj_dim), nn.LayerNorm(cars_proj_dim), nn.ReLU(), nn.Dropout(dropout))
			self.emp_proj = nn.Sequential(nn.Linear(emp_dim, emp_proj_dim), nn.LayerNorm(emp_proj_dim), nn.ReLU(), nn.Dropout(dropout))
			self.misc_proj = nn.Sequential(nn.Linear(misc_dim, misc_proj_dim), nn.LayerNorm(misc_proj_dim), nn.ReLU(), nn.Dropout(dropout))

			encoder_input_dim = cont_dim + geo_proj_dim + cars_proj_dim + emp_proj_dim + misc_proj_dim
			self.encoder = nn.Sequential(
				nn.Linear(encoder_input_dim, 32),
				nn.LayerNorm(32),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(32, 16),
				nn.LayerNorm(16),
			)
			# Autonomous decoder: recurrent dynamics start from h0 only.
			self.gru = nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)

			self.activity_head = nn.Linear(16, n_activity)
			self.mode_head = nn.Linear(16, n_mode)
			self.departure_head = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
			self.trip_count_head = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

			# Reconstruction head to keep latent informative (autoencoder-like regularization).
			self.reconstruction = nn.Sequential(
				nn.Linear(16, 32),
				nn.LayerNorm(32),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(32, encoder_input_dim),
			)

		def _one_hot(self, idx, depth):
			return F.one_hot(idx.clamp(min=0, max=depth - 1), num_classes=depth).float()

		def _build_encoder_input(self, x):
			geo = torch.cat([
				self._one_hot(x["canton_id"], self.card["canton_id"]),
				self._one_hot(x["sp_region"], self.card["sp_region"]),
				self._one_hot(x["municipality_type"], self.card["municipality_type"]),
				self._one_hot(x["ovgk"], self.card["ovgk"]),
			], dim=1)

			cars = torch.cat([
				self._one_hot(x["number_of_cars_class"], self.card["number_of_cars_class"]),
				self._one_hot(x["car_availability"], self.card["car_availability"]),
			], dim=1)

			emp = torch.cat([
				self._one_hot(x["employed"], self.card["employed"]),
				self._one_hot(x["employment_status"], self.card["employment_status"]),
				self._one_hot(x["work_location_type"], self.card["work_location_type"]),
			], dim=1)

			misc = torch.cat([
				self._one_hot(x["income_class"], self.card["income_class"]),
				self._one_hot(x["sex"], self.card["sex"]),
				self._one_hot(x["marital_status"], self.card["marital_status"]),
				self._one_hot(x["number_of_bikes_class"], self.card["number_of_bikes_class"]),
			], dim=1)

			x_all = torch.cat([
				x["cont"],
				self.geo_proj(geo),
				self.cars_proj(cars),
				self.emp_proj(emp),
				self.misc_proj(misc),
			], dim=1)
			return x_all

		def encode(self, x):
			x_all = self._build_encoder_input(x)
			return self.encoder(x_all)

		def forward(self, x, seq_len):
			x_all = self._build_encoder_input(x)
			z = self.encoder(x_all)
			h0 = z.unsqueeze(0)
			seq_in = torch.zeros((z.shape[0], int(seq_len), 1), dtype=z.dtype, device=z.device)
			h, _ = self.gru(seq_in, h0)
			act_logits = self.activity_head(h)
			mode_logits = self.mode_head(h)
			dep_pred = self.departure_head(h).squeeze(-1)
			trip_pred = self.trip_count_head(z).squeeze(-1)
			x_recon = self.reconstruction(z)
			return z, act_logits, mode_logits, dep_pred, trip_pred, x_recon, x_all
else:
	class EmbeddingGRUModel:
		def __init__(self, *args, **kwargs):
			raise RuntimeError("PyTorch is required for matched_v2. Install torch in this environment.")


def train_embedding_model(
	x,
	y,
	random_seed=0,
	epochs=25,
	batch_size=256,
	lr=1.0e-3,
	dropout=0.2,
	weight_decay=1.0e-4,
	loss_weights=None,
	proj_dims=None,
	use_class_weighting=True,
):
	"""
	Train the embedding model from static features x and sequence targets y.
	Returns a trained torch model.
	"""
	if torch is None:
		raise RuntimeError("PyTorch is required for matched_v2. Install torch in this environment.")

	if loss_weights is None:
		loss_weights = {
			"activity": 1.0,
			"trip_count": 1.1,
			"mode": 0.2,
			"departure": 0.5,
			"reconstruction": 0.3,
			"latent_norm": 0.01,
			"latent_spread": 0.15,
		}

	torch.manual_seed(random_seed)
	np.random.seed(random_seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cardinalities = x["cardinalities"]
	if proj_dims is None:
		proj_dims = {"geo": 10, "cars": 4, "emp": 4, "misc": 8}
	model = EmbeddingGRUModel(
		cont_dim=x["cont"].shape[1],
		cardinalities=cardinalities,
		n_activity=int(y["num_activity_classes"]),
		n_mode=int(y["num_mode_classes"]),
		dropout=float(dropout),
		geo_proj_dim=int(proj_dims["geo"]),
		cars_proj_dim=int(proj_dims["cars"]),
		emp_proj_dim=int(proj_dims["emp"]),
		misc_proj_dim=int(proj_dims["misc"]),
	).to(device)

	def _one_hot_masked(target_idx, num_classes):
		valid = (target_idx >= 0).float()
		t_safe = torch.clamp(target_idx, min=0)
		oh = F.one_hot(t_safe, num_classes=num_classes).float()
		oh = oh * valid.unsqueeze(-1)
		return oh, valid

	x_t = {}
	for k, arr in x.items():
		if k == "cardinalities":
			continue
		dtype = torch.float32 if k == "cont" else torch.long
		x_t[k] = torch.tensor(arr, dtype=dtype, device=device)

	act_tgt = torch.tensor(y["act_tgt"], dtype=torch.long, device=device)
	mode_tgt = torch.tensor(y["mode_tgt"], dtype=torch.long, device=device)
	dep_tgt = torch.tensor(y["dep_tgt"], dtype=torch.float32, device=device)
	dep_mask = torch.tensor(y["dep_mask"], dtype=torch.float32, device=device)
	trip_count_tgt = torch.tensor(y["trip_count_tgt"], dtype=torch.float32, device=device)
	sample_w = torch.tensor(y["sample_weight"], dtype=torch.float32, device=device)

	act_weight = None
	mode_weight = None
	if use_class_weighting:
		with torch.no_grad():
			flat_a = act_tgt.reshape(-1)
			flat_m = mode_tgt.reshape(-1)

			valid_a = flat_a[flat_a >= 0]
			valid_m = flat_m[flat_m >= 0]

			n_a = int(y["num_activity_classes"])
			n_m = int(y["num_mode_classes"])

			if len(valid_a) > 0:
				counts_a = torch.bincount(valid_a, minlength=n_a).float()
				counts_a = torch.clamp(counts_a, min=1.0)
				act_weight = (counts_a.sum() / counts_a) / n_a
				act_weight[0] = 0.0
				act_weight = act_weight.to(device)

			if len(valid_m) > 0:
				counts_m = torch.bincount(valid_m, minlength=n_m).float()
				counts_m = torch.clamp(counts_m, min=1.0)
				mode_weight = (counts_m.sum() / counts_m) / n_m
				mode_weight[0] = 0.0
				mode_weight = mode_weight.to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
	n = x_t["cont"].shape[0]
	epochs = max(int(epochs), 1)
	batch_size = max(int(batch_size), 1)

	for epoch in range(epochs):
		model.train()
		perm = np.random.permutation(n)
		total_loss = 0.0

		for i in range(0, n, batch_size):
			idx = torch.tensor(perm[i:i + batch_size], dtype=torch.long, device=device)
			x_batch = {k: v[idx] for k, v in x_t.items()}
			w_batch = sample_w[idx]
			w_norm = w_batch / torch.clamp(w_batch.mean(), min=1e-6)
			seq_len = act_tgt[idx].shape[1]

			z, act_logits, mode_logits, dep_pred, trip_pred, x_recon, x_all = model(
				x_batch, seq_len
			)

			act_oh, act_mask = _one_hot_masked(act_tgt[idx], act_logits.shape[-1])
			act_logp = F.log_softmax(act_logits, dim=-1)
			act_nll = -(act_oh * act_logp).sum(dim=-1)
			if act_weight is not None:
				act_class_w = torch.sum(act_oh * act_weight.view(1, 1, -1), dim=-1)
				act_nll = act_nll * act_class_w
			act_per_person = (act_nll * act_mask).sum(dim=1) / torch.clamp(act_mask.sum(dim=1), min=1.0)
			act_loss = (act_per_person * w_norm).mean()

			mode_oh, mode_mask = _one_hot_masked(mode_tgt[idx], mode_logits.shape[-1])
			mode_logp = F.log_softmax(mode_logits, dim=-1)
			mode_nll = -(mode_oh * mode_logp).sum(dim=-1)
			if mode_weight is not None:
				mode_class_w = torch.sum(mode_oh * mode_weight.view(1, 1, -1), dim=-1)
				mode_nll = mode_nll * mode_class_w
			mode_per_person = (mode_nll * mode_mask).sum(dim=1) / torch.clamp(mode_mask.sum(dim=1), min=1.0)
			mode_loss = (mode_per_person * w_norm).mean()

			dep_sq = ((dep_pred - dep_tgt[idx]) ** 2) * dep_mask[idx]
			dep_per_person = dep_sq.sum(dim=1) / torch.clamp(dep_mask[idx].sum(dim=1), min=1.0)
			dep_mse = (dep_per_person * w_norm).mean()

			trip_count_per_person = (trip_pred - trip_count_tgt[idx]) ** 2
			trip_count_mse = (trip_count_per_person * w_norm).mean()

			recon_per_person = ((x_recon - x_all) ** 2).mean(dim=1)
			recon_loss = (recon_per_person * w_norm).mean()
			latent_norm = (((z ** 2).mean(dim=1)) * w_norm).mean()
			latent_spread = F.relu(0.5 - z.var(dim=0).mean())

			loss = (
				float(loss_weights["activity"]) * act_loss
				+ float(loss_weights["trip_count"]) * trip_count_mse
				+ float(loss_weights["mode"]) * mode_loss
				+ float(loss_weights["departure"]) * dep_mse
				+ float(loss_weights["reconstruction"]) * recon_loss
				+ float(loss_weights["latent_norm"]) * latent_norm
				+ float(loss_weights["latent_spread"]) * latent_spread
			)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += float(loss.item()) * len(idx)

		if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
			logger.info("Embedding model epoch %d/%d - loss %.4f", epoch + 1, epochs, total_loss / n)

	model.eval()
	return model


def _encode_with_model(model, x, batch_size=4096):
	device = next(model.parameters()).device
	x_t = {}
	for k, arr in x.items():
		if k == "cardinalities":
			continue
		dtype = torch.float32 if k == "cont" else torch.long
		x_t[k] = torch.tensor(arr, dtype=dtype, device=device)
	n = x_t["cont"].shape[0]
	out = []
	with torch.no_grad():
		for i in range(0, n, batch_size):
			batch = {k: v[i:i + batch_size] for k, v in x_t.items()}
			z = model.encode(batch).cpu().numpy()
			out.append(z)
	return _l2_normalize(np.vstack(out))


def get_mz_persons(context):
	df_persons = context.stage("data.microcensus.persons")
	df_trips = context.stage("data.microcensus.trips")[0]

	employed_persons = set(df_persons[df_persons["employed"] == True]["person_id"])
	persons_with_work = set(df_trips[(df_trips["origin_purpose"] == "work") | (df_trips["purpose"] == "work")]["person_id"])
	persons_to_remove = persons_with_work - employed_persons

	logger.info("Removing %d inconsistent MZ persons (non-employed with work purpose).", len(persons_to_remove))
	return df_persons[~df_persons["person_id"].isin(persons_to_remove)].copy()


def _prepare_common_features(df_source, df_population, const):
	for df in (df_source, df_population):
		if "municipality_type" not in df.columns:
			df["municipality_type"] = "unknown"
		df["municipality_type"] = df["municipality_type"].astype(str).str.lower().fillna("unknown")

		if "sex" in df.columns:
			df["sex"] = pd.to_numeric(df["sex"], errors="coerce").fillna(-1).astype(int)
		if "employed" in df.columns:
			df["employed"] = pd.to_numeric(df["employed"], errors="coerce").fillna(0)
			df["employed"] = (df["employed"] == 1).astype(int)
		if "sp_region" in df.columns:
			df["sp_region"] = pd.to_numeric(df["sp_region"], errors="coerce").fillna(-1)
		if "canton_id" in df.columns:
			df["canton_id"] = pd.to_numeric(df["canton_id"], errors="coerce").fillna(-1)

		if "ovgk" not in df.columns:
			df["ovgk"] = 0
		df["ovgk"] = (df["ovgk"].astype(str) != "None").astype(int)

		if "N_children_under_12" not in df.columns:
			if "N_children_under_18" in df.columns:
				df["N_children_under_12"] = pd.to_numeric(df["N_children_under_18"], errors="coerce").fillna(0)
			else:
				df["N_children_under_12"] = 0
		df["N_children_under_12"] = pd.to_numeric(df["N_children_under_12"], errors="coerce").fillna(0)

		if "income_class" not in df.columns:
			df["income_class"] = -1
		df["income_class"] = pd.to_numeric(df["income_class"], errors="coerce").fillna(-1)

		if "number_of_cars_class" not in df.columns:
			df["number_of_cars_class"] = 0
		df["number_of_cars_class"] = pd.to_numeric(df["number_of_cars_class"], errors="coerce").fillna(0)

		if "household_size" not in df.columns:
			df["household_size"] = 0
		df["household_size"] = pd.to_numeric(df["household_size"], errors="coerce").fillna(0)

		if "employment_status" not in df.columns:
			df["employment_status"] = 0
		df["employment_status"] = pd.to_numeric(df["employment_status"], errors="coerce").fillna(0)

		if "marital_status" not in df.columns:
			df["marital_status"] = 0
		df["marital_status"] = pd.to_numeric(df["marital_status"], errors="coerce").fillna(0)

		if "number_of_bikes_class" not in df.columns:
			df["number_of_bikes_class"] = 0
		df["number_of_bikes_class"] = pd.to_numeric(df["number_of_bikes_class"], errors="coerce").fillna(0)

		if "commute_distance" not in df.columns:
			df["commute_distance"] = -1.0
		df["commute_distance"] = pd.to_numeric(df["commute_distance"], errors="coerce").fillna(-1.0)
		df.loc[df["employed"] != 1, "commute_distance"] = -1.0

		if "work_location_type" not in df.columns:
			df["work_location_type"] = "none"
		work_map = {"none": 0, "fixed": 1, "remote": 2, "moving": 3}
		df["work_location_type"] = df["work_location_type"].map(work_map).fillna(0).astype(int)

		if "car_availability" not in df.columns:
			df["car_availability"] = 0
		car_raw = pd.to_numeric(df["car_availability"], errors="coerce")
		df["car_availability"] = np.where(car_raw == const.CAR_AVAILABILITY_NEVER, 0, 1).astype(int)

	if "age" not in df_population.columns and "age_class" in df_population.columns:
		df_population["age"] = pd.to_numeric(df_population["age_class"], errors="coerce").fillna(0) * 10.0
	if "age" not in df_source.columns and "age_class" in df_source.columns:
		df_source["age"] = pd.to_numeric(df_source["age_class"], errors="coerce").fillna(0) * 10.0

	df_source["age"] = pd.to_numeric(df_source["age"], errors="coerce").fillna(0)
	df_population["age"] = pd.to_numeric(df_population["age"], errors="coerce").fillna(0)

	num_features = [
		"age", "N_children_under_12", "commute_distance", "income_class",
		"sp_region", "canton_id", "employment_status", "number_of_cars_class", "household_size"
	]
	cat_features = [
		"sex", "employed", "car_availability", "ovgk", "municipality_type", "work_location_type"
	]

	all_features = [c for c in num_features + cat_features if c in df_source.columns and c in df_population.columns]
	num_features = [c for c in num_features if c in all_features]
	cat_features = [c for c in cat_features if c in all_features]

	return df_source, df_population, all_features, num_features, cat_features


def _build_sequence_targets(source_ids, df_trips, max_sequence_len=8, trip_count_scale=10.0, sample_weights=None):
	seq_map = {}
	for pid, g in df_trips.groupby("person_id"):
		gs = g.sort_values("trip_id")
		acts = gs["purpose"].fillna("home").astype(str).tolist()
		modes = gs["mode"].fillna("no trip").astype(str).tolist()
		deps = pd.to_numeric(gs["departure_time"], errors="coerce").fillna(0.0).tolist()
		seq_map[int(pid)] = (acts, modes, deps)

	seqs = []
	trip_counts_raw = []
	activity_values = set()
	mode_values = set()
	max_len = 1

	for pid in source_ids:
		if int(pid) in seq_map:
			acts, modes, deps = seq_map[int(pid)]
			trip_counts_raw.append(len(acts))
		else:
			acts, modes, deps = (["home"], ["no trip"], [0.0])
			trip_counts_raw.append(0)
		max_len = max(max_len, len(acts))
		activity_values.update(acts)
		mode_values.update(modes)
		seqs.append((acts, modes, deps))

	activity_tokens = ["<pad>", "<bos>"] + sorted(activity_values)
	mode_tokens = ["<pad>", "<bos>"] + sorted(mode_values)
	act2idx = {k: i for i, k in enumerate(activity_tokens)}
	mode2idx = {k: i for i, k in enumerate(mode_tokens)}

	n = len(seqs)
	t = max(1, min(max_len, int(max_sequence_len)))
	act_tgt = np.full((n, t), -100, dtype=np.int64)
	mode_tgt = np.full((n, t), -100, dtype=np.int64)
	dep_tgt = np.zeros((n, t), dtype=np.float32)
	dep_mask = np.zeros((n, t), dtype=np.float32)

	dep_scale = 30.0 * 3600.0

	for i, (acts, modes, deps) in enumerate(seqs):
		l = len(acts)
		dep_norm = np.clip(np.array(deps, dtype=np.float32) / dep_scale, 0.0, 1.0)
		for k in range(min(l, t)):
			act_tgt[i, k] = act2idx[acts[k]]
			mode_tgt[i, k] = mode2idx[modes[k]]
			dep_tgt[i, k] = dep_norm[k]
			dep_mask[i, k] = 1.0

	trip_scale = max(float(trip_count_scale), 1.0)
	trip_count_tgt = np.clip(np.array(trip_counts_raw, dtype=np.float32) / trip_scale, 0.0, 1.0)

	if sample_weights is None:
		sample_weight = np.ones((n,), dtype=np.float32)
	else:
		sample_weight = np.array(sample_weights, dtype=np.float32)
		sample_weight = np.where(np.isfinite(sample_weight), sample_weight, 0.0)
		sample_weight = np.maximum(sample_weight, 1e-6)
		mean_w = float(np.mean(sample_weight))
		if mean_w > 0:
			sample_weight = sample_weight / mean_w
		else:
			sample_weight = np.ones((n,), dtype=np.float32)

	return {
		"act_tgt": act_tgt,
		"mode_tgt": mode_tgt,
		"dep_tgt": dep_tgt,
		"dep_mask": dep_mask,
		"trip_count_tgt": trip_count_tgt,
		"sample_weight": sample_weight,
		"num_activity_classes": len(activity_tokens),
		"num_mode_classes": len(mode_tokens),
	}


def execute(context):
	const = context.stage("data.constants")
	random_seed = int(context.config("random_seed"))
	rng = np.random.RandomState(random_seed)

	top_k = int(context.config("matching_embedding_top_k"))
	temperature = float(context.config("matching_embedding_temperature"))
	epochs = int(context.config("matching_embedding_epochs"))
	batch_size = int(context.config("matching_embedding_batch_size"))
	lr = float(context.config("matching_embedding_lr"))
	dropout = float(context.config("matching_embedding_dropout"))
	weight_decay = float(context.config("matching_embedding_weight_decay"))
	max_sequence_len = int(context.config("matching_embedding_max_sequence_len"))
	loss_weights = {
		"activity": float(context.config("matching_loss_weight_activity")),
		"trip_count": float(context.config("matching_loss_weight_trip_count")),
		"mode": float(context.config("matching_loss_weight_mode")),
		"departure": float(context.config("matching_loss_weight_departure")),
		"reconstruction": float(context.config("matching_loss_weight_reconstruction")),
		"latent_norm": float(context.config("matching_loss_weight_latent_norm")),
		"latent_spread": float(context.config("matching_loss_weight_latent_spread")),
	}
	trip_count_scale = float(context.config("matching_trip_count_scale"))
	proj_dims = {
		"geo": int(context.config("matching_embedding_geo_proj_dim")),
		"cars": int(context.config("matching_embedding_cars_proj_dim")),
		"emp": int(context.config("matching_embedding_emp_proj_dim")),
		"misc": int(context.config("matching_embedding_misc_proj_dim")),
	}
	use_class_weighting = bool(context.config("matching_use_class_weighting"))
	scenario_day = context.config("specific_day_scenario")

	df_mz = get_mz_persons(context)
	df_source = df_mz.copy()
	df_source["person_weight"] = df_source["person_weight"].fillna(df_source["person_weight"].mean()).astype(float)

	if scenario_day == "workday":
		df_source = df_source[df_source["workday"]]
	elif scenario_day == "weekend":
		df_source = df_source[df_source["weekend"]]
	elif scenario_day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
		df_source = df_source[df_source["day"] == scenario_day]
	else:
		raise ValueError(f"Unimplemented day for scenario: {scenario_day}")

	if len(df_source) == 0:
		logger.warning("No source observations for day '%s'. Returning unmatched ids.", scenario_day)
		df_population = context.stage("synthesis.population.sampled")[["person_id"]].copy()
		df_population["mz_person_id"] = -1
		return df_population, set()

	df_source = df_source.rename(columns={"person_id": "mz_id"}).copy()

	df_population = context.stage("synthesis.population.sampled").copy()
	df_work = context.stage("work_locations")[["person_id", "work_location_type", "commute_distance"]]
	df_population = df_population.merge(df_work, on="person_id", how="left")
	df_population["commute_distance"] = pd.to_numeric(df_population["commute_distance"], errors="coerce").fillna(-1)
	df_population["work_location_type"] = df_population["work_location_type"].fillna("none")

	if "is_student" not in df_population.columns:
		df_population["is_student"] = 0

	df_population.loc[:, "employment_status"] = 0
	df_population.loc[df_population["employed"] == 1, "employment_status"] = 1
	df_population.loc[(df_population["employed"] == 3) & (df_population["is_student"] == 1), "employment_status"] = 2
	df_population.loc[(df_population["employed"] == 2) & (df_population["is_student"] == 1), "employment_status"] = 2
	df_population.loc[(df_population["employed"] == 1) & (df_population["is_student"] == 1), "employment_status"] = 3

	df_source, df_population, features, num_features, cat_features = _prepare_common_features(df_source, df_population, const)
	logger.info("Embedding matching: grouped projections geo=%d cars=%d emp=%d misc=%d (class_weighting=%s).",
		proj_dims["geo"], proj_dims["cars"], proj_dims["emp"], proj_dims["misc"], use_class_weighting)

	scaler, mappings, cont_cols = _fit_feature_processors(df_source, df_population)
	x_source = _transform_model_inputs(df_source, scaler, mappings, cont_cols)

	trips_cols = ["person_id", "trip_id", "departure_time", "purpose", "mode"]
	df_trips = context.stage("data.microcensus.trips")[0][trips_cols].copy()
	y_source = _build_sequence_targets(
		df_source["mz_id"].values,
		df_trips,
		max_sequence_len=max_sequence_len,
		trip_count_scale=trip_count_scale,
		sample_weights=df_source["person_weight"].values,
	)

	model = train_embedding_model(
		x_source,
		y_source,
		random_seed=random_seed,
		epochs=epochs,
		batch_size=batch_size,
		lr=lr,
		dropout=dropout,
		weight_decay=weight_decay,
		loss_weights=loss_weights,
		proj_dims=proj_dims,
		use_class_weighting=use_class_weighting,
	)
	emb_source = _encode_with_model(model, x_source)

	source_weights = np.maximum(pd.to_numeric(df_source["person_weight"], errors="coerce").fillna(1e-6).values, 1e-6)

	if const.census == "statpop" and "age" in df_population.columns:
		eligible_selector = df_population["age"] >= const.MZ_AGE_THRESHOLD
	elif const.census == "are_synpop" and "age_class" in df_population.columns:
		eligible_selector = df_population["age_class"] > 0
	else:
		eligible_selector = np.ones((len(df_population),), dtype=bool)

	df_eligible = df_population.loc[eligible_selector].copy().reset_index(drop=True)
	x_target = _transform_model_inputs(df_eligible, scaler, mappings, cont_cols)
	emb_target = _encode_with_model(model, x_target)

	src_sex = df_source["sex"].values
	src_emp = df_source["employed"].values
	tgt_sex = df_eligible["sex"].values
	tgt_emp = df_eligible["employed"].values
	mz_ids = df_source["mz_id"].values

	assigned_mz = np.full((len(df_eligible),), -1, dtype=np.int64)
	group_keys = pd.DataFrame({"sex": tgt_sex, "employed": tgt_emp}).drop_duplicates().itertuples(index=False)

	for key in group_keys:
		tmask = (tgt_sex == key.sex) & (tgt_emp == key.employed)
		if not np.any(tmask):
			continue

		smask = (src_sex == key.sex) & (src_emp == key.employed)
		if not np.any(smask):
			logger.warning("No donor found for strict group sex=%s employed=%s; leaving %d agents unmatched.", key.sex, key.employed, int(np.sum(tmask)))
			continue

		src_local = np.where(smask)[0]
		tgt_local = np.where(tmask)[0]

		chosen_local = _sample_knn(
			emb_source[src_local],
			source_weights[src_local],
			emb_target[tgt_local],
			top_k=top_k,
			temperature=temperature,
			rng=rng,
		)
		assigned_mz[tgt_local] = mz_ids[src_local[chosen_local]]

	df_matching = df_population[["person_id"]].copy()
	df_matching["mz_person_id"] = -1

	df_assigned = pd.DataFrame({
		"person_id": df_eligible["person_id"].values,
		"mz_person_id": assigned_mz,
	})
	df_matching = df_matching.merge(df_assigned, on="person_id", how="left", suffixes=("", "_new"))
	df_matching["mz_person_id"] = pd.to_numeric(df_matching["mz_person_id_new"], errors="coerce").fillna(df_matching["mz_person_id"]).astype(int)
	del df_matching["mz_person_id_new"]

	if const.census == "statpop" and "age" in df_population.columns:
		unmatched = df_matching.loc[df_matching["mz_person_id"] == -1, "person_id"]
		if len(unmatched) > 0:
			assert np.all(df_population[df_population["person_id"].isin(unmatched)]["age"] < const.MZ_AGE_THRESHOLD)

	assert len(df_matching) == len(df_population)
	matched_eligible = int(np.sum(assigned_mz != -1))
	unmatched_eligible = int(np.sum(assigned_mz == -1))
	logger.info("PyTorch embedding matching done. Matched %d eligible persons; %d remained unmatched due to strict constraints.", matched_eligible, unmatched_eligible)

	return df_matching, set()
