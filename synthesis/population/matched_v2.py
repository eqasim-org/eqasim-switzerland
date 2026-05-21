import logging
import importlib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
	context.config("matching_embedding_temperature", 0.25)
	context.config("matching_embedding_epochs", 25)
	context.config("matching_embedding_batch_size", 256)
	context.config("matching_embedding_lr", 1.0e-3)
	context.config("matching_embedding_dropout", 0.2)
	context.config("matching_embedding_weight_decay", 1.0e-4)
	context.config("specific_day_scenario", default="workday")

	context.stage("data.microcensus.persons")
	context.stage("data.microcensus.trips")
	context.stage("data.microcensus.activity_chains")
	context.stage("synthesis.population.sampled")
	context.stage("synthesis.population.spatial.primary.work.work_locations", alias="work_locations")
	context.stage("data.constants")


def _make_one_hot_encoder():
	try:
		return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	except TypeError:
		return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _l2_normalize(x):
	norms = np.linalg.norm(x, axis=1, keepdims=True)
	return x / np.maximum(norms, 1e-12)


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
		def __init__(self, input_dim, n_activity, n_mode, dropout=0.2):
			super().__init__()
			self.encoder = nn.Sequential(
				nn.Linear(input_dim, 32),
				nn.LayerNorm(32),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(32, 16),
				nn.LayerNorm(16),
			)
			self.activity_emb = nn.Embedding(n_activity, 8, padding_idx=0)
			self.mode_emb = nn.Embedding(n_mode, 8, padding_idx=0)
			self.gru = nn.GRU(input_size=17, hidden_size=16, num_layers=1, batch_first=True)

			self.activity_head = nn.Linear(16, n_activity)
			self.mode_head = nn.Linear(16, n_mode)
			self.departure_head = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

			# Reconstruction head to keep latent informative (autoencoder-like regularization).
			self.reconstruction = nn.Sequential(
				nn.Linear(16, 32),
				nn.LayerNorm(32),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(32, input_dim),
			)

		def encode(self, x):
			return self.encoder(x)

		def forward(self, x, act_in, mode_in, dep_in):
			z = self.encode(x)
			h0 = z.unsqueeze(0)
			seq_in = torch.cat([
				self.activity_emb(act_in),
				self.mode_emb(mode_in),
				dep_in.unsqueeze(-1),
			], dim=-1)
			h, _ = self.gru(seq_in, h0)
			act_logits = self.activity_head(h)
			mode_logits = self.mode_head(h)
			dep_pred = self.departure_head(h).squeeze(-1)
			x_recon = self.reconstruction(z)
			return z, act_logits, mode_logits, dep_pred, x_recon
else:
	class EmbeddingGRUModel:
		def __init__(self, *args, **kwargs):
			raise RuntimeError("PyTorch is required for matched_v2. Install torch in this environment.")


def train_embedding_model(x, y, random_seed=0, epochs=25, batch_size=256, lr=1.0e-3, dropout=0.2, weight_decay=1.0e-4):
	"""
	Train the embedding model from static features x and sequence targets y.
	Returns a trained torch model.
	"""
	if torch is None:
		raise RuntimeError("PyTorch is required for matched_v2. Install torch in this environment.")

	torch.manual_seed(random_seed)
	np.random.seed(random_seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = EmbeddingGRUModel(
		input_dim=x.shape[1],
		n_activity=int(y["num_activity_classes"]),
		n_mode=int(y["num_mode_classes"]),
		dropout=float(dropout),
	).to(device)

	x_t = torch.tensor(x, dtype=torch.float32, device=device)
	act_in = torch.tensor(y["act_in"], dtype=torch.long, device=device)
	mode_in = torch.tensor(y["mode_in"], dtype=torch.long, device=device)
	dep_in = torch.tensor(y["dep_in"], dtype=torch.float32, device=device)
	act_tgt = torch.tensor(y["act_tgt"], dtype=torch.long, device=device)
	mode_tgt = torch.tensor(y["mode_tgt"], dtype=torch.long, device=device)
	dep_tgt = torch.tensor(y["dep_tgt"], dtype=torch.float32, device=device)
	dep_mask = torch.tensor(y["dep_mask"], dtype=torch.float32, device=device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
	n = x_t.shape[0]
	epochs = max(int(epochs), 1)
	batch_size = max(int(batch_size), 1)

	for epoch in range(epochs):
		model.train()
		perm = np.random.permutation(n)
		total_loss = 0.0

		for i in range(0, n, batch_size):
			idx = torch.tensor(perm[i:i + batch_size], dtype=torch.long, device=device)

			z, act_logits, mode_logits, dep_pred, x_recon = model(
				x_t[idx], act_in[idx], mode_in[idx], dep_in[idx]
			)

			act_loss = F.cross_entropy(
				act_logits.reshape(-1, act_logits.shape[-1]),
				act_tgt[idx].reshape(-1),
				ignore_index=-100,
			)
			mode_loss = F.cross_entropy(
				mode_logits.reshape(-1, mode_logits.shape[-1]),
				mode_tgt[idx].reshape(-1),
				ignore_index=-100,
			)
			dep_mse = ((dep_pred - dep_tgt[idx]) ** 2 * dep_mask[idx]).sum() / dep_mask[idx].sum().clamp(min=1.0)

			recon_loss = F.mse_loss(x_recon, x_t[idx])
			latent_norm = (z ** 2).mean()
			latent_spread = F.relu(0.5 - z.var(dim=0).mean())

			loss = act_loss + mode_loss + 0.4 * dep_mse + 0.2 * recon_loss + 0.01 * latent_norm + 0.1 * latent_spread

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
	x_t = torch.tensor(x, dtype=torch.float32, device=device)
	out = []
	with torch.no_grad():
		for i in range(0, len(x_t), batch_size):
			z = model.encode(x_t[i:i + batch_size]).cpu().numpy()
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

		if "commute_distance" not in df.columns:
			df["commute_distance"] = -1.0
		df["commute_distance"] = pd.to_numeric(df["commute_distance"], errors="coerce").fillna(-1.0)

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


def _build_sequence_targets(source_ids, df_trips):
	seq_map = {}
	for pid, g in df_trips.groupby("person_id"):
		gs = g.sort_values("trip_id")
		acts = gs["purpose"].fillna("home").astype(str).tolist()
		modes = gs["mode"].fillna("no trip").astype(str).tolist()
		deps = pd.to_numeric(gs["departure_time"], errors="coerce").fillna(0.0).tolist()
		seq_map[int(pid)] = (acts, modes, deps)

	seqs = []
	activity_values = set()
	mode_values = set()
	max_len = 1

	for pid in source_ids:
		acts, modes, deps = seq_map.get(int(pid), (["home"], ["no trip"], [0.0]))
		max_len = max(max_len, len(acts))
		activity_values.update(acts)
		mode_values.update(modes)
		seqs.append((acts, modes, deps))

	activity_tokens = ["<pad>", "<bos>"] + sorted(activity_values)
	mode_tokens = ["<pad>", "<bos>"] + sorted(mode_values)
	act2idx = {k: i for i, k in enumerate(activity_tokens)}
	mode2idx = {k: i for i, k in enumerate(mode_tokens)}

	n = len(seqs)
	t = max_len
	act_in = np.zeros((n, t), dtype=np.int64)
	mode_in = np.zeros((n, t), dtype=np.int64)
	dep_in = np.zeros((n, t), dtype=np.float32)
	act_tgt = np.full((n, t), -100, dtype=np.int64)
	mode_tgt = np.full((n, t), -100, dtype=np.int64)
	dep_tgt = np.zeros((n, t), dtype=np.float32)
	dep_mask = np.zeros((n, t), dtype=np.float32)

	dep_scale = 30.0 * 3600.0
	act_bos = act2idx["<bos>"]
	mode_bos = mode2idx["<bos>"]

	for i, (acts, modes, deps) in enumerate(seqs):
		l = len(acts)
		dep_norm = np.clip(np.array(deps, dtype=np.float32) / dep_scale, 0.0, 1.0)
		for k in range(l):
			act_tgt[i, k] = act2idx[acts[k]]
			mode_tgt[i, k] = mode2idx[modes[k]]
			dep_tgt[i, k] = dep_norm[k]
			dep_mask[i, k] = 1.0

		act_in[i, 0] = act_bos
		mode_in[i, 0] = mode_bos
		dep_in[i, 0] = 0.0
		for k in range(1, t):
			prev = min(k - 1, l - 1)
			act_in[i, k] = act2idx[acts[prev]]
			mode_in[i, k] = mode2idx[modes[prev]]
			dep_in[i, k] = dep_norm[prev]

	return {
		"act_in": act_in,
		"mode_in": mode_in,
		"dep_in": dep_in,
		"act_tgt": act_tgt,
		"mode_tgt": mode_tgt,
		"dep_tgt": dep_tgt,
		"dep_mask": dep_mask,
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
	scenario_day = context.config("specific_day_scenario")

	df_mz = get_mz_persons(context)
	df_source = df_mz.copy()

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
	logger.info("Embedding matching: %d features (%d numeric, %d categorical)", len(features), len(num_features), len(cat_features))

	preprocessor = ColumnTransformer([
		("num", StandardScaler(), num_features),
		("cat", _make_one_hot_encoder(), cat_features),
	])

	x_source = preprocessor.fit_transform(df_source[features]).astype(np.float32)

	trips_cols = ["person_id", "trip_id", "departure_time", "purpose", "mode"]
	df_trips = context.stage("data.microcensus.trips")[0][trips_cols].copy()
	y_source = _build_sequence_targets(df_source["mz_id"].values, df_trips)

	model = train_embedding_model(
		x_source,
		y_source,
		random_seed=random_seed,
		epochs=epochs,
		batch_size=batch_size,
		lr=lr,
		dropout=dropout,
		weight_decay=weight_decay,
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
	x_target = preprocessor.transform(df_eligible[features]).astype(np.float32)
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
			smask = (src_sex == key.sex)
		if not np.any(smask):
			smask = np.ones((len(df_source),), dtype=bool)

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
	logger.info("PyTorch embedding matching done. Matched %d eligible persons.", len(df_assigned))

	return df_matching, set()
