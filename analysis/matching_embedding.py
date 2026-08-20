import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger("synpp")


def configure(context):
	context.config("analysis_path")
	context.config("matching_embedding_plot_method", "tsne")  # pca | tsne
	context.config("matching_embedding_plot_sample_size", 40000)
	context.config("matching_embedding_plot_alpha", 0.6)
	context.config("matching_embedding_plot_marker_size", 7)
	context.config("random_seed")
	context.config("matching_embedding_plot_top_categories", 12)
	context.config("matching_embedding_driver_min_group_size", 30)
	context.config("matching_embedding_driver_top_n", 12)
	context.config("matching_embedding_activity_top_k", 10)
	context.config("matching_embedding_activity_influence_min_group_size", 100)
	context.config("matching_embedding_commute_clip_m", 80000)

	context.stage("synthesis.population.matching.matched_v2")
	context.stage("data.microcensus.persons")
	context.stage("data.microcensus.activity_chains")
	context.stage("data.microcensus.trips")


def _reduce_to_2d(X, method="pca", random_seed=865836):
	method = str(method).lower().strip()
	if method == "tsne":
		reducer = TSNE(
			n_components=2,
			random_state=int(random_seed),
			init="pca",
			learning_rate="auto",
			perplexity=30,
		)
		return reducer.fit_transform(X)

	reducer = PCA(n_components=2, random_state=int(random_seed))
	return reducer.fit_transform(X)


def _prepare_labels(series, top_n=12):
	s = series.astype("string").fillna("<na>").astype(str)
	vc = s.value_counts(dropna=False)
	keep = set(vc.head(int(top_n)).index.tolist())
	return s.where(s.isin(keep), "other")


def _compose_cross_label(df, col_a, col_b):
	if col_a not in df.columns or col_b not in df.columns:
		return None
	a = df[col_a].astype("string").fillna("<na>").astype(str)
	b = df[col_b].astype("string").fillna("<na>").astype(str)
	return a + " | " + b


def _plot_scatter(df, x_col, y_col, group_col, out_path, alpha=0.6, marker_size=7, top_categories=12):
	if group_col not in df.columns:
		return

	g = _prepare_labels(df[group_col], top_n=top_categories)
	categories = sorted(g.unique().tolist())

	n = len(categories)
	cmap = plt.cm.get_cmap("tab20", max(n, 2))
	color_map = {cat: cmap(i) for i, cat in enumerate(categories)}

	fig, ax = plt.subplots(figsize=(9, 7), dpi=220)
	for cat in categories:
		mask = (g == cat).values
		if not np.any(mask):
			continue
		ax.scatter(
			df.loc[mask, x_col].values,
			df.loc[mask, y_col].values,
			s=float(marker_size),
			alpha=float(alpha),
			c=[color_map[cat]],
			label=f"{cat} (n={int(np.sum(mask))})",
			linewidths=0,
		)

	ax.set_title(f"MZ embeddings 2D colored by {group_col}")
	ax.set_xlabel(x_col)
	ax.set_ylabel(y_col)
	ax.grid(True, alpha=0.2)
	ax.legend(loc="best", fontsize=7, markerscale=1.8, frameon=True)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def _plot_continuous(df, x_col, y_col, value_col, out_path, alpha=0.6, marker_size=7, cmap="viridis", label=None):
	if value_col not in df.columns:
		return

	data = df[[x_col, y_col, value_col]].copy()
	data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
	data = data.dropna(subset=[x_col, y_col, value_col])
	if len(data) == 0:
		return

	fig, ax = plt.subplots(figsize=(9, 7), dpi=220)
	sc = ax.scatter(
		data[x_col].values,
		data[y_col].values,
		c=data[value_col].values,
		cmap=cmap,
		s=float(marker_size),
		alpha=float(alpha),
		linewidths=0,
	)
	cb = fig.colorbar(sc, ax=ax)
	title_label = value_col if label is None else label
	cb.set_label(title_label)
	ax.set_title(f"MZ embeddings 2D colored by {title_label}")
	ax.set_xlabel(x_col)
	ax.set_ylabel(y_col)
	ax.grid(True, alpha=0.2)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def _age_shaded_colors(age_vals, base_rgb):
	age = pd.to_numeric(pd.Series(age_vals), errors="coerce").astype(float).values
	age_min = np.nanmin(age)
	age_max = np.nanmax(age)
	if not np.isfinite(age_min) or not np.isfinite(age_max) or age_max <= age_min:
		t = np.full_like(age, 0.5, dtype=float)
	else:
		t = (age - age_min) / (age_max - age_min)
	# Blend from clear (near white) to dark (base color) according to age.
	white = np.array([1.0, 1.0, 1.0], dtype=float)
	base = np.array(base_rgb[:3], dtype=float)
	colors = white[None, :] * (1.0 - t[:, None]) + base[None, :] * t[:, None]
	return np.clip(colors, 0.0, 1.0)


def _plot_age_with_category_shading(df, x_col, y_col, age_col, cat_col, out_path, alpha=0.6, marker_size=7, top_categories=8):
	if age_col not in df.columns or cat_col not in df.columns:
		return

	data = df[[x_col, y_col, age_col, cat_col]].copy()
	data[age_col] = pd.to_numeric(data[age_col], errors="coerce")
	data[cat_col] = data[cat_col].astype("string").fillna("<na>").astype(str)
	data = data.dropna(subset=[x_col, y_col, age_col])
	if len(data) == 0:
		return

	data[cat_col] = _prepare_labels(data[cat_col], top_n=top_categories)
	categories = sorted(data[cat_col].unique().tolist())
	cmap = plt.cm.get_cmap("tab10", max(len(categories), 2))

	fig, ax = plt.subplots(figsize=(9, 7), dpi=220)
	for i, cat in enumerate(categories):
		mask = (data[cat_col] == cat).values
		if not np.any(mask):
			continue
		sub = data.loc[mask]
		colors = _age_shaded_colors(sub[age_col].values, cmap(i))
		ax.scatter(
			sub[x_col].values,
			sub[y_col].values,
			c=colors,
			s=float(marker_size),
			alpha=float(alpha),
			label=f"{cat} (n={len(sub)})",
			linewidths=0,
		)

	ax.set_title(f"MZ embeddings 2D by {cat_col} with age shading")
	ax.set_xlabel(x_col)
	ax.set_ylabel(y_col)
	ax.grid(True, alpha=0.2)
	ax.legend(loc="best", fontsize=7, markerscale=1.8, frameon=True)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def _categorical_driver_score(Z, series, min_group_size=30):
	s = series.astype("string").fillna("<na>").astype(str)
	group_counts = s.value_counts(dropna=False)
	valid_groups = set(group_counts[group_counts >= int(min_group_size)].index.tolist())
	mask = s.isin(valid_groups).values
	if np.sum(mask) < 2 or len(valid_groups) < 2:
		return np.nan, int(np.sum(mask)), int(len(valid_groups))

	Zs = Z[mask]
	ss_total = np.sum((Zs - Zs.mean(axis=0, keepdims=True)) ** 2, axis=0)
	den = np.maximum(ss_total, 1e-12)

	between = np.zeros((Zs.shape[1],), dtype=float)
	ss_mean = Zs.mean(axis=0)
	ss = pd.Series(s[mask])
	for g in sorted(valid_groups):
		gm = (ss == g).values
		if not np.any(gm):
			continue
		ng = float(np.sum(gm))
		mu = Zs[gm].mean(axis=0)
		between += ng * ((mu - ss_mean) ** 2)

	eta2 = np.clip(between / den, 0.0, 1.0)
	weights = den / np.maximum(np.sum(den), 1e-12)
	score = float(np.sum(eta2 * weights))
	return score, int(np.sum(mask)), int(len(valid_groups))


def _continuous_driver_score(Z, series):
	x = pd.to_numeric(series, errors="coerce").values.astype(float)
	mask = np.isfinite(x)
	if np.sum(mask) < 3:
		return np.nan, int(np.sum(mask))

	xs = x[mask]
	Zs = Z[mask]
	xs = xs - np.mean(xs)
	std_x = np.std(xs)
	if std_x <= 1e-12:
		return np.nan, int(np.sum(mask))
	xs = xs / std_x

	ss_total = np.sum((Zs - Zs.mean(axis=0, keepdims=True)) ** 2, axis=0)
	den = np.maximum(ss_total, 1e-12)

	r2 = []
	for j in range(Zs.shape[1]):
		z = Zs[:, j]
		z = z - np.mean(z)
		std_z = np.std(z)
		if std_z <= 1e-12:
			r2.append(0.0)
			continue
		z = z / std_z
		r = float(np.mean(xs * z))
		r2.append(np.clip(r * r, 0.0, 1.0))

	r2 = np.array(r2, dtype=float)
	weights = den / np.maximum(np.sum(den), 1e-12)
	score = float(np.sum(r2 * weights))
	return score, int(np.sum(mask))


def _plot_driver_scores(df_scores, out_path, top_n=12):
	if len(df_scores) == 0:
		return

	df = df_scores.sort_values("score", ascending=False).head(int(top_n)).copy()
	df = df.iloc[::-1]

	fig, ax = plt.subplots(figsize=(10, 6), dpi=220)
	colors = ["#1f77b4" if t == "categorical" else "#ff7f0e" for t in df["type"].tolist()]
	ax.barh(df["attribute"].values, df["score"].values, color=colors)
	ax.set_xlabel("driver score")
	ax.set_title("Top embedding drivers (higher = stronger separation)")
	ax.grid(True, axis="x", alpha=0.25)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def _build_person_activity_shares(df_trips):
	if len(df_trips) == 0:
		return pd.DataFrame(columns=["person_id"])

	required = ["person_id", "trip_id", "origin_purpose", "purpose"]
	available = [c for c in required if c in df_trips.columns]
	if len(available) < 3:
		return pd.DataFrame(columns=["person_id"])

	rows = []
	for pid, g in df_trips.groupby("person_id"):
		gs = g.sort_values("trip_id")
		acts = []
		if "origin_purpose" in gs.columns:
			origin_first = gs["origin_purpose"].iloc[0]
			acts.append("unknown" if pd.isna(origin_first) else str(origin_first))
		acts.extend(gs["purpose"].fillna("unknown").astype(str).tolist())
		if len(acts) == 0:
			continue

		count = pd.Series(acts).value_counts(dropna=False)
		total = float(count.sum())
		row = {"person_id": pid}
		for act, c in count.items():
			row[f"act_share__{act}"] = float(c) / max(total, 1.0)
		rows.append(row)

	df = pd.DataFrame(rows)
	if len(df) == 0:
		return pd.DataFrame(columns=["person_id"])

	share_cols = [c for c in df.columns if c.startswith("act_share__")]
	df[share_cols] = df[share_cols].fillna(0.0)
	return df


def _compute_activity_influence_table(df_attr, attr_cols, activity_cols, min_group_size=100):
	if len(activity_cols) == 0:
		return pd.DataFrame(columns=["attribute", "group", "activity", "delta_share", "group_mean_share", "overall_mean_share", "n"])

	global_mean = df_attr[activity_cols].mean(axis=0)
	rows = []
	for attr in attr_cols:
		if attr not in df_attr.columns:
			continue
		s = df_attr[attr].astype("string").fillna("<na>").astype(str)
		group_counts = s.value_counts(dropna=False)
		valid_groups = group_counts[group_counts >= int(min_group_size)].index.tolist()
		for g in valid_groups:
			mask = (s == g).values
			n = int(np.sum(mask))
			if n < int(min_group_size):
				continue
			group_mean = df_attr.loc[mask, activity_cols].mean(axis=0)
			delta = group_mean - global_mean
			for col in activity_cols:
				rows.append({
					"attribute": attr,
					"group": g,
					"activity": col.replace("act_share__", ""),
					"delta_share": float(delta[col]),
					"group_mean_share": float(group_mean[col]),
					"overall_mean_share": float(global_mean[col]),
					"n": n,
				})

	return pd.DataFrame(rows)


def _plot_activity_influence_heatmap(df_inf, attribute, out_path, top_k_activities=10):
	df = df_inf[df_inf["attribute"] == attribute].copy()
	if len(df) == 0:
		return

	activity_strength = (
		df.groupby("activity")["delta_share"]
		.apply(lambda x: float(np.mean(np.abs(x))))
		.sort_values(ascending=False)
	)
	selected_activities = activity_strength.head(int(top_k_activities)).index.tolist()
	df = df[df["activity"].isin(selected_activities)].copy()
	if len(df) == 0:
		return

	pivot = df.pivot_table(index="group", columns="activity", values="delta_share", aggfunc="mean")
	if pivot.shape[0] == 0 or pivot.shape[1] == 0:
		return

	pivot = pivot.fillna(0.0)
	vmax = float(np.nanmax(np.abs(pivot.values)))
	vmax = max(vmax, 1e-6)

	fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 4, 0.45 * pivot.shape[0] + 3), dpi=220)
	im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

	ax.set_xticks(np.arange(pivot.shape[1]))
	ax.set_yticks(np.arange(pivot.shape[0]))
	ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right")
	ax.set_yticklabels(pivot.index.tolist())
	ax.set_title(f"Activity share influence by {attribute} (group mean - overall mean)")

	cb = fig.colorbar(im, ax=ax)
	cb.set_label("delta activity share")

	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def execute(context):
	# Ensure embeddings are generated by matched_v2.
	context.stage("synthesis.population.matching.matched_v2")

	analysis_path = context.config("analysis_path")
	out_dir = os.path.join(analysis_path, "matching_embedding")
	os.makedirs(out_dir, exist_ok=True)

	method = context.config("matching_embedding_plot_method")
	sample_size = int(context.config("matching_embedding_plot_sample_size"))
	alpha = float(context.config("matching_embedding_plot_alpha"))
	marker_size = float(context.config("matching_embedding_plot_marker_size"))
	random_seed = int(context.config("random_seed"))
	top_categories = int(context.config("matching_embedding_plot_top_categories"))
	activity_top_k = int(context.config("matching_embedding_activity_top_k"))
	activity_min_group = int(context.config("matching_embedding_activity_influence_min_group_size"))
	commute_clip_m = float(context.config("matching_embedding_commute_clip_m"))

	emb_stage_path = context.path("synthesis.population.matching.matched_v2")
	emb_file = os.path.join(emb_stage_path, "mz_embeddings", "mz_embeddings.csv")

	if not os.path.exists(emb_file):
		raise RuntimeError(f"Embedding file not found: {emb_file}")

	df_emb = pd.read_csv(emb_file)
	if "person_id" not in df_emb.columns:
		raise RuntimeError("Embedding file must contain 'person_id'.")

	emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
	if len(emb_cols) == 0:
		raise RuntimeError("Embedding file has no embedding columns (expected emb_*).")

	if sample_size > 0 and len(df_emb) > sample_size:
		df_emb = df_emb.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)

	X = df_emb[emb_cols].astype(float).values
	X2 = _reduce_to_2d(X, method=method, random_seed=random_seed)

	df_2d = df_emb[["person_id"]].copy()
	df_2d["x2d"] = X2[:, 0]
	df_2d["y2d"] = X2[:, 1]

	df_mz = context.stage("data.microcensus.persons").copy()
	df_chain = context.stage("data.microcensus.activity_chains").copy()
	if "person_id" in df_chain.columns and "activity_chain" in df_chain.columns:
		df_mz = df_mz.merge(df_chain[["person_id", "activity_chain"]], on="person_id", how="left")
	meta_cols = [
		"person_id", "sex", "canton_id", "municipality_type", "sp_region",
		"employed", "employment_status", "car_availability", "income_class",
		"work_location_type", "marital_status", "number_of_bikes_class", "ovgk", "age", "commute_distance", "activity_chain",
	]
	meta_cols = [c for c in meta_cols if c in df_mz.columns]
	df_plot = df_2d.merge(df_mz[meta_cols], on="person_id", how="left")

	# Robust commute feature for visualization/scoring: clip to 80km then log-scale.
	if "commute_distance" in df_plot.columns:
		commute_raw = pd.to_numeric(df_plot["commute_distance"], errors="coerce")
		commute_clip = np.clip(commute_raw, 0.0, max(commute_clip_m, 1.0))
		df_plot["commute_distance_clip"] = commute_clip
		df_plot["commute_distance_log"] = np.log1p(commute_clip)

	# Add cross-effect labels for richer cluster interpretation.
	cross_definitions = [
		("employed", "car_availability", "employed_x_car_availability"),
		("employment_status", "car_availability", "employment_status_x_car_availability"),
		("employed", "ovgk", "employed_x_ovgk"),
	]
	for col_a, col_b, out_col in cross_definitions:
		cross = _compose_cross_label(df_plot, col_a, col_b)
		if cross is not None:
			df_plot[out_col] = cross

	groupings = [
		"sex",
		"canton_id",
		"municipality_type",
		"sp_region",
		"employed",
		"employment_status",
		"work_location_type",
		"car_availability",
		"income_class",
		"ovgk",
		"employed_x_car_availability",
		"employment_status_x_car_availability",
		"employed_x_ovgk",
	]

	generated = []
	for g in groupings:
		if g not in df_plot.columns:
			continue
		out_path = os.path.join(out_dir, f"mz_embeddings_{g}.png")
		_plot_scatter(
			df_plot,
			"x2d",
			"y2d",
			g,
			out_path,
			alpha=alpha,
			marker_size=marker_size,
			top_categories=top_categories,
		)
		generated.append(out_path)

	age_plot = os.path.join(out_dir, "mz_embeddings_age.png")
	_plot_continuous(
		df_plot,
		"x2d",
		"y2d",
		"age",
		age_plot,
		alpha=alpha,
		marker_size=marker_size,
		label="age",
	)
	if os.path.exists(age_plot):
		generated.append(age_plot)

	commute_plot = os.path.join(out_dir, "mz_embeddings_commute_distance.png")
	_plot_continuous(
		df_plot,
		"x2d",
		"y2d",
		"commute_distance_log",
		commute_plot,
		alpha=alpha,
		marker_size=marker_size,
		cmap="plasma",
		label=f"log1p(min(commute_distance, {int(commute_clip_m)} m))",
	)
	if os.path.exists(commute_plot):
		generated.append(commute_plot)

	age_sex_plot = os.path.join(out_dir, "mz_embeddings_age_x_sex.png")
	_plot_age_with_category_shading(
		df_plot,
		"x2d",
		"y2d",
		"age",
		"sex",
		age_sex_plot,
		alpha=alpha,
		marker_size=marker_size,
	)
	if os.path.exists(age_sex_plot):
		generated.append(age_sex_plot)

	age_emp_plot = os.path.join(out_dir, "mz_embeddings_age_x_employed.png")
	_plot_age_with_category_shading(
		df_plot,
		"x2d",
		"y2d",
		"age",
		"employed",
		age_emp_plot,
		alpha=alpha,
		marker_size=marker_size,
	)
	if os.path.exists(age_emp_plot):
		generated.append(age_emp_plot)

	# Quantify which attributes drive embedding separation.
	min_group_size = int(context.config("matching_embedding_driver_min_group_size"))
	driver_top_n = int(context.config("matching_embedding_driver_top_n"))
	df_driver = df_emb[["person_id"] + emb_cols].merge(df_mz[meta_cols], on="person_id", how="left")
	Z = df_driver[emb_cols].astype(float).values

	driver_rows = []
	categorical_drivers = [
		"sex", "employed", "employment_status", "car_availability", "ovgk",
		"sp_region", "canton_id", "municipality_type", "income_class",
		"work_location_type",
	]
	if "commute_distance" in df_driver.columns:
		commute_raw = pd.to_numeric(df_driver["commute_distance"], errors="coerce")
		commute_clip = np.clip(commute_raw, 0.0, max(commute_clip_m, 1.0))
		df_driver["commute_distance_log"] = np.log1p(commute_clip)

	continuous_drivers = ["age", "commute_distance_log"]

	for col in categorical_drivers:
		if col not in df_driver.columns:
			continue
		score, n_used, n_groups = _categorical_driver_score(Z, df_driver[col], min_group_size=min_group_size)
		if not np.isfinite(score):
			continue
		driver_rows.append({
			"attribute": col,
			"type": "categorical",
			"score": score,
			"n_used": n_used,
			"n_groups": n_groups,
		})

	for col in continuous_drivers:
		if col not in df_driver.columns:
			continue
		score, n_used = _continuous_driver_score(Z, df_driver[col])
		if not np.isfinite(score):
			continue
		driver_rows.append({
			"attribute": col,
			"type": "continuous",
			"score": score,
			"n_used": n_used,
			"n_groups": np.nan,
		})

	df_scores = pd.DataFrame(driver_rows).sort_values("score", ascending=False).reset_index(drop=True)
	driver_csv = os.path.join(out_dir, "embedding_driver_scores.csv")
	df_scores.to_csv(driver_csv, index=False)

	driver_plot = os.path.join(out_dir, "embedding_driver_scores.png")
	_plot_driver_scores(df_scores, driver_plot, top_n=driver_top_n)
	if os.path.exists(driver_plot):
		generated.append(driver_plot)

	# Attribute influence on activity composition from per-person activity shares.
	df_trips = context.stage("data.microcensus.trips")[0].copy()
	df_act = _build_person_activity_shares(df_trips)
	activity_cols = [c for c in df_act.columns if c.startswith("act_share__")]
	activity_influence_csv = None
	if len(activity_cols) > 0:
		df_attr = df_mz[[c for c in meta_cols if c != "activity_chain"]].merge(df_act, on="person_id", how="inner")
		influence_attrs = [
			"employed", "sex", "employment_status", "car_availability", "ovgk",
			"work_location_type", "income_class", "sp_region", "commute_distance",
		]
		df_inf = _compute_activity_influence_table(
			df_attr,
			influence_attrs,
			activity_cols,
			min_group_size=activity_min_group,
		)
		if len(df_inf) > 0:
			activity_influence_csv = os.path.join(out_dir, "activity_share_influence.csv")
			df_inf.to_csv(activity_influence_csv, index=False)
			for attr in influence_attrs:
				heatmap_path = os.path.join(out_dir, f"activity_influence_{attr}.png")
				_plot_activity_influence_heatmap(
					df_inf,
					attr,
					heatmap_path,
					top_k_activities=activity_top_k,
				)
				if os.path.exists(heatmap_path):
					generated.append(heatmap_path)

	csv_path = os.path.join(out_dir, "mz_embeddings_2d.csv")
	df_plot.to_csv(csv_path, index=False)

	logger.info("Matching embedding analysis done. Generated %d plots in: %s", len(generated), out_dir)
	return {
		"path": out_dir,
		"embedding_file": emb_file,
		"plots": generated,
		"table": csv_path,
		"driver_scores": driver_csv,
		"driver_plot": driver_plot,
		"activity_influence_table": activity_influence_csv,
		"n_points": len(df_plot),
		"method": method,
	}
