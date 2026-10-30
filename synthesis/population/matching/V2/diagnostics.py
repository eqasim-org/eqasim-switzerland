"""V2 stage exports for model evidence, matching support and donor reuse."""

import json
from pathlib import Path

import pandas as pd

from .features import feature_inventory


class MatchingDiagnostics:
    """Write auditable copies without changing predictions or upstream inputs."""

    @staticmethod
    def write(output, matcher, population, target, prediction):
        output = Path(output)
        embedding_dir = output / "mz_embeddings"
        embedding_dir.mkdir(parents=True, exist_ok=True)
        embeddings = matcher.donor_embeddings_
        exported = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
        exported.insert(0, "person_id", matcher.donors_["mz_id"].to_numpy())
        exported.to_csv(embedding_dir / "mz_embeddings.csv", index=False)
        assigned = prediction.donor_ids
        matched = assigned >= 0
        details = prediction.diagnostics.copy()
        details["mz_person_id"] = assigned
        for column in ("age", "income_class", "employed", "is_student", "sex",
                       "employment_status", "car_availability", "sp_region"):
            details[column] = target[column].to_numpy()
        if "household_counts_reconstructed" in target:
            details["household_counts_reconstructed"] = target["household_counts_reconstructed"].to_numpy()
        details.to_csv(output / "matching_diagnostics.csv", index=False)
        reuse = pd.Series(assigned[matched]).value_counts().rename_axis("mz_person_id").rename("recipient_count")
        reuse.to_csv(output / "donor_reuse.csv")
        feature_inventory(matcher.donors_, population).to_csv(output / "feature_inventory.csv", index=False)
        report = {
            "eligible_persons": len(target), "matched_persons": int(matched.sum()),
            "removed_persons": len(prediction.unmatched_person_ids), "used_donors": len(reuse),
            "maximum_donor_reuse": int(reuse.max()) if len(reuse) else 0,
            "household_counts_reconstructed_fraction": float(population.get(
                "household_counts_reconstructed", pd.Series(False, index=population.index)).mean()),
            "pools": prediction.pools, "training": matcher.training_report_,
        }
        (output / "matching_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        return report
