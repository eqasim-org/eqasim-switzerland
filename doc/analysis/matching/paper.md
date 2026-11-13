# Matching manuscript and Overleaf project

The manuscript is in `analysis/matching/paper/`. It is a generated analysis
artifact with editable LaTeX sections, BibTeX references, figures, aggregate
numerical evidence, a compiled PDF and a self-contained Overleaf ZIP.

- PDF: `analysis/matching/paper/main.pdf`
- Overleaf: `analysis/matching/paper/eqasim_matching_overleaf.zip`
- Main source: `analysis/matching/paper/main.tex`
- Authors, affiliation, contact and title: `metadata.tex`
- Numerical evidence and build checks: `evidence.json`, `build_info.json`

Upload the ZIP to Overleaf and compile `main.tex` using pdfLaTeX and BibTeX.
No Python execution, private data, cluster path or network request is needed
for compilation. The authors, affiliation and contact email are preserved from
the previous project. The current preprint template is retained; the content
is oriented towards Transportation's travel-behaviour, policy and planning scope.

## Current empirical revision

The manuscript uses the completed `repeated_shared_v1_v2_v3` experiment:
37,876 survey respondents, one encoder with a 34,088/3,788 fitting/validation
split, three household donor-exclusion groups, 50 diary draws per respondent,
and ten full configured-population assignments per method. This supersedes the
older paper's 12,000-person, fold-refitted results. The three display names are
Recursive Hot-Deck Matching (RHDM), Behaviour-Supervised Latent Matching (BSLM),
and Order-independent Hot-Deck Matching (OHDM).

The paper presents eqasim's established recursive hot deck as the default-method
reference. This describes the reference approach, not the current private
configuration's `matching_algorithm: v3` selection. The model explanation covers
inputs, encoder/decoder architecture, losses, training and donor sampling rather
than pipeline files, classes or stage-specific exceptions.

The replay is explicitly donor-restricted reconstruction: respondents are kept
out of their own donor group, but most contributed to encoder fitting. The 10%
validation group chooses the checkpoint and is not an untouched final test set.
Separate training-role results are provided. The population comparison is
sensitivity analysis, with the survey as a descriptive reference rather than
known target-population behaviour. Ranges over ten assignments quantify matching
randomness only. No transport simulation or policy effect is newly estimated.

## Local analysis and figures

Saved assignments are joined to the sampled population by person ID. Central
areas are 2 km circles around Zürich HB and Genève-Cornavin, using base-stop
coordinates from the scenario GTFS feed in EPSG:2056. They include 9,643 and
13,899 eligible synthetic residents. Within 4 km of each station, homes are also
grouped into 1 km grid cells. Cells require at least 100 residents and the
relative count comparison requires at least 0.05 RHDM visits per person-day.
All qualifying cells are retained: 44 around Zürich and 37 around Geneva.
These results concern residents' diaries, not destinations or traffic inside
those areas. National reconstructed means are checked against saved report means.

All manuscript graphics use the latest experiment. They include the model
architecture, replay errors, training-role diagnostics, probability calibration,
daily-distance distributions, national/canton visit counts, the fifteen most
frequent survey purpose chains, central-area contrasts and the complete local
cell-difference distributions. A fresh five-panel t-SNE illustration of the
single fitted encoder is included in the appendix as a high-resolution PNG.
No old permutation-importance, raw-input kNN or transported-risk results are
mixed into the revised experiment.

`prepare_latest.py` reads the saved experiment and creates aggregate tables and
figures. It never calls a matching algorithm or fits an encoder. Its optional
model-cache argument loads the existing encoder only for the t-SNE projection.
The evidence records source-file hashes and the actual fitted-model options.
Private person IDs, donor maps and model files are excluded from the ZIP.

```bash
uv run -m analysis.matching.paper.prepare_latest \
  --results /path/to/analysis/matching/repeated_shared_v1_v2_v3 \
  --roster /path/to/sampled-population.p \
  --stops /path/to/gtfs/output/stops.txt \
  --model-cache /path/to/shared-model.p
module load texlive/20220321
uv run -m analysis.matching.paper.build
```

There is no default page cap. The build checks actual page count, unresolved
citations/references, bibliography warnings and overfull boxes. It packages only
current sections, cited figures and current aggregate evidence, preserving older
artifacts outside the archive. The previous Overleaf project is backed up under
`paper/review/manuscript_before_shared_model_revision.zip`.

Before journal submission, authors still need to confirm funding, competing
interests, contributions and any required AI-assistance disclosure under the
journal's current instructions. These declarations are not inferred from code
or funding assumptions. The manuscript itself includes a factual data/code
availability paragraph.
