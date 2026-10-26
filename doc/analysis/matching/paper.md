# Matching manuscript and Overleaf project

The research manuscript lives in `analysis/matching/paper/` as requested. It is
an analysis artifact, separate from repository guides. It contains editable
LaTeX sections, a BibTeX bibliography, publication-sized PDF figures, aggregate
numerical evidence, a compiled PDF and a self-contained Overleaf ZIP.

- PDF: `analysis/matching/paper/main.pdf`
- Overleaf: `analysis/matching/paper/eqasim_matching_overleaf.zip`
- Main document: `analysis/matching/paper/main.tex`
- Author information/title: `analysis/matching/paper/metadata.tex`
- Provenance and checks: `evidence.json`, `build_info.json`

Upload the ZIP as a new Overleaf project, select `main.tex` as the main document,
and use **pdfLaTeX**. Standard BibTeX/natbib support is sufficient. The bundle
contains its figures and tables; no cluster path, raw survey, Python execution,
model fitting or network data fetch is needed to compile it. Authors are
anonymous until the agreed names/affiliations are entered in `metadata.tex`.

There is no default page cap, following the revised manuscript requirements.
The build verifies the actual page count, rejects unresolved references/citations
and overfull boxes, and packages the sources only after successful checks. An
optional `--max-pages N` enforces a future submission limit including references.

```bash
module load texlive/20220321
.venv/bin/python analysis/matching/paper/build.py
```

## Method terminology

The abstract defines **Recursive Hot-Deck Matching (RHDM)** for the existing
recursive matcher and **Behaviour-Supervised Latent Matching (BSLM)** for the
learned matcher. The manuscript text, tables and figure labels subsequently use
these abbreviations. Archived data and replay identifiers retain `V1`/`V2` for
compatibility; `evidence.json` records their display-name mapping.

The employment/student hard mask is explained in the methodology as a downstream
eqasim compatibility safeguard. It is not presented as a headline contribution
in the abstract, introduction or conclusion. Matching status does not require a
work or education trip on every observed day.

## Figures and empirical scope

Figure 1 is a detailed, editable TikZ network architecture: the numeric/missing
inputs and four categorical projections, 48-value concatenation, 64-unit encoder,
16-value latent representation, training noise, recurrent prediction heads,
trip-count/reconstruction branches and separate real-diary retrieval path.

Figure 4 uses the original 1,500 held-out respondents and original t-SNE
coordinates in five panels: age, income, density, recorded sex and work-location
type. The added attributes are joined one-to-one from the survey cache after
its prepared feature fingerprint is verified against the original experiment.
The categorical panels have discrete legends; no projection is refitted.

Figure 6 compares observed survey, RHDM-matched and BSLM-matched **purpose-chain
frequencies in the held-out survey experiment**. Its ten categories are fixed
by the survey-weighted observed ranking. Denominators include every recipient;
the remaining chains appear in a separate panel. Recipient survey weights are
used for all three distributions, and 500 household-bootstrap intervals average
model/draw repeats per recipient first. Full-distribution total variation is
reported descriptively without claiming a significant difference between methods.

The donor draws were not originally exported, but can be replayed using saved
fold IDs, normalized embeddings, weights, settings and seeds. BSLM requires neither
model loading nor training; RHDM uses its original unchanged implementation. Input
feature/trip fingerprints, donor/recipient household separation and **both**
per-person energy and ordered purpose/mode Brier scores are checked against the
archive before accepting chain shares. This revision verified agreement to
floating-point precision across all six fold/repeat combinations.

Only aggregate frequencies, intervals and replay audit information enter the
paper bundle. Resumable donor-index arrays remain in a private temporary
replay directory and are not included in the ZIP. The replay is not a new
full-population application and does not supply BSLM population matches.

Figure 7 retains an explicitly labelled weighting-sensitivity illustration:
the same observed survey days under survey weights and estimated target weights.
Important balance checks fail for that reweighting. It is presented as sensitivity,
not as measured target behaviour or a corrected population distribution.

The paper defines raw-input nearest neighbours and status-stratified random
matching before using their short labels. It also introduces network layers,
regression/classification losses, CRPS/Brier scores and representation/shift
diagnostics in words before interpreting the figures.

## Scientific qualifications

The manuscript reports improvement over RHDM and also the simple controls that
equal or beat BSLM on several outcomes. Upstream survey `work_location_type` partly
uses the observed trip diary. Holding households out during training does not
remove that input dependence, so these are conditional prepared-covariate
results. An outcome-independent feature ablation is needed for fully blind
raw-input validation. Target transport also fails important balance checks.

The references distinguish the California paper (San Francisco/San Diego) from
Los Angeles support documented by the California implementation. The published
Paris paper's DOI/article number is **10.1016/j.trc.2021.103291**. The Switzerland
baseline source is a 2022 technical report. Sources were checked against
publisher, author/institutional or official framework records.

## Reproducing the replay and manuscript evidence

Run the donor-only replay with explicitly selected trusted local Synpp artifacts:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m analysis.matching.paper.replay_matching \
    --validation /path/to/analysis.matching.validation__HASH.cache \
    --persons /path/to/data.microcensus.persons__HASH.p \
    --trips /path/to/data.microcensus.trips__HASH.p \
    --chains /path/to/data.microcensus.activity_chains__HASH.p \
    --pipeline /path/to/pipeline.json
```

The selected person cache must reproduce the archived feature fingerprint.
Similar-looking caches from different imputation seeds can have different
fingerprints; mismatches are rejected. This tool does not run validation,
retrain models, repeat feature permutations or modify upstream caches. It does
repeat the inexpensive matching selections if verified resume files are absent.

Then prepare publication figures and numerical tables:

```bash
.venv/bin/python -m analysis.matching.paper.prepare_artifacts \
    --reports /path/to/matching/rebuilt \
    --chains /path/to/data.microcensus.activity_chains__HASH.p \
    --persons /path/to/data.microcensus.persons__HASH.p
```

Rebuild the PDF afterwards. The scripts export aggregate evidence and file
fingerprints without respondent IDs. Pickle inputs must be trusted local stage
artifacts. Interpretational text and manually described source/runtime settings
must be reviewed if a new experiment replaces the archived one.
