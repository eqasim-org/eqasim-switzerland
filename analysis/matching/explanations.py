"""Plain-language reading guides shared by the HTML reports."""

FIGURES = {
    "paired_predictive_skill": (
        "Does V2 predict hidden survey days better?",
        "For each hidden respondent, we scored the set of possible donor days against their actual day. "
        "Joint behaviour combines trip counts, time, distance, purposes and modes. Dependence checks relationships "
        "between day summaries. Ordered diary checks the probability of the complete activity/mode sequence. "
        "We compare V2 with V1, matching directly on input characteristics (raw kNN), and random matching within "
        "employment/student groups. Results use survey weights.",
        "Each point is the average V2 error minus the reference error, expressed as a percentage of the reference "
        "error. Left of zero favours V2; right favours the reference. Bars are 95% intervals from resampling whole "
        "households 500 times by default. Model repeats are averaged per person first. The actual bootstrap count "
        "is recorded in report_settings.json for rebuilt reports. An interval crossing zero leaves the direction uncertain."),
    "behaviour_error_distributions": (
        "How many people receive useful distance, time and chain predictions?",
        "We sort respondents from small to large prediction error and add their survey weights. Distance and time "
        "use CRPS: a score rewarding possible days close to the observed value while penalizing an unnecessarily "
        "spread-out prediction. Chain Brier error checks probabilities assigned to complete ordered activity/mode chains.",
        "At an error on the horizontal axis, the curve gives the share of respondents with that error or less. "
        "A curve further left or higher is better. These are distributions of per-person prediction errors, "
        "not distributions of population travel. Only the horizontal view is cropped at the pooled 98th percentile; "
        "all people remain in the calculation. The same people appear in both curves."),
    "chain_skill_by_day_type": (
        "Which kinds of days are matched better?",
        "We group hidden respondents using their real observed day: stayed home; attended education (possibly also "
        "worked); worked without education; or made another kind of journey. We then score the full ordered purpose "
        "and mode chain, rather than only checking which group their day belonged to. This does not force predicted "
        "days to have the same purpose as the observed day.",
        "Lower chain probability error is better. The coloured points compare V1 and V2 for the same respondents; "
        "bars are 95% household-bootstrap intervals. Separate algorithm bars are descriptive: use the paired "
        "differences figure for evidence about the V1/V2 difference. Groups are defined only for this report and "
        "are never matching masks. Counts are provided in the table."),
    "mean_prediction_calibration": (
        "Are expected daily distances, times and trip counts realistic?",
        "For each respondent, we average their sampled donor days to obtain an expected daily value. For each "
        "algorithm separately, we sort these expected values into up to ten equally sized groups, then compare "
        "survey-weighted predicted and observed means within each group.",
        "Points near the dotted diagonal mean that the predicted average agrees with the actual average. Above "
        "the line means underprediction; below means overprediction. These curves assess average calibration. "
        "They do not show the full distribution of sampled days, and matching a group average alone does not "
        "establish accurate individual predictions. The two algorithms can place people in different bins."),
    "predictive_interval_coverage": (
        "Do the possible donor days capture real day-to-day variation?",
        "For each person and model fit, the saved donor ensemble supplied central 50%, 80% and 95% prediction "
        "intervals. We count how often the real day fell inside, average repeated fits per person, and then apply "
        "survey weights. This reuses the saved interval indicators; it does not reconstruct missing donor draws.",
        "Near the diagonal, the claimed range contains the advertised fraction of actual days. Below it suggests "
        "ranges are too narrow; above can indicate broad ranges or discrete outcomes such as trip counts. "
        "Coverage must be read with interval width (in the table): broad ranges can cover many days without being "
        "useful. Finite donor ensembles and integer outcomes limit exact agreement with the diagonal."),
    "cohort_predictive_skill": (
        "Does improvement reach different ages, incomes and household settings?",
        "We calculate V2 minus V1 joint behaviour error for each respondent, then average within reporting "
        "groups using survey weights. Age and residential density use up to five equally populated survey groups; "
        "income retains its original classes and children are counts, with three or more combined for readability. "
        "Density is the upstream count of residents within 2.5 km. Groups with fewer than 30 respondents are omitted.",
        "Points left of zero favour V2. Bars are 95% paired household-bootstrap intervals. "
        "These ranges summarize reporting groups only: exact age and child counts remain model inputs, and "
        "neither age, income nor density becomes a hard matching rule. Many groups are exploratory, so isolated "
        "intervals should not be treated as a set of independent significance tests."),
    "event_reliability": (
        "Can we trust predicted activity and mode probabilities?",
        "For each person, the fraction of sampled donor days containing an event is its predicted probability. "
        "We average model repeats, group probabilities into five fixed bands, then measure the actual event "
        "frequency using survey weights. Car means being the driver; PT means public transport.",
        "A point near the diagonal indicates agreement between predicted probability and observed frequency. "
        "Vertical bars are 95% household intervals for the observed frequency. Reliability and prediction error "
        "answer different questions; a constant average probability can be calibrated yet uninformative."),
    "rank_calibration": (
        "Are real days systematically outside the predicted range?",
        "For every saved person/model prediction, we locate the real value among the sampled donor values, "
        "using the saved counts below and equal to the observation. We randomize ties reproducibly and draw "
        "a survey-weighted rank histogram. Small ranks place the actual value near the low end of the prediction.",
        "A roughly flat histogram is consistent with calibrated predictive distributions. A U shape suggests "
        "too little spread; an uneven left/right pattern suggests bias. Ties, finite ensembles and repeated "
        "model measurements affect this descriptive picture. It is not a formal significance test."),
    "continuous_age_skill": (
        "How does the V1/V2 difference vary with age?",
        "For each respondent, we subtract V1 joint behaviour error from V2 error. At each age on the curve, "
        "we average these paired differences with survey weights and a smooth age window (Gaussian bandwidth "
        "four years), allowing nearby ages to contribute gradually.",
        "Below zero favours V2. This is a descriptive smoothed trend without confidence bands. It does not "
        "create age bins in matching, and patterns near rare ages need particular care."),
    "heldout_representation": (
        "Who is near whom in the learned representation?",
        "Each dot is one respondent from the first held-out fold and first model seed, sampled reproducibly. "
        "The same dots appear in each panel, coloured by age, original income class or log residential density. "
        "t-SNE bends the 16-dimensional representation into two dimensions to show local neighbourhoods; PCA "
        "is a linear view when that option is selected. Neither view changes matching.",
        "Nearby dots may share model inputs or learned behaviour. Look for gradual colour changes and "
        "consistent local groups. Apparent islands are not proof of distinct behavioural clusters; island size, "
        "large gaps and axis directions have no direct behavioural units. Trustworthiness in the table checks "
        "how well plotted neighbours reflect original-space neighbours."),
    "embedding_behaviour": (
        "Do nearby respondents actually have similar observed behaviour?",
        "These are the same held-out dots and coordinates as the input-colour figure, now coloured by observed "
        "trip count, log daily distance, work participation and education participation. These observed outcomes "
        "were hidden during model training and are added only to interpret this figure.",
        "Smooth patches of colour suggest that the representation organizes these aspects of behaviour. "
        "Mixed colours are expected because similar people can have different days. This picture is exploratory; "
        "held-out prediction scores and original-space neighbourhood measurements provide the quantitative test."),
    "projection_sensitivity": (
        "Does the apparent grouping depend on the plotting choice?",
        "We plot the exact same people using PCA and t-SNE with two neighbourhood settings (perplexities, "
        "normally 15 and 50), and the report's selected main view. Colour is exact age in every panel. "
        "All projections use a fixed random seed. Small samples use reduced perplexities.",
        "Look for local patterns that persist across views. t-SNE focuses on local neighbours, so distances "
        "between separate islands and their relative sizes can change across settings. This sensitivity "
        "check varies plotting settings only; model-seed stability is measured separately in the table."),
    "neighbour_behaviour": (
        "Are learned neighbours more useful than neighbours in the raw inputs?",
        "For each hidden respondent, we find up to 5 or 20 eligible training neighbours in learned or raw-input "
        "space. We compare each neighbour's observed numeric day summaries with the hidden respondent's day "
        "using Euclidean distance after training-fold scaling. Employment/student eligibility is kept fixed.",
        "Lower average behaviour distance is better. Lines compare spaces within the same outer fold and "
        "model repeat. These are descriptive neighbourhood checks, not the V1 matching algorithm and not "
        "confidence intervals. They complement the main comparison of complete predictive donor ensembles."),
    "predictive_feature_contributions": (
        "Which inputs help predict travel behaviour?",
        "We reuse a test already done during validation: scramble one input group across held-out people "
        "within the same employment/student status, keep related household counts together, re-encode those "
        "people, and match again using the fitted model. We compare the new joint behaviour error with its "
        "unscrambled baseline. Model and shuffle repeats are averaged per person before household resampling.",
        "Each point is the average increase in error caused by scrambling that group. Further right means "
        "the model relied more on its predictive information; bars are 95% household-bootstrap intervals. "
        "Near zero means little detectable contribution in this test; negative values mean shuffling helped. "
        "Correlated inputs can substitute for each other, and unrealistic shuffled combinations can inflate "
        "importance. These are predictive associations, not causal effects. No shuffling or matching is rerun "
        "when rebuilding the report. Work context means commute distance, work-location type and job position. "
        "Mobility resources means car access, household cars and driving licence. Personal context means sex, "
        "marital status and Swiss nationality. Residential context means canton, region, municipality type and "
        "home public-transport quality. Household composition means household size, adult counts and child "
        "counts by age threshold. Density and border proximity are separate inputs. Employment and student "
        "status stay fixed in this test because they are hard eligibility rules."),
}
