# GA ownership by postcode and Junior travelcards

To enable both changes, add this setting under the existing `config` block. Its default is `false`.

```yaml
    use_sbb_ga_rank_and_junior_split: true
```

Run `synthesis.population.models.subscriptions` to inspect the assigned products or `matsim.simulation.prepare` to use them in a prepared scenario.

## Swiss residents aged 16 and over

The model ranks home postcodes by their 2015 SBB GA ownership percentage. Higher rates get higher ranks. Ties receive their average rank. Flagged and missing rates are excluded. The weighted MZ subscription model learns the association between rank and individual ownership alongside its existing predictors. A common shift in the predicted GA log odds keeps the country's expected number of adult GA holders equal to the original model. The realised count can vary with the person-level draw.

For residents without a usable postcode rank, the model uses the original probabilities. For all other residents, the relative probabilities of the four alternatives to GA are preserved. The stage log reports postcode coverage and the expected adult GA count before and after the adjustment.

## Children aged 6–15

Children keep the existing draw for the pooled GA/Junior subscription. Within that group the model samples GA only, Junior only, or both. It estimates those proportions nationally from MZ 2015 weekday respondents aged 6–15, using the person weight `WP` and valid yes/no responses for both products (`f41610a` and `f41610f`). This split uses its own seeded draw.

Use the boolean person attributes `has_ga` and `has_junior` for ownership analysis. A child can have both. Routing and pricing continue to read `pt_subscription`, where either product gives code 1 (written as `subscriptions="GA"` in the MATSim person attributes). Children under six and residents held out of modelling with home-only plans have both flags set to false. Imported agents without an ownership estimate have neither flag in the XML.

## Inputs

The enabled option reads these existing files relative to `data_path`.

- `pt_reference_data/generalabo-halbtax-mit-bevolkerungsdaten.csv` supplies the 2015 SBB postcode rates and suppression flags.
- `microcensus/haushalte.csv` supplies the respondent's home postcode (`W_PLZ`) by household number (`HHNR`).
- `microcensus/zielpersonen.csv` supplies weekday child ownership and survey weights.
- `spatial/postal_codes/AMTOVZ_ZIP.shp` and its companion files locate modelled home coordinates in postcode polygons.
