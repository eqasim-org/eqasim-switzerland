**Statistical matching**

Milos had noticed that the synthetic population contained agents under 18 years of age but with driving license = True. The reason was that statistical matching uses age_class as a matching variable, but not as a "priority" one. For the household-based matching, it was 3rd on the list and even 4th for the individual-based matching. As a result, ~9% of the agents (in the second matching) were matched with MZ respondents potentially not in their age group.
Two improvements were proposed:
- In synthesis.population.enriched, a few lines were added to ensure that driving license status and age match, so that this issue doesn't happen again.
- To reduce the number of affected agents, a new version of the statistical matching algorithm was developed. It is a recursive version that:
    - considers given variables (argument: mandatory_columns) as extremely important for the matching.
    - iteratively reduces the minimum number of observations to ensure that as many agents as possible are matched using at least the provided mandatory columns.
Minor update: the "household_size_class" variable has 5 possible values, which creates a large number of population categories to explore. We decided to reduce this number - once again, to ensure that as many agents as possible are matched on the important variables - to 2 by replacing household_size_class by a binary variable assessing the presence of children (under 12 years old) in the population. This is why changes in data.statpop.statpop and data.microcensus were necessary.
Ultimate goal (which will get important when we will add the possibility to choose which census to use): make this class as flexible as possible, defining columns and processes to use in a separate class.

The arguments of parallel_statistical_matching are now:
- context -- as before
- df_source, source_identifier, source_weight -- as before
- df_target, target_identifier -- as before
- columns -- as before
- mandatory_columns -- those are the columns most important for the matching algorithm. Default value: None. Choose the default value to use the non-recursive statistical matching. Must be a "left-slice" of columns.
- minimum_observations -- as before.

At each recursion step, the minimum number of observations decreases according to the formula given in the "decrease_minimum_observation" function. Please make sure that it is a strictly decreasing function that reaches 1 at some point, otherwise nothing guarantees that the recursion will ever stop.


**Including GTFS in the simulation**

We can now choose the public transport schedule we want to use in the simulation.
- A parameter named "pt_schedule" was added in the config. It is read in the matsim.scenario.network.convert_pt_schedule stage and accepts two values: "hafas" and "gtfs".
- If "hafas" is chosen, the usual stage matsim.scenario.network.convert_hafas will be executed. The date given in the config parameter "hafas_date" will be used. It should have the format "MM.DD.YYYY" - or "DD.MM.YYYY" (?)
- If "gtfs" is chosen (and this is the default value), instead of  matsim.scenario.network.convert_hafas, matsim.scenario.network.convert_gtfs will be executed. 
- pt2matsim is unable to directly process GTFS zipped files. A new stage was consequently added in the "data" folder to read the GTFS and convert it to a format that pt2matsim can use. This stage is data.gtfs.cleaned.
- Similar to when we are using HAFAS, we can specify a particular day of interest in the config with the "gtfs_date" parameter. The date should be given in the format "yyyymmdd". If no date is specified, the default value "dayWithMostTrips" is used. A final option is to give the value "dayWithMostServices". Please refer to the documentation of pt2matsim to learn about those options.
- The stage data.gtfs.cleaned starts by exploring the contents of a gtfs folder located in the data repository (config.data_path/gtfs). If no GTFS zipped file is within this folder, it will return an error. If multiple such files are found, by default, all zipped GTFS will be merged into a single schedule. It is possible to choose one specific GTFS by providing its name in the config parameter "gtfs_name". In that case, only the file config.data_path/gtfs/gtfs_name.zip will be converted.
- The most recent HAFAS and GTFS data can be found here: https://opentransportdata.swiss/de/.
- Our focus year for the CMDP project will be 2024. I am currently using a GTFS from December 2023, when the PT schedule was released: https://data.opentransportdata.swiss/de/dataset/timetable-2024-gtfs2020/resource/184fea8b-6f93-460e-bdb1-9fad5f9b82aa.

**Selecting the pt2matsim version**
- We can now select the version of pt2matsim we want to use in the simulation. Two parameters were included in the config to do this: pt2matsim_version and pt2matsim_branch. 
- Currently, we are still using the 22.3 version (pt2matsim_version: "22.3" and pt2matsim_branch: v22.3 in the config). The goal is to switch to the most recent release (25.2) but this still yields errors in the simulation (something about finding the closest links? to be investigated) so we cannot change the default version at the moment.