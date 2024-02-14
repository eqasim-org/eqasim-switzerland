def configure(context):
    context.stage("synthesis.population.spatial.primary.work.locations")
    context.stage("synthesis.population.spatial.primary.education.locations")

    if context.config("run_snn"):
        context.config("run_snn")
        context.config("snn_heuristic")
        context.stage("synthesis.population.spatial.primary.work_from_home.locations")


def execute(context):
    df_work = context.stage("synthesis.population.spatial.primary.work.locations")
    df_education = context.stage("synthesis.population.spatial.primary.education.locations")

    if context.config("run_snn"):
        if context.config("snn_heuristic") != 0:
            df_wfh = context.stage("synthesis.population.spatial.primary.work_from_home.locations")
            return df_work, df_education, df_wfh

    return df_work, df_education
