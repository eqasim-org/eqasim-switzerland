def configure(context):
    context.config("work_location_algorithm", default = "synthesis.population.spatial.primary.work.work_locations")
    
    if context.config("work_location_algorithm") == "synthesis.population.spatial.primary.work.work_locations":
        context.stage("synthesis.population.spatial.primary.work.work_locations", alias="work_locations")
    else:
        context.stage("synthesis.population.spatial.primary.education.locations", alias="work_locations")

    context.stage("synthesis.population.spatial.primary.education.locations", alias="education_locations")

def execute(context):
    df_work = context.stage("work_locations")
    df_education = context.stage("education_locations")

    return df_work, df_education
