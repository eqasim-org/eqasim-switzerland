
def configure(context):
    context.config("travel_times_from", default="tomtom")
    assert context.config("travel_times_from") in ["google", "tomtom"], \
        "travel_times_from must be either 'google' or 'tomtom'"
    
    if context.config("travel_times_from") == "google":
        context.stage("analysis.travel_times.APIs.travel_times_google", alias="travel_times")
    else:
        context.stage("analysis.travel_times.APIs.travel_times_tomtom", alias="travel_times")

def execute(context):
    df = context.stage("travel_times")
    return df
    