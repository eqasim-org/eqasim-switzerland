"""
Routing car passenger is the same as routing cars, so will just return cars travel times
"""


def configure(context):
    context.stage("mode_choice.travel_times.car")

def execute(context):    
    return context.stage("mode_choice.travel_times.car")
    
    
    

    