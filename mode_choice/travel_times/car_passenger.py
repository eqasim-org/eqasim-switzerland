import pandas as pd
import numpy as np
import os

"""
Routing car passenger is the same as routing cars, so will just return cars travel times
"""
def config(context):
    context.stage("mode_choice.travel_times.car")


def execute(context):    
    return context.stage("mode_choice.travel_times.car")
    
    
    

    