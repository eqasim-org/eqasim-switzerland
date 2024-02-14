import numpy as np
import pandas as pd


def configure(context):
    context.stage("synthesis.population.activities")


def execute(context):
    df_activities = context.stage("synthesis.population.activities")
    activity_lists = ','.join(df_activities["purpose"].unique())
    return activity_lists
