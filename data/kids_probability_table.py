import pandas as pd

def configure(context):
    context.config("kids_proba_table_data_path")
    
def execute(context):
    path = context.config("kids_proba_table_data_path")
    proba = pd.read_csv(path)
    
    proba         = proba[["Ages", "CH_home", "CH_educ", "CH_else"]]
    proba.columns =      ["Age", "Home", "Education", "Non-education"]
    
    return proba