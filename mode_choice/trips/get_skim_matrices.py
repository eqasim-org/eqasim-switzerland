import pandas as pd

def configure(context):
    context.config("generate_skim_matrices", default = False)
    if context.config("generate_skim_matrices"):
        context.stage("calibration.pt_pricing.process_results_uspat")
    else:
        context.config("skim_matrices_path")


def execute(context):
    generate_matrices = context.config("generate_skim_matrices")
    if generate_matrices:
        matrices = context.stage("calibration.pt_pricing.process_results_uspat")
    else:
        path = context.config("skim_matrices_path")
        matrices = pd.read_csv(path)

    print(matrices.head())
    return matrices

