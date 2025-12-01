def configure(context):
    context.config("use_skim_matrices")
    if context.config("use_skim_matrices"):
        context.stage("mode_choice.variables.pt_skim_matrices")
    else:
        context.stage("mode_choice.variables.pt_java")

def execute(context):
    use_skim_matrices = context.config("use_skim_matrices")
    if use_skim_matrices:
        pt = context.stage("mode_choice.variables.pt_skim_matrices")
    else:
        pt = context.stage("mode_choice.variables.pt_java")

    return pt
