"""Report the doubled-count experiment without new fits."""
from .ablation_double_counts import render


def configure(context):
    context.stage('analysis.matching.ablation_double_counts')


def execute(context):

    # Render the completed follow-up and saved control scores.

    return render(context.stage('analysis.matching.ablation_double_counts'))
