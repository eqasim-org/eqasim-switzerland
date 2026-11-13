"""Private routing config, with optional reuse of an existing PT scenario."""
from pathlib import Path
from matsim.config import tools


def configure(context):
    context.config('pt_microcensus_source_config', '')
    if not context.config('pt_microcensus_source_config'):
        context.stage('matsim.config.pt')


def execute(context):

    # Resolve inputs relative to their source config; cached scenario files stay read-only.

    source = context.config('pt_microcensus_source_config') or context.stage('matsim.config.pt')
    target = Path(context.path()) / 'routing_config.xml'
    tools.copy_config(source, target, input_paths={
        'network': ['inputNetworkFile'], 'transit': ['transitScheduleFile', 'vehiclesFile'],
        'vehicles': ['vehiclesFile'], 'facilities': ['inputFacilitiesFile'],
        'eqasim': ['costParametersPath', 'modeParametersPath'],
        'sbbPt': ['sbbDataFile'], 'ptZones': ['ptZonesFilePath', 'sbbDistancesPath', 'pricingDescriptionPath']})
    tools.disable_population_inputs(target)
    tree = tools.read_config(target)
    network = tree.find("module[@name='network']/param[@name='inputNetworkFile']").get('value')
    return {'config': str(target.resolve()), 'network': network}


def validate(context):
    source = context.config('pt_microcensus_source_config')
    if source:
        path = Path(source)
        return [str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size]
