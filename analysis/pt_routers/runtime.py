"""Build the comparison's component-capable Java runtime without invalidating scenario caches."""
from matsim.runtime import eqasim
from analysis.pt_routers.context import ExecuteView


def configure(context):
    context.stage('matsim.runtime.git')
    context.stage('matsim.runtime.java')
    context.stage('matsim.runtime.maven')
    context.config('eqasim_version', eqasim.DEFAULT_EQASIM_VERSION)
    context.config('eqasim_branch', eqasim.DEFAULT_EQASIM_BRANCH)
    context.config('eqasim_repository', 'https://github.com/eqasim-org/eqasim-java.git')
    context.config('pt_router_analysis_eqasim_commit', eqasim.DEFAULT_EQASIM_COMMIT)
    context.config('pt_router_analysis_eqasim_path', '')


def execute(context):

    # Isolate this build from the source SUMO configuration's older Java commit.

    view = ExecuteView(context, {}, {'eqasim_commit': context.config('pt_router_analysis_eqasim_commit'),
                                    'eqasim_path': context.config('pt_router_analysis_eqasim_path')})
    return eqasim.execute(view)


def validate(context):
    path = context.config('pt_router_analysis_eqasim_path')
    if not path:
        return True
    import os
    return os.stat(path).st_mtime_ns
