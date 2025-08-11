import subprocess as sp
import shutil, os
import warnings

def configure(context):
    context.config("osmosis_binary", "osmosis")

    context.config("java_binary", "java")
    context.config("java_memory", "50G")

def run(context, arguments = [], cwd = None):
    """
        This function calls osmosis.
    """
    # Make sure there is a dependency
    if context.stage("matsim.runtime.osmosis"):
        if cwd is None:
            cwd = context.path()

        # Prepare command line
        command_line = [
            shutil.which(context.config("osmosis_binary"))
        ] + arguments

        # Prepare environment
        environment = os.environ.copy()
        environment["JAVACMD"] = shutil.which(context.config("java_binary"))
        environment["JAVACMD_OPTIONS"] = "-Xmx%s" % context.config("java_memory")

        # Run Osmosis
        return_code = sp.check_call(command_line, cwd = cwd, env = environment)

        if not return_code == 0:
            raise RuntimeError("Osmosis return code: %d" % return_code)
    else:
        raise RuntimeError(f"Cannot find Osmosis binary at: {context.config('osmosis_binary')}")


def is_osmosis_installed(context):
    """
    Returns True if the Osmosis binary is found in PATH, otherwise False.
    """
    binary = context.config("osmosis_binary")
    if not binary:  # None or empty string
        return False
    return shutil.which(binary) is not None


def validate(context):
    if not is_osmosis_installed(context):
        warnings.warn(f"Cannot find Osmosis binary at: {context.config('osmosis_binary')}")
        return False
    else:
        osmosis_path = shutil.which(context.config("osmosis_binary"))
        try:
            output = sp.check_output([osmosis_path, "-v"], stderr=sp.STDOUT)
            if b"0.48." not in output:
                warnings.warn("Osmosis of at least version 0.48.x is recommended!")            
        except Exception as e:
            warnings.warn(f"Failed to check Osmosis version: {e}")
        return True
    

def execute(context):
    return True if is_osmosis_installed(context) else False