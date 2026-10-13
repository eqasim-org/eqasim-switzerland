import subprocess as sp
import os, shutil
import logging
import re
logger = logging.getLogger("synpp")


def configure(context):
    context.config("java_binary", "java")
    context.config("java_memory", "50G", volatile=True)


def run(context, entry_point, arguments = [], class_path = None, vm_arguments = [], cwd = None, memory = None, mode = "raise"):
    """
        This function calls java code. There are three modes:
        - return_code: Returns the return code of the Java call
        - output: Returns the output of the Java call
        - raise (default): Raises an exception if the return code is not zero
    """
    # Make sure there is a dependency
    settings = context.stage("matsim.runtime.java")

    # Prepare temp folder
    temp_path = "%s/__java_temp" % context.path()
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    # Prepare arguments
    xmx, xms = get_memories(context, settings, memory)
    vm_arguments = [
        "-Xmx" + xmx,
        "-Xms" + xms,
        "-XX:+UseParallelGC",
        "-Djava.io.tmpdir=%s" % temp_path,
        "-Dmatsim.useLocalDtds=true",
        "-Djava.awt.headless=true"
    ] + vm_arguments

    # Prepare classpath
    if type(class_path) == list or type(class_path) == tuple:
        class_path = ":".join(class_path)

    # Preapre CWD
    if cwd is None:
        cwd = context.path()

    # Prepare command line
    command_line = [
        shutil.which(settings["binary"]),
        "-cp", class_path
    ] + vm_arguments + [
        entry_point
    ] + arguments

    command_line = list(map(str, command_line))

    logger.info("Executing java: %s", " ".join(command_line))

    if mode == "raise" or mode == "return_code":
        return_code = sp.check_call(command_line, cwd = cwd)

        if not return_code == 0:
            raise RuntimeError("Java return code: %d" % return_code)

        return return_code
    elif mode == "output":
        return sp.check_output(command_line, cwd = cwd)
    else:
        raise RuntimeError("Mode is expected to be one of 'raise', 'return_code' or 'output'")


def validate(context):
    if shutil.which(context.config("java_binary")) in ["", None]:
        raise RuntimeError("Cannot find Java binary at: %s" % context.config("java_binary"))

    version_output = sp.check_output([
            shutil.which(context.config("java_binary")),
            "-version"
        ], stderr = sp.STDOUT)
    version_match = re.search(rb'version\s+"(\d+)(?:\.(\d+))?', version_output)
    if version_match is None:
        raise RuntimeError("Cannot determine Java version from: %r" % version_output)

    version = int(version_match.group(1))
    if version == 1 and version_match.group(2) is not None:
        version = int(version_match.group(2))

    if version < 21:
        raise RuntimeError("Java version 21 or later is required; found version %d." % version)
    if version < 25:
        logger.warning("A Java JDK of at least version 25 is recommended.")


def get_memories(context, settings, memory):
    if memory is None:
        try:
            memory = context.config("java_memory")
        except:
            memory = settings["memory"]

    # getting the memory as integer
    if isinstance(memory, str):
        if memory.lower().endswith("g"):
            memory = memory[:-1]
    
    memory_value = int(memory)
    xmx = str(memory_value)+'G'
    xms = str(int(0.7 * memory_value))+'G'
    return xmx, xms


def execute(context):
    return {
        "binary": context.config("java_binary"),
        "memory": context.config("java_memory")
    }
