import subprocess as sp

class JavaRunner:
    def __init__(self, binary, memory):
        self.memory = memory
        self.binary = binary

    def __call__(self, classpath, entry_point, arguments, vm_arguments = None, cwd = None, memory = None, output = False):
        memory = self.memory if memory is None else memory

        if vm_arguments is None:
            vm_arguments = []

        vm_arguments = ["-Xmx" + self.memory] + vm_arguments

        if type(classpath) == list or type(classpath) == tuple:
            classpath = ":".join(classpath)

        command_line = [self.binary, "-cp", classpath] + vm_arguments + [entry_point] + arguments

        print("Executing Java:")
        print("  " + " ".join(command_line))

        if output:
            return sp.check_output(command_line, cwd = cwd)
        else:
            return sp.check_call(command_line, cwd = cwd)

def configure(context, require):
    require.config("java_memory", "10G")
    require.config("java_binary", "java")

    # Not ideal, because we assume that "java" is the right binary.
    # This should better go into a "validate" step between configure and
    # execute ... TODO
    assert("1.8" in sp.check_output(["java", "-version"], stderr = sp.STDOUT).decode("utf-8"))

def execute(context):
    return JavaRunner(context.config["java_binary"], context.config["java_memory"])
