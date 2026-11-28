from .assignment import assign, assignment_version, configure_assignment


def configure(context):
    configure_assignment(context, "fixed")


def execute(context):

    # Assign the employed residents in this workplace category.

    return assign(context, "fixed")


def validate(context):
    return assignment_version()
