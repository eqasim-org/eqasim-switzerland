from .assignment import assign, assignment_version, configure_assignment


def configure(context):
    configure_assignment(context, "moving")


def execute(context):

    # Assign the employed residents in this workplace category.

    return assign(context, "moving")


def validate(context):
    return assignment_version()
