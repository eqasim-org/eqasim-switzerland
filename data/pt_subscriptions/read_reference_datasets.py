def configure(context):
    context.stage("data.pt_subscriptions.npvm")
    context.stage("data.pt_subscriptions.sbb")

def execute(context):
    npvm = context.stage("data.pt_subscriptions.npvm")
    sbb  = context.stage("data.pt_subscriptions.sbb")

    print(npvm.head())
    print(sbb.head())

    return {"npvm": npvm, "sbb": sbb}