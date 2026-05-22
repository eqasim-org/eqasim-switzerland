import shutil
import pandas as pd
import logging

logger = logging.getLogger("synpp")
def configure(context):
    context.config("data_path")
    pt_pricing_option = context.config("generate_pt_pricing_inputs", default=False)

    if pt_pricing_option:
        context.stage("data.pt_pricing.t603.create_sbb_distances")
        context.stage("data.pt_pricing.t651.process_all_authorities")
    
def execute(context):
    data_path         = context.config("data_path")
    pt_pricing_option = context.config("generate_pt_pricing_inputs")

    if not pt_pricing_option:
        sbb_input_file     = f"{data_path}/pt_pricing/output/SBB_all_distances.csv"
        zones_input_file   = f"{data_path}/pt_pricing/output/gtfs_zones.csv"
        pricing_input_file = f"{data_path}/pt_pricing/output/pricingDescription.xml"

        shutil.copy(sbb_input_file, context.path())
        shutil.copy(zones_input_file, context.path())
        shutil.copy(pricing_input_file, context.path())

        logger.info("Files copied")

    else:
        sbb_distances = context.stage("data.pt_pricing.t603.create_sbb_distances")
        authorities   = context.stage("data.pt_pricing.t651.process_all_authorities")

        # TODO create the stages that create the pricing description xml file
        pricing_input_file = f"{data_path}/pt_pricing/output/pricingDescription.xml"
        shutil.copy(pricing_input_file, context.path())

        sbb_distances.to_csv(f"{context.path()}/SBB_all_distances.csv", index=False)
        authorities.to_csv(f"{context.path()}/gtfs_zones.csv", index=False)

    sbb_path     = f"{context.path()}/SBB_all_distances.csv"
    zones_path   = f"{context.path()}/gtfs_zones.csv"
    pricing_path = f"{context.path()}/pricingDescription.xml"

    return sbb_path, zones_path, pricing_path

