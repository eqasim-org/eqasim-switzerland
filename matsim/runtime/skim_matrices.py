import os
import matsim.runtime.java as java


def configure(context):
    # Ensure MATSim has run and SBB extensions are built
    context.stage("matsim.simulation.run")
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")


def execute(context):
    # Access working directory from previous stages
    working_directory = context.stage("matsim.simulation.run")
    eqasim = context.stage("matsim.runtime.eqasim")
    sim = os.path.join(working_directory, "simulation_output")

    # Input tessellation grid
    zones = "/cluster/work/ivt_vpl/anding/Secondary_Locations/preprocessed/SkimMatrixGrid.geojson"
    zones_id = "GRID_ID"

    # MATSim outputs - use facilities instead of network for sampling points
    facilities = os.path.join(sim, "output_facilities.xml.gz")
    network = os.path.join(sim, "output_network.xml.gz")
    schedule = os.path.join(sim, "output_transitSchedule.xml.gz")
    events = os.path.join(sim, "output_events.xml.gz")  # Can be empty string if using freespeed

    # Output directory
    out_dir = os.path.join(sim, "skim_matrices")
    os.makedirs(out_dir, exist_ok=True)

    # Parameters
    points_per_zone = 4
    threads = 8
    
    # Time periods for car (can be multiple separated by semicolon)
    times_car = "06:00:00"  # Morning peak

    # Time periods for PT (start;end for each)
    times_pt = "06:00:00;08:00:00"  # 06:00-08:00
    
    # Which modes to calculate
    modes = "car,pt"  # comma-separated

    print("Calculating skim matrices using SBB MATSim extensions...")
    print(f"  Zones: {zones}")
    print(f"  Points per zone: {points_per_zone}")
    print(f"  Output: {out_dir}")

    
    # CalculateSkimMatrices.main(String[] args) expects:
    # args[0]: zonesShapeFilename
    # args[1]: zonesIdAttributeName
    # args[2]: facilitiesFilename
    # args[3]: networkFilename
    # args[4]: transitScheduleFilename
    # args[5]: eventsFilename (can be empty for freespeed)
    # args[6]: outputDirectory
    # args[7]: numberOfPointsPerZone
    # args[8]: numberOfThreads
    # args[9]: timesCar (semicolon-separated, e.g., "06:00:00;08:00:00;17:00:00")
    # args[10]: timesPt (semicolon-separated, e.g., "06:00:00;08:00:00")
    # args[11]: modes (comma-separated, e.g., "car,pt")
    
    eqasim.run(context, "ch.sbb.matsim.analysis.skims.CalculateSkimMatrices", [
        zones,
        zones_id,
        facilities,
        network,
        schedule,
        events,
        out_dir,
        str(points_per_zone),
        str(threads),
        times_car,
        times_pt,
        modes
    ])

    print("Skim matrices saved to:", out_dir)

    return out_dir
