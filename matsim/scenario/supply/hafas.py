import os.path
import calendar
from datetime import datetime

from matsim.runtime import pt2matsim


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    context.stage("utils.java")
    context.config("data_path")
    context.config("hafas_date")

    context.config("weekend_scenario", False)
    context.config("specific_weekend_scenario", "all")  # options are "all", "saturday", "sunday"
    context.config("specific_day_scenario", "avgworkday")  # options can be any of the days of the week or "avgworkday"
    context.config("scaling_year")



def execute(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.pt2matsim")
    java = context.stage("utils.java")

    #check that hafas_date corresponds to scenario
    # hafas date format is DD.MM.YYYY - see pg 56 of
    # https://transportdatamanagement.ch/content/uploads/2020/04/HRDF.5.20.39-Guidelines-e.pdf
    pt_schedule_date = context.config("hafas_date")
    pt_schedule_object = datetime.strptime(pt_schedule_date, '%d.%m.%Y').date()
    hafas_day = pt_schedule_object.weekday()  # 0-6 from monday to Sunday
    hafas_day_str = calendar.day_name[hafas_day]
    specified_weekend_scenario = context.config("specific_weekend_scenario")
    specific_day_scenario = context.config("specific_day_scenario")

    if (context.config("weekend_scenario") & (hafas_day < 5)) | (
            (specified_weekend_scenario != "all") & (hafas_day_str.lower() != specified_weekend_scenario.lower())) | (
            specific_day_scenario != "avgworkday") & (hafas_day_str.lower() != specific_day_scenario.lower()):
        raise Exception(f"Wrong Hafas date ({hafas_day_str}) for the specified scenario")

    #check if hafas date based on scaling year
    scaling_year = context.config("scaling_year")
    hafas_year = pt_schedule_object.year
    current_year = datetime.today().date().year

    #gtfs files for sbb schedule starts from 2016
    #https://opentransportdata.swiss/en/group/timetables-gtfs

    #if (scaling_year >= 2016 & scaling_year <= current_year) & (hafas_year != scaling_year):
    #    raise Exception(f"Scaling year {scaling_year} does not match hafas year {hafas_year}")

    #check if hafas database contains hafas year. If not download 
    #if scaling is higher than current year then hafas_year = current_year
    #Note that the hafas year have to be pre-downloaded. Now we are using the hafas 5.2 format


    # Create MATSim schedule
    
    hafas_path = "%s/hafas/" % context.config("data_path")
    hafas_path += str(hafas_year)

    pt2matsim.run(context, "org.matsim.pt2matsim.run.Hafas2TransitSchedule", [
        hafas_path, "EPSG:2056",
        "%s/transit_schedule.xml.gz" % context.path(),
        "%s/transit_vehicles.xml.gz" % context.path(),
        context.config("hafas_date")
    ])

    assert (os.path.exists("%s/transit_schedule.xml.gz" % context.path()))
    assert (os.path.exists("%s/transit_vehicles.xml.gz" % context.path()))

    # return {
    #     "schedule": "%s/transit_schedule.xml.gz" % context.cache_path,
    #     "vehicles": "%s/transit_vehicles.xml.gz" % context.cache_path
    # }

    return dict(
        schedule_path="transit_schedule.xml.gz",
        vehicles_path="transit_vehicles.xml.gz"
    )
