import subprocess as sp
import os, os.path, shutil
import xml.etree.ElementTree

import matsim.runtime.git as git
import matsim.runtime.java as java
import matsim.runtime.maven as maven
import matsim.runtime.eqasim as eqasim


def configure(context):
    context.stage("matsim.runtime.git")
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.maven")
    #context.stage("SNN.SNN_run_zurich_astra_config")
    context.stage("matsim.runtime.eqasim")
    context.config("output_path")
    context.stage("matsim.scenario.population")
    context.stage("synthesis.population.SNN_population")
    context.stage("matsim.simulation.run")


def install_ebike_city_eqasim(context):
    eqasim_astra_ebikecity_path = "https://github.com/matsim-eth/e-bike-city.git"

    # Normal case: we clone eqasim
    if eqasim_astra_ebikecity_path != "":
        # Clone repository and checkout version
        git.run(context, [
            "clone", eqasim_astra_ebikecity_path
        ])


def create_config_without_bike_avail_modification(context):

    path = context.path() + "/e-bike-city/src/main/java/ebikecity/project/config/"

    with open(path + "AstraConfigurator.java", "r+") as file1:
        with open(path + "AstraConfiguratorSNN.java", "w") as file2:

            old = file1.readlines()
            begining = old[:94]
            next = old[106:]

            for l in begining:
                l = l.replace("AstraConfigurator", "AstraConfiguratorSNN")
                file2.write(l)

            for m in next:
                m = m.replace("AstraConfigurator", "AstraConfiguratorSNN")
                file2.write(m)

    path = context.path() + "/e-bike-city/src/main/java/ebikecity/utils/impute_headway/"

    with open(path + "RunImputeHeadway.java", "r+") as file1:
        with open(path + "RunImputeHeadwaySNN.java", "w") as file2:

            old = file1.readlines()

            for l in old:
                l = l.replace("1.0 * 3600", "0.5 * 3600")
                l = l.replace("RunImputeHeadway", "RunImputeHeadwaySNN")
                file2.write(l)

    path = context.path() + "/e-bike-city/"
    with open("/nas/asallard/Switzerland/ch-zh-synpop_baseline/SNN/ASTRA_config_pom.xml") as file1:
        with open(path + "pom.xml", "w") as file2:

            old = file1.readlines()

            for l in old:
                file2.write(l)


    path = context.path() + "/e-bike-city/src/main/java/ebikecity/project/mode_choice/AstraModeParameters.java"

    with open(path, "r+") as file1:
        lines = file1.readlines()

    with open(path, "w") as file2:

        for line in lines:
            line = line.replace("parameters.car.alpha_u = -0.8;", "parameters.car.alpha_u = 0.2235;") 
            file2.write(line)

    maven.run(context, ["-Pstandalone", "--also-make", "package"], cwd = "%s/e-bike-city" % context.path())


def run_headway_imputer(context):

    jar_path = "%s/e-bike-city/target/ebike.city-0.0.1-SNAPSHOT.jar" % (
        context.path()
    )

    command = "ebikecity/utils/impute_headway/RunImputeHeadwaySNN"

    arguments = ["--config-path",  context.config("output_path") + "/Zurich5kmconfig.xml", 
                 "--output-path",  context.config("output_path") + "/Zurich5kmpopulation_Headways.xml.gz",
                 "--threads", str(24),
                 ]

    java.run(context, command, arguments, jar_path)


def run_city_attributes(context):

    jar_path = "%s/e-bike-city/target/ebike.city-0.0.1-SNAPSHOT.jar" % (
        context.path()
    )

    command = "ebikecity/utils/cityattribute/RunImputeCityAttribute"

    arguments = ["--input-path",   context.config("output_path") + "/Zurich5kmpopulation_Headways.xml.gz", # input population
                 "--output-path",  context.config("output_path") + "/Zurich5kmpopulation_CityAttribute.xml.gz",
                 "--extent-path", "/nas/asallard/Switzerland/SNN_shapefile_zurich/zurich_5km.shp",
                 ]

    java.run(context, command, arguments, jar_path)



def run_astra_config(context):

    zurich_config_path = context.config("output_path") + "/Zurich5kmconfig.xml"
    plans_path         = context.config("output_path") + "/Zurich5kmpopulation_CityAttribute.xml.gz"

    jar_path = "%s/e-bike-city/target/ebike.city-0.0.1-SNAPSHOT.jar" % (
        context.path()
    )

    command = "ebikecity/project/mode_choice/RunBaselineSimulation"

    arguments = [
        "--config-path", zurich_config_path,
        "--config:controler.lastIteration", str(60),
        "--config:controler.writeEventsInterval", str(10),
        "--config:controler.writePlansInterval", str(10),
        "--config:plans.inputPlansFile", plans_path,
    ]

    java.run(context,  command, arguments, jar_path)


def execute(context):
    install_ebike_city_eqasim(context)

    # 1.1 Modify the parameter in RunImputeHeadway and remove bike availability modifications in Astra Configurator
    create_config_without_bike_avail_modification(context)

    # 2.1 Impute headways
    run_headway_imputer(context)

    # 2.2 Impute city attributes
    run_city_attributes(context)

    # 3. run
    run_astra_config(context)



    
