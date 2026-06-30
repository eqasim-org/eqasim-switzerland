import os


def configure(context):
    context.config("data_path")
    context.config("counts_path", default=os.path.join(context.config("data_path"),"traffic_counts"))

def execute(context):
    year = 2025
    # Define paths
    data_path = context.config("counts_path")
    count_stations_file = os.path.join(data_path,"Switzerland","liste_des_postesdecomptage.xlsx")
    counts_data_file = os.path.join(data_path,"Switzerland",f"Bulletin_{year}_fr.xlsx")
    ### these files are processed in the counts class, during matching
    ### the reason why is ... historical
    ### I started with this, so all the code as based on these data
    #TODO: refactor later (process these files here)
    
    return (count_stations_file, counts_data_file, year)