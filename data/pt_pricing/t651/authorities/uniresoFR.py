def assign_zones_to_gtfs(gtfs):

    zone_to_places = {
        "250": ["Versonnex", "Divonne-les-B", "Gex", "Grilly",
                "Ségny", "Cessy", "Maconnex", "Ornex", "Ferney",
                "Prévessin", "Crozet", "Chevry", "Saint-Genis",
                "Sergy", "Thoiry"],
        "240": ["Challex", "Pougny-Chancy", "Pougny, gare (F)"],
        "400": ["Bellegarde (Valserine), gare", "Bellegarde-sur-Valserine"],
        "300": ["Thonon-les-Bains, Gare", "Evian-les-Bains, gare SNCF"],
        "200": ["Veigy", "Veigy-Foncenex"],
        "210": ["Machilly", "Saint-Cergues (F)", "St-Cergues (F)", "Saint-Cergues", 
                "Cranves-Sales", "Ville-la-Grand", "Annemasse", "Gaillard", "Ambilly"],
        "380": ["Groisy-Thorens-la-Caille", "Pringy (Haute-Savoie)",
                "Annecy, Pringy Gare", "Pringy-Gare-Arret-Sibra",
                "Annecy, Gare routière"],
        "230": ["Saint-Julien, douane", "Saint-Julien, Hutins", "Saint-Julien-en-Genevois",
                "Saint-Julien-en-G", "Saint-Julien, Lathoy-Hameau", "Archamps",
                "Collonges", "Collonges-sous-Salève", "Beaumont le Châble", "Neydens",
                "Saint-Julien, Cervonnex", "Saint-Julien, Casino", "Saint-Julien, Rue des Muguets",
                "Viry", "Chênex", "Valleiry", "Vulbens"]
    }

    places_to_zones = {}

    for key, value in zone_to_places.items():
        value_clean = list(set(value))
        for place_name in value_clean:
            if not place_name in places_to_zones:
                places_to_zones[place_name] = key
            else:
                places_to_zones[place_name] += ("/" + key) 

    gtfs = gtfs[["stop_id", "stop_name", "geometry"]]

    gtfs["zones"] = None

    for place_name, zones in places_to_zones.items():
        mask = (gtfs["stop_name"].str.startswith(place_name + ",")) | (gtfs["stop_name"].str.startswith(place_name + " ")) | (gtfs["stop_name"] == place_name)
        if not mask.any():
            continue
        gtfs.loc[mask, "zones"] =  zones

    mask = (gtfs["stop_name"]=="Annecy")
    gtfs.loc[mask, "zones"]  = "380"

    unireso = gtfs[gtfs["zones"].notna()]
    
    unireso["tarif network"] = "Unireso"
    unireso["local network"] = "Genève"
    unireso["zones"] = (
        unireso["zones"]
        .fillna("")  # Handle NaNs
        .apply(lambda z: [int(x) for x in z.split("/") if x.strip().isdigit()] if z else [])
    )

    uniresoFR = unireso[["stop_id", "stop_name", "geometry", "tarif network", "local network", "zones"]]

    return uniresoFR