def assign_zones_to_gtfs(gtfs):

    zone_to_places = {
        "1": ["Sion, Centre Funéraire", "Sion, Carrefour de Platta",
              "Sion, Vieux-Moulin", "Sion, Brasserie", "Sion, Pont-sur-la-Sionne",
              "Sion, Nord", "Sion, Capucins", "Sion, école d'infirmières",
              "Sion, Mont d'Orge", "Pont-de-la-Morge, centre", "Sion, Garenne",
              "Sion, Tennis Iles", "Sion, Les Iles Est", "Sion, Auto-Pôle",
              "Sion, Voirie", "Salins, Turin", "Sion, Les Fournaises",
              "Sion, Maragnénaz", "Bramois, la Crettaz", "Bramois, Est",
              "Bramois, Cassières", "Sion, SUVA", "Sion, Hôpital de Sion"],
        "2": ["Sion, Molignon", "Sion, Batassé", "Uvrier, centre commercial",
              "Uvrier, Le Pont", "St-Léonard, Le Lac", "St-Léonard", "Uvrier, sud",
              "Châteauneuf (Conthey), Pinède", "Châteauneuf-Conthey",
              "Sion, Camping des Iles", "Aproz, village", "Salins, La Courtaz",
              "Salins, Arvillard", "Salind, Pravidondaz", "Pravidondaz (Salins), école",
              "Salins, Misériez", "Salins, village"],
        "3": ["Mayens-de-l'Ours, télécabine", "Les Agettes, bif. M.-de-l'Ours",
              "Les Mayens-de-Sion, Ouest", "Les Agettes, Crételannaz",
              "Les Agettes, d'en haut", "Les Agettes, village", "Les Agettes, Stade US",
              "Les Agettes,s Crettaz-à-l'Oeil", "Les Agettes, contour du Pêchot", 
              "Les Agettes, les Crêtes", "Les Agettes, Crettaz-à-l'Oeil"]
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

    sion = gtfs[gtfs["zones"].notna()]
    
    sion["tarif network"] = "SionPT"
    sion["local network"] = "Sion"
    sion["zones"] = (
        sion["zones"]
        .fillna("")  # Handle NaNs
        .apply(lambda z: [int(x) for x in z.split("/") if x.strip().isdigit()] if z else [])
    )

    sion = sion[["stop_id", "stop_name", "geometry", "tarif network", "local network", "zones"]]

    return sion