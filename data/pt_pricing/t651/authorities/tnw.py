def assign_zones_to_gtfs(gtfs):

    zone_to_places = {
        "10": ["Basel", "Riehen", "Riehen Niederholz", "Bettingen",
               "Birsfelden", "Münchenstein", "Muttenz", "Binningen,",
               "Bottmingen", "Allschwil", "Schönenbuch"],
        "11": ["Oberwil BL", "Biel-Benken BL", "Therwil", "Ettingen",
               "Reinach BL", "Arlesheim", "Dornach,", "Aesch BL", "Pfeffingen"],
        "12": ["Leymen", "Flüh", "Bättwil", "Witterswil", "Hofstetten SO", 
               "Rodersdorf"],
        "66": ["Mariastein", "Metzerlen", "Biederthal", "Burg im Leimental",
               "Challhöchi"],
        "74": ["Kleinlützel"],
        "75": ["Roggenburg", "Kiffis, Les Forges (F)", "Ederswiler, Jurastrasse"],
        "73": ["Challhöchi", "Röschenz", "Bärschwil", "Liesberg", "Grindel"],
        "70": ["Wahlen b. Laufen", "Laufen", "Zwingen", "Brislach", "Breitenbach,",
               "Büsserach"],
        "65": ["Dittingen", "Blauen", "Nenzlingen"],
        "60": ["Duggingen", "Grellingen"],
        "61": ["Himmelried", "Seewen SO"],
        "62": ["Hochwald", "Gempen"],
        "78": ["Fehren", "Meltingen", "Zullwil"],
        "76": ["Erschwil", "Beinwil SO", "Passwang"],
        "23": ["Nunningen", "Bretzwil", "Lauwil", "Reigoldswil"],
        "22": ["Ziefen", "Arboldswil", "Titterten"],
        "21": ["Büren SO", "St. Pantaleon", "Nuglar", "Seltisberg",
               "Lupsingen", "Bubendorf", "Ramlinsburg", "Lampenberg-Ramlinsburg"],
        "20": ["Liestal", "Frenkendorf", "Füllinsdorf", "Frenkendorf-Füllinsdorf"],
        "15": ["Pratteln", "Augst"],
        "27": ["Kaiseraugst", "Giebenach", "Arisdorf", "Hersberg"],
        "28": ["Lausen", "Itingen"],
        "24": ["Lampenberg", "Niederdorf", "Hölstein"],
        "25": ["Bennwil", "Oberdorf BL", "Liedertswil", "Waldenburg"],
        "26": ["Langenbruck"],
        "40": ["Rheinfelden,", "Rheinfelden Augarten"],
        "45": ["Olsberg", "Magden"],
        "41": ["Möhlin"],
        "42": ["Wallbach,", "Zeiningen", "Mumpf"],
        "43": ["Zuzgen", "Stein AG", "Stein-Säckingen", "Münchwilen AG", "Eiken",
               "Sisseln"],
        "44": ["Obermumpf", "Schupfart", "Hellikon", "Wegenstetten"],
        "30": ["Sissach", "Böckten", "Zunzgen", "Thürnen"],
        "32": ["Diepflingen", "Gelterkinden", "Sommerau", "Rümlingen",
               "Tenniken", "Diegten"],
        "33": ["Wintersingen", "Nusshof"],
        "34": ["Rickenbach BL,", "Buus", "Hemmiken", "Rothenfluh", "Tecknau",
               "Wenslingen", "Kilchberg BL", "Zeglingen", "Ormalingen", "Rünenberg"],
        "35": ["Wittinsburg", "Känerkinden", "Eptingen", "Läufelfingen",
               "Buckten", "Häfelfingen"],
        "36": ["Anwil", "Oltingen", "Kienberg"],
        "37": ["Maisprach"],
        "50": ["Frick", "Gipf-Oberfrick", "Oeschgen"],
        "51": ["Densbüren", "Oberhof,"],
        "52": ["Kaisten", "Laufenburg,"],
        "53": ["Ittenthal", "Hornussen"],
        "54": ["Wittnau", "Ueken", "Herznach", "Wölflinswil"],
        "56": ["Rheinsulz", "Sulz", "Obersulz"],
        "57": ["Elfingen", "Bözen", "Effingen", "Ziehen"],
        "58": ["Schwaderloch", "Etzgen", "Mettau", "Oberhofen AG", "Wil AG"],
        "59": ["Gansingen", "Hottwil"],
        "14": ["Basel EuroAirport"]
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

    tnw = gtfs[gtfs["zones"].notna()]
    
    tnw["tarif network"] = "TNW"
    tnw["local network"] = "Basel"
    tnw["zones"] = (
        tnw["zones"]
        .fillna("")  # Handle NaNs
        .apply(lambda z: [int(x) for x in z.split("/") if x.strip().isdigit()] if z else [])
    )

    tnw = tnw[["stop_id", "stop_name", "geometry", "tarif network", "local network", "zones"]]

    return tnw