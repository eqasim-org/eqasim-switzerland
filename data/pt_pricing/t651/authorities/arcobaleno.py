def assign_zones_to_gtfs(gtfs):

    zone_to_places = {
        "40": ["Chiasso", "Pedrinate", "Seseglio",
               "Balerna", "Novazzano", "Genestrerio", "Ligornetto",
               "Brusata (Novazzano)", "Pizzamiglio", "Vacallo",
               "Morbio Inferiore", "Morbio Superiore", "Sagno",
               "Castel S.Pietro", "Mendrisio", "Rancate", "Salorino",
               "Somazzo", "Caneggio", "Campora", "Monte,"],
        "41": ["Monte,", "Campora", "Caneggio", "Bruzella", "Cabbio", "Muggio",
               "Scudellate", "Roncapiano", "Casima", "S.Pietro", "Stabio",
               "Gaggiolo", "Cantello Gaggiolo", "Besazio", "Arzo", "Tremona",
               "Meride", "Riva S. Vitale", "Brusino Arsizio", "Porto Ceresio",
               "Capolago", "Capolago-Riva S. Vitale", "Melano", "Maroggia",
                "Maroggia-Melano", "Rovio", "Arogno", "S. Pietro di Stabio"],
        "11": ["Maroggia","Maroggia-Melano", "Bissone", "Campione", "Melide", "Vico Morcote",
               "Olivella", "Morcote", "Figino", "Casoro", "Barbengo", "Cernesio",
               "Garaverio", "Carona", "Carabbia", "Grancia", "Carabietta", "Scairolo",
               "Agra", "Montagnola", "Piodella", "Cappella-Agnuzzo", "Bioggio Molinazzo",
               "Agno", "Neggio", "Magliaso", "Pura", "Caslano", "Ponte Tresa", "Vernate",
               "Cimo", "Gaggio", "Bosco Luganese", "Manno", "Savosa Liceo", "Gravesano",
               "Bedano", "Torricella", "Taverne", "Lamone", "Cadempino", "Marnigo",
               "Vezia", "Porza", "Cureglia", "Origlio", "Ponte Capriasca", "Tesserete",
               "Lopagno", "Cagiallo", "Villa Luganese", "Sonvico", "Sureggio", "Canobbio",
               "Comano", "Cadro", "Davesco", "Soragno", "Gandria", "Aldesago", "Brè",
               "Cureggia", "Ligaino", "Soragno", "Piano Stampa", "Trevano"],
        "10": ["Lugano", "Paradiso", "Pazzallo", "Gentilino", "Pambio", "Noranco",
               "Pambio-Noranco", "Certenago", "Sorengo", "Cappella-Agnuzzo", 
               "Breganzona", "Muzzano", "Savosa Liceo", "Savosa", "Vezia", "Molino Nuovo",
               "Cassarate", "Castagnola", "Albonago", "Pregassona", "Resega", "Viganello"],
        "13": ["Luino, Stazione FS (I)", "Fornasette", "Molinazzo",
               "Sessa", "Termine", "Monteggio", "Astano", "Ponte Tresa", "Croglio",
               "Beride", "Bedigliora", "Curio", "Novaggio", "Bonbinasco", "Miglieglia",
               "Breno", "Aranno", "Iseo", "Cademario", "Fescoggia", "Vezio", "Mugena",
               "Arosio", "Purasca", "Ponte Cremenaga", "Bombinasco"],
        "12": ["Lopagno", "Roveredo (TI)", "Bidogno", "Corticiasca", "Maglio", "Insone",
               "Colla", "Bogno", "Certara", "Piandera", "Cimadera", "Curtina", "Oggio",
               "Pezzolo", "Lelgio", "Sigirino", "Mezzovico", "Camignolo",
               "Medeglia", "Isone", "Bironico", "Rivera", "Passo del Ceneri", "Signôra",
               "Cozzo, Paese", "Rivera, Passo del Ceneri"],
        "30": ["Ascona", "Locarno", "Losone", "Solduno", "Orselina", "Muralto",
               "Minusio", "Brione s. Minusio", "Tenero"],
        "31": ["Brissago", "Porto Ronco", "Ascona, Moscia", "Arcegno",
               "Ronco sopra Ascona", "Zandone", "Golino", "Intragna", "Corcapolo",
               "Cavigliano", "Verscio", "Tegna", "Cresmino", "Loco", "Berzona",
               "Ponte Brolla", "Avegno", "Gordevio", "Aurigeno", "Moghegno",
               "Maggia", "Contra", "Mergoscia", "Tenero", "Gordola", "Riazzino",
               "Agarone", "Medoscio", "Gordemo", "Berzona (Verzasca)",
               "Vogorno", "Corippo", "Gerra P.", "Cugnasco", "Cadenazzo",
               "Contone", "Quartino", "Madagino", "Orgnana", "Vira (Gambarogno)", 
               "S. Nazzaro", "Gerra (Gambarogno)", "Piazzogna", "Fosano", 
               "Alpe di Neggia", "Ronco (Gambarogno)", "Casenzano", "Vairano", "Alabardia",
               "Mondacce", "Auressio", "Aurigeno-Moghegno", "Ronchini", "Porto Ronco", 
               "Gruppaldo"],
        "36": ["Ranzo-S. Abbondio", "Ranzo", "S. Abbondio", "Caviano", "Dirinella", "Pino-Tronzano",
               "Pino, sotto Stazione","Scaiano", "Indemini"],
        "20": ["Bellinzona", "Castione", "Arbedo", "Gorduno", "Carasso", "Sementina",
               "Artore", "Ravecchia", "Lôro", "Camorino", "Vigana", "Paiardi",
               "S. Antonino", "Giubiasco", "Castione-Arbedo"],
        "21": ["Carena", "S. Antonio", "Vellano", "Paudo", "Pianezzo", 
               "Paiardi", "S. Antonino", "Gudo", "Cugnasco", "Gnosca", "Claro",
               "Preonzo", "Prosito", "Cresciano", "Osogna", "Lodrino", "Castione",
               "Lumino", "S. Vittore", "Roveredo GR", "Grono", "Leggia", "Cama",
               "Verdabbio", "Buseno", "Cadenazzo", "Robasacco",
               "Passo del Ceneri", "Rivera, Passo del Ceneri", "Castione-Arbedo",
               "Melera", "Moleno", "Sta. Maria in Calanca"],
        "28": ["Arvigo", "Selma", "Cauco", "Sta. Domenica", "Augio", "Rossa", "Sorte",
               "Lostallo", "Cabbiolo"],
        "25": ["Soazza", "Mesocco", "Pian S. Giacomo", "S. Bernardino, Villaggio"],
        "22": ["Osogna", "Lodrino", "Iragna", "Biasca", "Loderio", "Pollegio",
               "Bodio", "Personico"],
        "23": ["Bodio", "Personico", "Semione", "Malvaglia", "Motto (Blenio)",
               "Dongio", "Acquarossa", "Ludiano", "Corzoneso", "Leontica", "Prugiasco",
               "Lavorgo", "Calonico", "Giornico", "Sobrio", "Nivo", "Chironico",
               "Motto-Ludiano", "Marogno", "Cumiasca", "Acquarossa-Comprovasco",
                "Comprovasco", "Cavagnago", "Anzonico"],
        "26": ["Acquarossa", "Prugiasco", "Castro", "Ponto Valentino", "Lottigna",
               "Torre", "Dangio", "Aquila", "Olivone", "Campo (Blenio)", "Ghirone",
               "Acquarossa-Comprovasco", "Comprovasco", "Marolta", "Ponte Semina"],
        "24": ["Lavorgo", "Chiggiogna", "Faido", "Rossura", "Tengia",  "Camperio",
               "Campra", "Piansecco", "Calpiogna", "Campello", "Carì", "Predelp",
               "Tortengo", "Mairengo", "Osco", "Polmengo", "Rodi", "Prato (Leventina)",
               "Dalpe", "Fiesso", "Ambrì", "Piotta", "Ambrì-Piotta", "Varenzo",
               "Quinto", "Altanca", "Lurengo", "Airolo", "Nante", "Fontana, Paese",
               "Bedretto", "All'Acqua", "Cioss Prato", "Ronco (Bedretto)", 
               "Villa (Bedretto)", "Molare", "Ronco (Quinto)", "Deggio", 
               "Ossasco (Bedretto)", "Larescia", "S. Martino (Quinto)", 
               "Motto Bartola", "Fontana, Gerora", "Fontana, Cioss di dentro"],
        "27": ["Piansecco", "Pian Segno", "Acquacalda", "Lukmanier Passhöhe",
               "Alpe Casaccia"],
        "35": ["Lavertezzo", "Brione (Verzasca)", "Gerra (Verzasca)", "Frasco",
               "Gerra(Verzasca)", "Sonogno"],
        "34": ["Camedo", "Palagnedra", "Verdasio", "Berzona", "Mosogno",
               "Russo", "Crana", "Gresso", "Vergeletto", "Spruga",
               "Comologno", "Vocaglia", "Corbella"],
        "32": ["Maggia", "Lodano", "Coglio", "Giumaglio", "Someo",
               "Riveo", "Cevio", "Linescio", "Bignasco", "Cavergno", "Brontallo"],
        "33": ["Fusio", "Mogno", "Peccia", "Piano di Peccia", "Sornico", "Prato, Ponte",
               "Broglio", "Menzonio", "Brontallo", "Mondada", "Foroglio",
               "Roseto", "S. Carlo (Bavona)", "Linescio", "Collinasca", "Cerentino",
               "Bosco/Gurin", "Campo (VMaggia)", "Cimalmotto",
               "S. Carlo di Peccia", "Cortignelli", "Cortemezzano", "Veglia di Peccia",
               "Fontana (Bavona)", "Sabbione", "Ritorto (Bavona)", "Sonlerto",
               "Fontanellata", "Roseto (Bavona)", "Foroglio (Bavona)", "Niva (Vallemaggia)"]
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

    arcobaleno = gtfs[gtfs["zones"].notna()]
    
    arcobaleno["tarif network"] = "Arcobaleno"
    arcobaleno["local network"] = "Lugano"
    arcobaleno["zones"] = (
        arcobaleno["zones"]
        .fillna("")  # Handle NaNs
        .apply(lambda z: [int(x) for x in z.split("/") if x.strip().isdigit()] if z else [])
    )

    arcobaleno = arcobaleno[["stop_id", "stop_name", "geometry", "tarif network", "local network", "zones"]]

    return arcobaleno