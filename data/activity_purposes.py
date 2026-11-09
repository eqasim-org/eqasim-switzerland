"""Shared microcensus activity labels and NOGA 2008 / OSM business compatibility."""

import re

import pandas as pd


SHOP = "shop"
LEISURE = "leisure"
# Retail, gastronomy, arts, museums/heritage, botanical/zoological gardens,
# gambling, sport/recreation and cinemas. Libraries and accommodation are other.
LEISURE_NOGA_PREFIXES = ("56", "90", "9102", "9103", "9104", "92", "93", "5914")
SHOP_TYPES = {
    "supermarket": "food_general", "convenience": "food_general", "general": "food_general",
    "department_store": "retail_general", "variety_store": "retail_general", "mall": "retail_general",
    "greengrocer": "fruit_vegetables", "farm": "food_other", "butcher": "butcher", "seafood": "fish",
    "bakery": "bakery", "pastry": "bakery", "confectionery": "bakery",
    "beverages": "beverages", "wine": "beverages", "alcohol": "beverages",
    "tobacco": "tobacco", "e-cigarette": "tobacco", "deli": "food_other", "cheese": "food_other",
    "tea": "food_other", "coffee": "food_other", "health_food": "food_other",
    "computer": "electronics", "electronics": "electronics", "mobile_phone": "telecom",
    "hifi": "audio_video", "fabric": "textiles", "curtain": "textiles",
    "hardware": "hardware", "doityourself": "hardware", "paint": "hardware",
    "flooring": "flooring", "carpet": "flooring", "appliance": "appliances",
    "furniture": "home_goods", "interior_decoration": "home_goods", "lighting": "home_goods",
    "musical_instrument": "instruments", "books": "books", "newsagent": "news_stationery",
    "kiosk": "news_stationery", "stationery": "news_stationery", "music": "music_video",
    "video": "music_video", "bicycle": "bicycle", "sports": "sports_retail", "outdoor": "sports_retail",
    "toys": "toys", "games": "toys", "clothes": "clothes", "fashion": "clothes",
    "shoes": "shoes", "bag": "shoes", "leather": "shoes", "chemist": "cosmetics",
    "cosmetics": "cosmetics", "perfumery": "cosmetics", "medical_supply": "medical",
    "hearing_aids": "medical", "optician": "optician", "florist": "plants_pets",
    "garden_centre": "plants_pets", "pet": "plants_pets", "jewelry": "jewelry_watches",
    "watches": "jewelry_watches", "second_hand": "second_hand", "antiques": "second_hand",
    "photo": "photographic", "gift": "gifts", "art": "art_retail",
}
NON_RETAIL_SHOPS = {
    "hairdresser", "beauty", "massage", "tattoo", "car", "car_repair", "car_parts",
    "motorcycle", "motorcycle_repair", "wholesale", "storage_rental", "rental",
    "travel_agency", "insurance", "estate_agent", "funeral_directors", "copyshop",
    "laundry", "dry_cleaning", "tailor", "shoemaker", "photo_studio", "locksmith", "printing",
    "caterer", "vacant", "closed", "disused", "abandoned", "no", "construction", "proposed",
}
AMENITY_TYPES = {
    "pharmacy": (SHOP, "pharmacy"), "fuel": (SHOP, "fuel"),
    "restaurant": (LEISURE, "restaurant"), "cafe": (LEISURE, "restaurant"),
    "fast_food": (LEISURE, "restaurant"), "food_court": (LEISURE, "restaurant"),
    "ice_cream": (LEISURE, "restaurant"), "bar": (LEISURE, "bar"),
    "pub": (LEISURE, "bar"), "biergarten": (LEISURE, "bar"), "nightclub": (LEISURE, "bar"),
    "cinema": (LEISURE, "cinema"), "theatre": (LEISURE, "culture"),
    "arts_centre": (LEISURE, "culture"), "casino": (LEISURE, "gambling"),
}
LEISURE_TYPES = {
    "fitness_centre": "fitness", "sports_centre": "sport", "stadium": "sport",
    "swimming_pool": "sport", "ice_rink": "sport", "golf_course": "sport",
    "bowling_alley": "recreation", "amusement_arcade": "recreation", "dance": "recreation",
    "escape_game": "recreation", "water_park": "recreation", "miniature_golf": "recreation",
    "trampoline_park": "recreation",
}
TOURISM_TYPES = {"museum": "museum", "gallery": "culture", "zoo": "zoo_garden", "aquarium": "zoo_garden", "theme_park": "recreation"}
# Longest prefixes override broad classes; avoid claiming six-digit NOGA
# specificity where the OSM tags cannot distinguish it.
NOGA_TYPES = {
    "4711": "food_general", "4719": "retail_general", "4721": "fruit_vegetables",
    "4722": "butcher", "4723": "fish", "4724": "bakery", "4725": "beverages",
    "4726": "tobacco", "4729": "food_other", "473": "fuel", "4741": "electronics",
    "4742": "telecom", "4743": "audio_video", "4751": "textiles", "4752": "hardware",
    "4753": "flooring", "4754": "appliances", "475901": "instruments", "4759": "home_goods",
    "4761": "books", "4762": "news_stationery", "4763": "music_video",
    "476401": "bicycle", "4764": "sports_retail", "4765": "toys", "4771": "clothes",
    "4772": "shoes", "4773": "pharmacy", "4774": "medical", "4775": "cosmetics",
    "4776": "plants_pets", "4777": "jewelry_watches", "477802": "optician",
    "477803": "photographic", "477804": "gifts", "477805": "art_retail", "4779": "second_hand",
    "561": "restaurant", "562": "catering", "563": "bar", "5914": "cinema",
    "90": "culture", "9102": "museum", "9103": "heritage", "9104": "zoo_garden",
    "92": "gambling", "9313": "fitness", "931": "sport", "932": "recreation",
}
NOGA_TYPE_PREFIXES = tuple(sorted(NOGA_TYPES, key=len, reverse=True))


def normalize_noga(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\.0+$", "", text)
    text = re.sub(r"[.\s]", "", text)
    return text if re.fullmatch(r"\d{2,6}", text) else ""


def noga_activity_types(value):
    code = normalize_noga(value)
    purpose = SHOP if code.startswith("47") else LEISURE if code.startswith(LEISURE_NOGA_PREFIXES) else None
    if purpose is None:
        return {}
    for prefix in NOGA_TYPE_PREFIXES:
        if code.startswith(prefix):
            return {purpose: NOGA_TYPES[prefix]}
    return {purpose: purpose}  # The broad purpose is known, the subtype is not.


def statent_activity_masks(noga):
    codes = noga.map(normalize_noga)
    return codes.str.startswith("47"), codes.str.startswith(LEISURE_NOGA_PREFIXES)


def osm_activity_types(tags):
    """Return purpose-specific business groups, aligned to NOGA 2008 offers."""
    result = {SHOP: set(), LEISURE: set()}
    if any(tags.get(key) == "yes" for key in ("disused", "abandoned", "construction")):
        return result
    for shop in tags.get("shop", "").split(";"):
        shop = shop.strip()
        if shop and shop not in NON_RETAIL_SHOPS:
            result[SHOP].add(SHOP_TYPES.get(shop, SHOP))
        elif shop == "caterer":
            result[LEISURE].add("catering")
    for amenity in tags.get("amenity", "").split(";"):
        if amenity.strip() in AMENITY_TYPES:
            purpose, group = AMENITY_TYPES[amenity.strip()]
            result[purpose].add(group)
    for key, mapping in (("leisure", LEISURE_TYPES), ("tourism", TOURISM_TYPES)):
        for value in tags.get(key, "").split(";"):
            if value.strip() in mapping:
                result[LEISURE].add(mapping[value.strip()])
    if tags.get("historic") in {"castle", "monument", "archaeological_site", "ruins", "building"}:
        result[LEISURE].add("heritage")
    if tags.get("leisure") == "garden" and tags.get("garden:type", tags.get("garden")) == "botanical":
        result[LEISURE].add("zoo_garden")
    return result
