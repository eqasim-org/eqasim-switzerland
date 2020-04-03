import os
from datetime import datetime

import jinja2
import yaml


def configure(context):
    pass

def min_date(dates):
    minimum = datetime.strptime("01/01/2050", "%d/%m/%Y")

    for date in dates:
        if date < minimum:
            minimum = date

    return minimum

def execute(context):
    this_path = os.path.dirname(os.path.abspath(__file__))

    with open("%s/../contracts.yml" % this_path) as f_contracts:
        data = yaml.load(f_contracts)

        datasets = data["datasets"]
        persons = data["persons"]
        contracts = data["contracts"]

        # Clean up contracts
        for contract in contracts:
            if not "groups" in contract:
                contract["groups"] = []

            if "group" in contract:
                contract["groups"].append(contract["group"])
                del contract["group"]

            contract["groups"] = set(contract["groups"])

            if not "persons" in contract:
                contract["persons"] = []

            if "person" in contract:
                contract["persons"].append(contract["person"])
                del contract["person"]

            contract["persons"] = set(contract["persons"])
            contract["until"] = datetime.strptime(contract["until"], "%d/%m/%Y")
            contract["is_active"] = contract["until"] > datetime.now()

        person_data = []

        # Analyze persons
        for person in persons:
            name, group = person["name"], person["group"]
            person_contracts = {}

            for contract in contracts:
                if contract["dataset"] in datasets.keys():
                    dataset = contract["dataset"]

                    if not dataset in person_contracts:
                        person_contracts[dataset] = datetime.strptime("01/01/1970", "%d/%m/%Y")

                    if name in contract["persons"] or group in contract["groups"]:
                        if contract["until"] > person_contracts[dataset]:
                            person_contracts[dataset] = contract["until"]

            person_until = min_date(person_contracts.values())

            person_data.append({
                "name": name, "group": group,
                "contracts": person_contracts, "until": person_until,
                "is_active": person_until > datetime.now()
            })

        with open("%s/template.html" % this_path) as f_template:
            template = jinja2.Template(f_template.read())
            result = template.render({
                "datasets": datasets,
                "contracts": contracts,
                "persons": person_data
            })

            output_path = "%s/CONTRACTS.html" % context.cache_path

            with open(output_path, "w+") as f_output:
                f_output.write(result)
                return output_path
