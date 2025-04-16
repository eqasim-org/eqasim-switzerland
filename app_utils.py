import plotly.graph_objects as go
import numpy as np
import datetime

def rename_labels(labels, prefix=''):
    return [prefix + label.capitalize() for label in labels]

def convert_sequence_names(names):
    return "-".join([name[0].upper() for name in names])

def get_weighted_num_activities(activities):
    summary = activities.groupby('person_id').agg(
        number_of_act=('activity_index', 'count'),
        weight=('person_weight', 'first')
    ).reset_index()

    summary.columns = ['person_id', 'num_activities', 'person_weight']
    total = summary['person_weight'].sum()
    freq = summary.groupby('num_activities').sum('person_weight').reset_index()

    freq['person_weight'] = freq['person_weight'] / total

    pairs = sorted(zip(freq['num_activities'], freq['person_weight']), key=lambda x: x[0])
    num_activities = [pair[0] for pair in pairs]
    proportions = [pair[1] for pair in pairs]

    return num_activities, proportions

def frequent_weighted_sequences(data):
    df = data.groupby('person_id').agg({
        'purpose': lambda x: convert_sequence_names(x),
        'person_weight': 'first'
    }).reset_index()

    df.columns = ['person_id', 'activities', 'person_weight']
    df = df.groupby('activities').sum('person_weight').reset_index()

    pairs = sorted(zip(df['activities'], df['person_weight']), key=lambda x: -x[1])[:9]
    total = sum([x[1] for x in pairs])
    sequence = [pair[0] for pair in pairs]
    proportions = [pair[1] / total for pair in pairs]

    return sequence, proportions

def frequent_out_of_home_activities(data):
    df = data[data['purpose'] != 'home']
    df = df.groupby(['person_id', 'purpose']).agg(
        count=('person_weight', 'count'),
        person_weight=('person_weight', 'first')
    ).reset_index()

    df['purpose'] = df['purpose'].apply(lambda x: x[0].upper())
    df['sequence'] = df['count'].astype(str) + df['purpose']
    df = df.groupby('sequence').sum('person_weight').reset_index()

    pairs = sorted(zip(df['sequence'], df['person_weight']), key=lambda x: -x[1])[:10]
    total = sum([x[1] for x in pairs])
    sequence = [pair[0] for pair in pairs]
    proportions = [pair[1] / total for pair in pairs]

    return sequence, proportions

def get_car_availability(data):
    data = data.groupby('number_of_cars_class').sum('household_weight').reset_index()
    total = data['household_weight'].sum()

    pairs = sorted(zip(data['number_of_cars_class'], data['household_weight']), key=lambda x: x[0])
    cars_class = [pair[0] for pair in pairs]
    proportions = [pair[1] / total for pair in pairs]

    return cars_class, proportions

def get_individual_car_class(data, feature=None, bins=None):
    data = data.groupby('number_of_cars_class').sum('person_weight').reset_index()
    total = data['person_weight'].sum()

    pairs = sorted(zip(data['number_of_cars_class'], data['person_weight']), key=lambda x: x[0])
    cars_class = [pair[0] for pair in pairs]
    proportions = [pair[1] / total for pair in pairs]

    return cars_class, proportions

def get_subscription_proportions(data, feature=None, bins=None):
    weighted_proportions = {}
    total_weight = data['person_weight'].sum()

    subscriptions = [
        'subscriptions_verbund', 'subscriptions_halbtax', 'subscriptions_strecke',
        'subscriptions_gleis7', 'subscriptions_junior', 'subscriptions_other', 'subscriptions_ga'
    ]

    values = []
    for cat in subscriptions:
        weighted_sum = (data[cat] * data['person_weight']).sum()
        proportion = weighted_sum / total_weight
        weighted_proportions[cat] = proportion
        values.append(proportion)

    subscriptions = [sub.split("_")[1].capitalize() for sub in subscriptions]
    return subscriptions, values

def get_histogram_time(data, feature, bins=None):
    times = data[feature]
    bins = np.arange(0, 86401, 1800)
    hist, bin_edges = np.histogram(times, bins=bins, weights=data['person_weight'], density=True)
    bin_labels = [str(datetime.timedelta(seconds=int(edge))) for edge in bin_edges]

    return bin_labels, hist

def get_histogram(data, feature, bins=None):
    hist, bin_edges = np.histogram(data[feature], bins=bins, weights=data['person_weight'], density=True)
    bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bins, hist

def generate_car_availability_distribution(persons, output_persons, attribute_values, attribute_name):
    fig = go.Figure()

    for attr_value in attribute_values:
        filtered_micro = persons.loc[persons[attribute_name] == attr_value]
        filtered_output = output_persons.loc[output_persons[attribute_name] == attr_value]

        micro_x, micro_y = get_individual_car_class(filtered_micro)
        output_x, output_y = get_individual_car_class(filtered_output)

        fig.add_trace(go.Bar(x=micro_x, y=micro_y, name=f'Microcensus ({attr_value})', marker=dict(color="blue"), visible=False))
        fig.add_trace(go.Bar(x=output_x, y=output_y, name=f'Output ({attr_value})', marker=dict(color="red"), visible=False))

    fig.data[0].visible = True
    fig.data[1].visible = True

    attribute_name = " ".join(attribute_name.split('_')).title()

    buttons = []
    for i, param in enumerate(attribute_values):
        param = str(param).capitalize()
        visibility = [False] * (2 * len(attribute_values))
        visibility[2 * i] = True
        visibility[2 * i + 1] = True

        buttons.append(dict(
            label=param,
            method="update",
            args=[{"visible": visibility}, {"title": f"Number of Cars Class - {attribute_name} {param}"}]
        ))

    fig.update_layout(
        title=f'Number of Cars Class - {attribute_name}',
        xaxis_title='Number of Cars Class',
        yaxis_title='Proportion',
        barmode='group',
        legend_title='Legend',
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
        }]
    )

    return fig

def generate_pt_distribution(persons, output_persons, attribute_values, attribute_name):
    fig = go.Figure()

    for idx, attr_value in enumerate(attribute_values):
        filtered_micro = persons.loc[persons[attribute_name] == attr_value]
        filtered_output = output_persons.loc[output_persons[attribute_name] == attr_value]

        micro_x, micro_y = get_subscription_proportions(filtered_micro)
        output_x, output_y = get_subscription_proportions(filtered_output)

        fig.add_trace(go.Bar(x=micro_x, y=micro_y, name=f'Microcensus ({attr_value})', marker=dict(color="blue"), visible=False))
        fig.add_trace(go.Bar(x=output_x, y=output_y, name=f'Output ({attr_value})', marker=dict(color="red"), visible=False))

    fig.data[0].visible = True
    fig.data[1].visible = True

    attribute_name = " ".join(attribute_name.split('_')).title()

    buttons = []
    for i, param in enumerate(attribute_values):
        param = str(param).capitalize()
        visibility = [False] * (2 * len(attribute_values))
        visibility[2 * i] = True
        visibility[2 * i + 1] = True

        buttons.append(dict(
            label=param,
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"Public Transport Subscription - {attribute_name} {param}"}
            ]
        ))

    fig.update_layout(
        title=f'Public Transport Subscriptions - {attribute_name}',
        xaxis_title='Subscriptions',
        yaxis_title='Proportion',
        barmode='group',
        legend_title='Legend',
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
        }]
    )

    return fig
