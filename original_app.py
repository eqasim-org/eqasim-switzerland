import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from app_utils import *

# Initialize the Dash app
app = dash.Dash(__name__)

prefix = '/cluster/project/cmdp/chaoch/'
activities = pd.read_csv(f'{prefix}microcensus_data/microcensus_act_geo.csv', sep=';', header=0)
trips = pd.read_csv(f'{prefix}microcensus_data/microcensus_trips_geo.csv', sep=',', header=0)
households = pd.read_csv(f'{prefix}microcensus_data/microcensus_households_geo.csv', sep=',', header=0)
persons = pd.read_csv(f'{prefix}microcensus_data/microcensus_persons_geo.csv', sep=',', header=0)

output_activities = pd.read_csv('/cluster/project/cmdp/chaoch/switzerland_data/output/switzerland_activities_geo.csv', sep=',', header=0)
output_trips = pd.read_csv('/cluster/project/cmdp/chaoch/switzerland_data/output/switzerland_trips_geo.csv', sep=',', header=0)
output_households = pd.read_csv('/cluster/project/cmdp/chaoch/switzerland_data/output/switzerland_households_geo.csv', sep=',', header=0)
output_persons = pd.read_csv('/cluster/project/cmdp/chaoch/switzerland_data/output/switzerland_persons_geo.csv', sep=';', header=0)
output_households['household_weight'] = 1

def num_activities(activities, output_activities):
    act_micro, freq_micro = get_weighted_num_activities(activities)
    act_out, freq_out = get_weighted_num_activities(output_activities)

    percent_differences = [abs(output - microcensus) for microcensus, output in zip(freq_micro, freq_out)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=act_micro, y=freq_micro, name='Microcensus', marker=dict(color="blue")))
    fig.add_trace(go.Bar(x=act_out, y=freq_out, name='Output', marker=dict(color="red")))    

    x_values = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    for i, x_val in enumerate(x_values):
        diff = percent_differences[i]
        text = f"{diff:.3f}"
        fig.add_annotation(x=x_val, y=max(freq_micro[i], freq_out[i]) + 0.02,
                            text=text, showarrow=False, font=dict(size=10, color='black'))
    
    fig.update_layout(
        title='Number of Activities',
        xaxis_title='Number of Activities',
        yaxis_title='Frequency',
        barmode='group',
        legend_title='Legend'
    )
    return fig

def distribution_frequent_sequences(activities, output_activities):
    act_micro, freq_micro = frequent_weighted_sequences(activities)
    act_out, freq_out = frequent_weighted_sequences(output_activities)

    fig = go.Figure(data=[
        go.Bar(name='Microcensus', x=act_micro, y=freq_micro, marker_color='blue'),
        go.Bar(name='Output', x=act_out, y=freq_out, marker_color='red')
    ])

    percent_differences = [abs(output - microcensus) for microcensus, output in zip(freq_micro, freq_out)]
    
    for i, x_val in enumerate(act_micro):
        diff = percent_differences[i]
        text = f"{diff:.3f}"
        fig.add_annotation(x=x_val, y=max(freq_micro[i], freq_out[i]) + 0.02,
                            text=text, showarrow=False, font=dict(size=10, color='black'))

    fig.update_layout(
        barmode='group',
        title_text="Comparison of Activity Sequences",
        xaxis_title="Activity Sequence",
        yaxis_title="Proportion",
        xaxis=dict(tickangle=45),
    )
    return fig

def distribution_out_of_home(activities, output_activities):
    act_micro, freq_micro = frequent_out_of_home_activities(activities)
    act_out, freq_out = frequent_out_of_home_activities(output_activities)

    fig = go.Figure(data=[
        go.Bar(name='Microcensus', x=act_micro, y=freq_micro, marker_color='blue'),
        go.Bar(name='Output', x=act_out, y=freq_out, marker_color='red')
    ])

    percent_differences = [abs(output - microcensus) for microcensus, output in zip(freq_micro, freq_out)]
    
    for i, x_val in enumerate(act_micro):
        diff = percent_differences[i]
        text = f"{diff:.3f}"
        fig.add_annotation(x=x_val, y=max(freq_micro[i], freq_out[i]) + 0.02,
                            text=text, showarrow=False, font=dict(size=10, color='black'))

    fig.update_layout(
        barmode='group',
        title_text="Frequent Out of Home Activity Proportions",
        xaxis_title="Out of Home Activities",
        yaxis_title="Proportion",
        xaxis=dict(tickangle=45),
    )
    return fig

def car_availability_class_comparison(households, output_households):
    household_micro, freq_micro = get_car_availability(households)
    household_out, freq_out = get_car_availability(output_households)

    percent_differences = [abs(output - microcensus) for microcensus, output in zip(freq_micro, freq_out)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=household_micro, y=freq_micro, name='Microcensus', marker=dict(color="blue")))
    fig.add_trace(go.Bar(x=household_out, y=freq_out, name='Output', marker=dict(color="red")))

    # Add percentage difference annotations
    for i, x_val in enumerate(household_micro):
        diff = percent_differences[i]
        text = f"{diff:.3f}"
        fig.add_annotation(x=x_val, y=max(freq_micro[i], freq_out[i]) + 0.02,
                            text=text, showarrow=False, font=dict(size=10, color='black'))
    
    fig.update_layout(
        title='Available Cars Class',
        xaxis_title='Cars Class',
        yaxis_title='Proportion',
        barmode='group',
        legend_title='Legend'
    )
    return fig


def pt_subscription_distribution_general(persons, output_persons):
    person_micro, freq_micro = get_subscription_proportions(persons)
    person_out, freq_out = get_subscription_proportions(output_persons)

    percent_differences = [abs(output - microcensus) for microcensus, output in zip(freq_micro, freq_out)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=person_micro, y=freq_micro, name='Microcensus', marker=dict(color="blue")))
    fig.add_trace(go.Bar(x=person_out, y=freq_out, name='Output', marker=dict(color="red")))

    for i, x_val in enumerate(person_micro):
        diff = percent_differences[i]
        text = f"{diff:.3f}"
        fig.add_annotation(x=x_val, y=max(freq_micro[i], freq_out[i]) + 0.02,
                            text=text, showarrow=False, font=dict(size=10, color='black'))
    
    fig.update_layout(
        title='Public Transport Subcriptions',
        xaxis_title='Subscriptions',
        yaxis_title='Proportion',
        barmode='group',
        legend_title='Legend'
    )
    return fig

def car_availability_distribution_income(persons, output_persons):
    attribute_values = persons[persons['household_income'] >= 0]['household_income'].unique()
    attribute_name = 'household_income'

    return generate_car_availability_distribution(persons, output_persons, sorted(attribute_values), attribute_name)

def car_availability_distribution_gender(persons, output_persons):
    attribute_values = [0, 1]
    attribute_name = 'sex'

    return generate_car_availability_distribution(persons, output_persons, sorted(attribute_values), attribute_name)

def car_availability_distribution_age(persons, output_persons):
    bins = [6, 15, 18, 24, 30, 45, 65, 80]
    labels = ['7-15', '16-18', '19-24', '25-30', '31-45', '46-65', '66-80']
    persons['age_group'] = pd.cut(persons['age'], bins=bins, labels=labels, right=False)
    output_persons['age_group'] = pd.cut(output_persons['age'], bins=bins, labels=labels, right=False)

    attribute_values = labels
    attribute_name = 'age_group'
    return generate_car_availability_distribution(persons, output_persons, sorted(attribute_values), attribute_name)


def pt_subscription_distribution_income(persons, output_persons, output_households):
    # NOTE using household weight doesn't solve the issue
    attribute_values = persons[persons['household_income'] >= 0]['household_income'].unique()
    attribute_name = 'household_income'

    return generate_pt_distribution(persons, output_persons, sorted(attribute_values), attribute_name)

def pt_subscription_distribution_gender(persons, output_persons):
    attribute_values = [0, 1]
    attribute_name = 'sex'

    return generate_pt_distribution(persons, output_persons, sorted(attribute_values), attribute_name)

def pt_subscription_distribution_age(persons, output_persons):
    bins = [6, 15, 18, 24, 30, 45, 65, 80]
    labels = ['[6, 15)', '[15, 18)', '[18, 24)', '[24, 30)', '[30, 45)', '[45, 65)', '[65, 80)']
    
    persons['age_group'] = pd.cut(persons['age'], bins=bins, labels=labels, right=False)
    output_persons['age_group'] = pd.cut(output_persons['age'], bins=bins, labels=labels, right=False)

    attribute_values = labels
    attribute_name = 'age_group'
    return generate_pt_distribution(persons, output_persons, sorted(attribute_values), attribute_name)


def trip_crowfly_distance(trips, output_trips):
    output_trips['person_weight'] = 1

    parameter_options = ['All', 'home', 'work', 'leisure', 'shop', 'other', 'education']

    fig = go.Figure()

    for param in parameter_options:
        # Independently determine bins for each purpose
        if param == 'All':
            trips_filtered = trips
            output_trips_filtered = output_trips
        else:
            trips_filtered = trips.loc[trips['following_purpose'] == param]
            output_trips_filtered = output_trips.loc[output_trips['following_purpose'] == param]

        min_dist = np.percentile(trips_filtered['crowfly_distance'], 5)
        max_dist = np.percentile(trips_filtered['crowfly_distance'], 95)

        bins = np.linspace(min_dist, max_dist, num=40)
        bins_micro, hist_micro = get_histogram(trips_filtered, feature='crowfly_distance', bins=bins)
        bins_out, hist_out = get_histogram(output_trips_filtered, feature='crowfly_distance', bins=bins)

        fig.add_trace(go.Bar(x=bins_micro, y=hist_micro, name=f'Microcensus ({param})', marker=dict(color="blue"), visible=(param == "All")))
        fig.add_trace(go.Bar(x=bins_out, y=hist_out, name=f'Output ({param})', marker=dict(color="red"), visible=(param == "All")))

    buttons = []
    for i, param in enumerate(parameter_options):
        param = param.capitalize()
        visibility = [False] * (2 * len(parameter_options))
        visibility[2*i] = True
        visibility[2*i + 1] = True

        buttons.append(dict(
            label=param,
            method="update",
            args=[{"visible": visibility}, {"title": f"Trip Crowfly Distance - {param}"}]
        ))

    fig.update_layout(
        title="Trip Crowfly Distance - All",
        xaxis_title="Distance [km]",
        yaxis_title="Proportion",
        barmode="group",
        legend_title="Legend",
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
        }]
    )

    return fig

def activity_duration_distribution(activity, output_activity):
    for df in [activity, output_activity]:
        df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
        df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")
        df["duration"] = df["end_time"] - df["start_time"]
    
    activity = activity.dropna(subset=["duration"])
    output_activity = output_activity.dropna(subset=["duration"])

    activity_types = sorted(set(activity["purpose"].unique()).union(output_activity["purpose"].unique()))
    activity_types.append('All')

    fig = go.Figure()

    for activity_type in activity_types:
        if activity_type == 'All':
            activity_micro = activity
            activity_output = output_activity
        else:
            activity_micro = activity[activity["purpose"] == activity_type]
            activity_output = output_activity[output_activity["purpose"] == activity_type]


        bins_micro, hist_micro = get_histogram_time(activity_micro, feature='duration')
        bins_out, hist_out = get_histogram_time(activity_output, feature='duration')

        fig.add_trace(go.Bar(x=bins_micro, y=hist_micro, name=f'Microcensus ({activity_type})', marker=dict(color="blue"), visible=(activity_type == "All")))
        fig.add_trace(go.Bar(x=bins_out, y=hist_out, name=f'Output ({activity_type})', marker=dict(color="red"), visible=(activity_type == "All")))

    buttons = []
    for i, param in enumerate(activity_types):
        param = param.capitalize()
        visibility = [False] * (2 * len(activity_types))
        visibility[2*i] = True
        visibility[2*i + 1] = True

        buttons.append(dict(
            label=param,
            method="update",
            args=[{"visible": visibility}, {"title": f"Activity Duration - {param}"}]
        ))

    fig.update_layout(
        title="Distribution of Activity Durations",
        xaxis_title="Activity Duration [seconds]",
        yaxis_title="Proportion",
        barmode="group",
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
        }]
    )

    return fig


def distribution_departure_times(trips, output_trips):
    output_trips['weight'] = 1
    	
    purpose_types = sorted(trips['following_purpose'].unique())
    purpose_types.append('All')

    fig = go.Figure()
    for purpose in purpose_types:
        if purpose == 'All':
            trips_micro = trips
            trips_output = output_trips
        else:
            trips_micro = trips[trips['following_purpose'] == purpose]
            trips_output = output_trips[output_trips['following_purpose'] == purpose]
        
        bins_micro, hist_micro = get_histogram_time(trips_micro, feature='departure_time')
        bins_out, hist_out = get_histogram_time(trips_output, feature='departure_time')

        fig.add_trace(go.Bar(x=bins_micro, y=hist_micro, name=f'Microcensus ({purpose})', marker=dict(color="blue"), visible=(purpose == "All")))
        fig.add_trace(go.Bar(x=bins_out, y=hist_out, name=f'Output ({purpose})', marker=dict(color="red"), visible=(purpose == "All")))

    buttons = []
    for i, param in enumerate(purpose_types):
        param = param.capitalize()
        visibility = [False] * (2 * len(purpose_types))
        visibility[2*i] = True
        visibility[2*i + 1] = True

        buttons.append(dict(
            label=param,
            method="update",
            args=[{"visible": visibility}, {"title": f"Departure Time - {param}"}]
        ))

    fig.update_layout(
        title="Distribution of Departure Times",
        xaxis_title="Departure Time [HH:MM:SS]",
        yaxis_title="Proportion",
        barmode="group",
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
        }]
    )

    return fig

def generate_html(canton_name):
    activities = pd.read_csv('microcensus_data/microcensus_act_geo.csv', sep=';', header=0).query(f"home_canton == '{canton_name}'")
    trips = pd.read_csv('microcensus_data/microcensus_trips_geo.csv', sep=',', header=0).query(f"home_canton == '{canton_name}'")
    households = pd.read_csv('microcensus_data/microcensus_households_geo.csv', sep=',', header=0).query(f"canton_name == '{canton_name}'")
    persons = pd.read_csv('microcensus_data/microcensus_persons_geo.csv', sep=',', header=0).query(f"canton_name == '{canton_name}'")

    output_activities = pd.read_csv('/cluster/home/chaoch/ch/switzerland_data/output/switzerland_activities_geo.csv', sep=',', header=0).query(f"home_canton == '{canton_name}'")
    output_trips = pd.read_csv('/cluster/home/chaoch/ch/switzerland_data/output/switzerland_trips_geo.csv', sep=',', header=0).query(f"home_canton == '{canton_name}'")
    output_households = pd.read_csv('/cluster/home/chaoch/ch/switzerland_data/output/switzerland_households_geo.csv', sep=',', header=0).query(f"canton_name == '{canton_name}'")
    output_persons = pd.read_csv('/cluster/home/chaoch/ch/switzerland_data/output/switzerland_persons_geo.csv', sep=',', header=0).query(f"canton_name == '{canton_name}'")

    return [
        dcc.Graph(
            id="bar-graph",
            figure=num_activities(activities, output_activities)
        ),
        dcc.Graph(
            id="frequent-sequences-bar-graph",
            figure=distribution_frequent_sequences(activities, output_activities)
        ),
        dcc.Graph(
            id='out-of-home-graph',
            figure=distribution_out_of_home(activities, output_activities)
        ),
        dcc.Graph(
            id='car-availability-graph',
            figure=car_availability_class_comparison(households, output_households)
        ),
        dcc.Graph(
            id='car-graph-income',
            figure=car_availability_distribution_income(persons, output_persons, output_households)
        ),
        dcc.Graph(
            id='car-graph-gender',
            figure=car_availability_distribution_gender(persons, output_persons)
        ),
        dcc.Graph(
            id='car-graph-age',
            figure=car_availability_distribution_age(persons, output_persons)
        ),
        dcc.Graph(
            id='pt-subscriptions-graph-general',
            figure=pt_subscription_distribution_general(persons, output_persons)
        ),
        dcc.Graph(
            id='pt-subscriptions-graph-income',
            figure=pt_subscription_distribution_income(persons, output_persons, output_households)
        ),
        dcc.Graph(
            id='pt-subscriptions-graph-gender',
            figure=pt_subscription_distribution_gender(persons, output_persons)
        ),
        dcc.Graph(
            id='pt-subscriptions-graph-age',
            figure=pt_subscription_distribution_age(persons, output_persons)
        ),
        dcc.Graph(
            id='trip-distance',
            figure=trip_crowfly_distance(trips, output_trips)
        ),
        dcc.Graph(
            id='activity-duration',
            figure=activity_duration_distribution(activities, output_activities)
        ),
        dcc.Graph(
            id='departure-times',
            figure=distribution_departure_times(trips, output_trips)
        )
    ]

app.layout = html.Div(children=[
    html.H1("Microcensus vs. Output (Synthetic Data)"),
    html.P("Comparison but real (microcensus) data and synthetic (output) data"),
    dcc.Graph(
        id="bar-graph",
        figure=num_activities(activities, output_activities)
    ),
     dcc.Graph(
        id="frequent-sequences-bar-graph",
        figure=distribution_frequent_sequences(activities, output_activities)
    ),
    dcc.Graph(
        id='out-of-home-graph',
        figure=distribution_out_of_home(activities, output_activities)
    ),
    dcc.Graph(
        id='car-availability-graph',
        figure=car_availability_class_comparison(households, output_households)
    ),
    dcc.Graph(
        id='car-graph-income',
        figure=car_availability_distribution_income(persons, output_persons, output_households)
    ),
    dcc.Graph(
        id='car-graph-gender',
        figure=car_availability_distribution_gender(persons, output_persons)
    ),
    dcc.Graph(
        id='car-graph-age',
        figure=car_availability_distribution_age(persons, output_persons)
    ),
    dcc.Graph(
        id='pt-subscriptions-graph-general',
        figure=pt_subscription_distribution_general(persons, output_persons)
    ),
    dcc.Graph(
        id='pt-subscriptions-graph-income',
        figure=pt_subscription_distribution_income(persons, output_persons, output_households )
    ),
    dcc.Graph(
        id='pt-subscriptions-graph-gender',
        figure=pt_subscription_distribution_gender(persons, output_persons)
    ),
    dcc.Graph(
        id='pt-subscriptions-graph-age',
        figure=pt_subscription_distribution_age(persons, output_persons)
    ),
    dcc.Graph(
        id='trip-distance',
        figure=trip_crowfly_distance(trips, output_trips)
    ),
    dcc.Graph(
        id='activity-duration',
        figure=activity_duration_distribution(activities, output_activities)
    ),
    dcc.Graph(
        id='departure-times',
        figure=distribution_departure_times(trips, output_trips)
    )
])

# Run the server
if __name__ == "__main__":
    app.run(debug=True)
