"""Read the final pre-simulation population using the production table contracts."""
import numpy as np
import shapely


def prepared_trips(population, households):
    """Match the MATSim analyzer's cohort, without writing or reading MATSim files."""

    # Attach household attributes without modifying the prepared tables.

    persons = population.persons.copy()
    persons = persons.loc[~persons.person_type.isin(['crossborder', 'freight'])].copy()
    persons['person_id'] = persons.person_id.astype(str)
    members = households.members.copy()
    members['person_id'] = members.person_id.astype(str)
    attributes = members.merge(households.households[['household_id', 'canton_id', 'canton_name', 'income_class']],
                               on='household_id', how='left', validate='many_to_one')
    persons = persons.drop(columns=['canton_id', 'canton_name', 'income_class'], errors='ignore').merge(
        attributes.drop(columns='household_id'), on='person_id', how='left', validate='one_to_one')

    # Adjacent main activities define the same trips as the population writer.

    activities = population.activities.sort_values(['person_id', 'activity_index']).reset_index(drop=True).copy()
    activities['person_id'] = activities.person_id.astype(str)
    following = activities.groupby('person_id', sort=False)[['geometry', 'purpose']].shift(-1)
    trips = activities.loc[~activities.is_last, ['person_id', 'activity_index', 'geometry', 'purpose', 'following_mode']].copy()
    trips['origin_x'] = shapely.get_x(trips.geometry.to_numpy())
    trips['origin_y'] = shapely.get_y(trips.geometry.to_numpy())
    trips['destination_x'] = shapely.get_x(following.loc[trips.index, 'geometry'].to_numpy())
    trips['destination_y'] = shapely.get_y(following.loc[trips.index, 'geometry'].to_numpy())
    trips['destination_purpose'] = following.loc[trips.index, 'purpose']
    airport = trips.purpose.astype('string').str.strip().str.lower().eq('airport') | trips.destination_purpose.astype('string').str.strip().str.lower().eq('airport')
    trips = trips.loc[~airport.fillna(False)].drop(columns=['geometry', 'purpose']).rename(
        columns={'destination_purpose':'purpose', 'following_mode':'mode'})
    trips['purpose'] = trips.purpose.str.replace('_secondary', '', regex=False)
    trips['trip_id'] = trips.person_id + '_' + trips.activity_index.astype(str)

    # Apply the common passenger-population mode, purpose and distance filters.

    trips['euclidean_distance_km'] = np.hypot(trips.destination_x-trips.origin_x, trips.destination_y-trips.origin_y)/1000
    trips = trips.loc[trips['mode'].isin(['car','pt','walk','bike','car_passenger'])
                      & trips.purpose.isin(['work','home','other','shop','education','leisure'])
                      & np.isfinite(trips.euclidean_distance_km) & trips.euclidean_distance_km.gt(.001)]
    trips = trips.merge(persons[['person_id', 'age', 'sex', 'home_x', 'home_y', 'income_class', 'canton_id', 'canton_name']],
                        on='person_id', how='inner', validate='many_to_one')
    trips['person_weight'] = 1.0

    # Return the analyzer's normalized, unweighted population trip schema.

    return trips.rename(columns={'person_id':'person'}).reset_index(drop=True)
