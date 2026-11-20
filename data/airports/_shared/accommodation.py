"""Shared STATENT definition for visitor accommodation and allocation weights."""


def select_accommodation(statent, columns):
    """Include hotels (551*) and other accommodation/dormitories (559*), including students."""
    codes = statent.noga.astype(str).str.replace(r'\.0$', '', regex=True)
    selected = statent.loc[codes.str.startswith(('551', '559')), columns].drop_duplicates().copy()
    selected['accommodation_type'] = codes.loc[selected.index].str.startswith('559').map(
        {True: 'dormitory', False: 'hotel'})
    if selected.enterprise_id.duplicated().any():
        raise ValueError('Accommodation establishment assigned to multiple locations or municipalities')
    return selected
