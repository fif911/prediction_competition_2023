import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

cm_features = pd.read_csv('data/cm_features_v0.6.csv')
# sort by month_id and country_id
cm_features = cm_features.sort_values(by=['month_id', 'country_id'])

# read additional data
rivalries = pd.read_csv('data/dyad/rivalries.csv')
majors = pd.read_csv('data/dyad/maoz_powers.csv')
min_dist = pd.read_csv('data/dyad/min_dist_dyads.csv')

base_shift = 3
# columns from ged_sb_y_4 to ged_sb_y_18
months_ahead_columns = {f'ged_sb_y_{i}' for i in range(1 + base_shift, 16 + base_shift)}

# Features to exclude from the dyadic DataFrame
features_exclude = ['date', 'country_id', 'month_id', 'country']  # Excluding 'country_name' to handle separately

# Features to create ratios for
features_for_ratios = ['wdi_sp_pop_totl', 'ged_sb', 'wdi_sp_dyn_imrt_in', 'vdem_v2x_ex_military', 'wdi_ms_mil_xpnd_zs']

no_riv = pd.DataFrame({
    'has_active_riv': 0,
    'years_since_riv_start': 0,
    'principal': 0,
    'asymmetric_principal': 0,
    # 'a_is_initiator': 0,
    'positional': 0,
    'spatial': 0,
    'ideological': 0,
    'interventionary': 0,
}, index=[0])


def get_rivalry_data(ccode_i, ccode_j, year, rivalries):
    conf = rivalries[((rivalries['ccode1'] == ccode_i) & (rivalries['ccode2'] == ccode_j) | (
            (rivalries['ccode1'] == ccode_j) & (rivalries['ccode2'] == ccode_i))) & (rivalries['start'] <= year)]

    # sort by end date to get the most recent rivalry
    conf = conf.sort_values(by='end', ascending=False).iloc[:1]
    if conf.empty:
        return False, no_riv

    # if more then 1 found - error
    # if len(conf) > 1:
    #     raise ValueError(f"Multiple rivalries found for {ccode_i} and {ccode_j}")

    # get amount of years for rivalry based on start
    active_riv = conf[conf['end'] >= year]
    has_active_riv = not active_riv.empty
    if has_active_riv:
        years_since_riv_start = year - active_riv['start'].values[0]
        # only one of the two rivals sees the other as its primary rival
        asymmetric_principal = int(conf['aprin'].values[0])
        # Create riv_metadata DataFrame
        riv_metadata = pd.DataFrame({
            'has_active_riv': int(has_active_riv),
            'years_since_riv_start': years_since_riv_start,
            'principal': conf['principal'].values[0],  # principal is the principal rival
            'asymmetric_principal': asymmetric_principal,
            # 'a_is_initiator': int(conf['ccode1'].values[0] == ccode_i if asymmetric_principal else 0),
            'positional': conf['positional'].values[0],
            'spatial': conf['spatial'].values[0],
            'ideological': conf['ideological'].values[0],
            'interventionary': conf['interventionary'].values[0],
        }, index=[0])
    else:
        # Create riv_metadata DataFrame
        riv_metadata = no_riv

    # True as there was at least one rivalry started before specified year
    return True, riv_metadata


def sort_dyadic_columns(dyadic_df: pd.DataFrame, base_month: int = 4, end_month: int = 18) -> pd.DataFrame:
    """
    Sorts the dyadic DataFrame columns to have 'a_' and 'b_' prefixed ged_sb columns
    in sequential order from the base_month to the end_month.

    Parameters:
    - dyadic_df: DataFrame containing the dyadic data.
    - base_month: The starting month for the ged_sb columns.
    - end_month: The ending month for the ged_sb columns.

    Returns:
    - A DataFrame with columns sorted as specified.
    """

    # Define months_ahead_columns based on the provided range
    months_ahead_columns = [f'ged_sb_y_{i}' for i in range(base_month, end_month + 1)]

    # Combine 'a_' and 'b_' prefix columns
    months_ahead_columns_c = [f'a_{col}' for col in months_ahead_columns] + [f'b_{col}' for col in months_ahead_columns]

    # Sort the months_ahead_columns_c based on the numeric part of the column names
    months_ahead_columns_c_sorted = sorted(
        months_ahead_columns_c,
        key=lambda x: (int(x.split('_')[-1]), x.split('_')[0])
    )

    # Identify all other columns that will remain at the beginning
    other_columns = [col for col in dyadic_df.columns if col not in months_ahead_columns_c]

    # Create the new column order
    new_order = other_columns + months_ahead_columns_c_sorted

    # Reorder the DataFrame columns
    return dyadic_df[new_order]


def process_month_data(month, cm_features: pd.DataFrame, features_exclude, features_for_ratios,
                       majors: pd.DataFrame) -> pd.DataFrame:
    print("Processing month: ", month)
    dyadic_data = []

    cm_features_month = cm_features[cm_features['month_id'] == month]
    countries = cm_features_month['country_id'].unique()
    date: str = cm_features_month['date'].values[0]
    year: int = cm_features_month['year'].values[0]
    min_dist_double_year = min_dist[(min_dist['year'] == year) | (min_dist['year'] == year + 1)]
    rivalries_until_year = rivalries[rivalries['start'] <= year]

    for i in range(len(countries)):
        print(f"Processing month: {month}, country {i + 1}/{len(countries)}")

        # for j in range(i + 1, len(countries)):
        for j in range(len(countries)):
            if i == j:
                continue
            country_i = countries[i]
            country_j = countries[j]

            # Filter data for each country and drop excluded features
            data_i = cm_features_month[cm_features_month['country_id'] == country_i]
            data_j = cm_features_month[cm_features_month['country_id'] == country_j]

            ccode_i = data_i['ccode'].values[0]
            ccode_j = data_j['ccode'].values[0]

            # Get country names for both countries
            country_name_i = data_i['gw_statename'].values[0]
            country_name_j = data_j['gw_statename'].values[0]
            if not country_name_i or not country_name_j:
                raise ValueError(
                    f"Country name not found for ccode {ccode_i} or {ccode_j}")

            # populate min_dist between countries
            min_dist_search = (min_dist_double_year['ccode1'] == ccode_i) & (
                    min_dist_double_year['ccode2'] == ccode_j)
            min_dist_ij = min_dist_double_year[min_dist_search & (min_dist_double_year['year'] == year)]
            # if not found raise error
            if min_dist_ij.empty:
                # fallback to next year
                min_dist_ij = min_dist_double_year[min_dist_search & (min_dist_double_year['year'] == year + 1)]
                if min_dist_ij.empty:
                    raise ValueError(
                        f"Country name not found for ccode {ccode_i} or {ccode_j} on {year}")

            # Check if rivalry exists between the two countries
            were_riv, riv_data = get_rivalry_data(ccode_i, ccode_j, year, rivalries_until_year)

            at_least_one_major = majors[(majors['ccode'] == ccode_i) | (majors['ccode'] == ccode_j)].shape[0] > 0
            dyad_is_relevant = False
            # mindist might be adjusted to 1000 to include more dyads
            if min_dist_ij['mindist'].values[0] <= 100 or at_least_one_major or were_riv:
                dyad_is_relevant = True

            if not dyad_is_relevant:
                continue

            # Drop excluded features
            data_i = data_i.drop(columns=features_exclude)
            data_j = data_j.drop(columns=features_exclude)

            # Initialize dyad dictionary with month, country IDs, and country names
            dyad = {
                'month_id': month,
                'date': date,
                'country_id_a': country_i,
                'country_id_b': country_j,
                'ccode_a': ccode_i,
                'ccode_b': ccode_j,
                'a_country_name': country_name_i,
                'b_country_name': country_name_j,
                'min_dist': min_dist_ij['mindist'].values[0],
                'a_is_major': majors[majors['ccode'] == ccode_i].shape[0] > 0,
                'b_is_major': majors[majors['ccode'] == ccode_j].shape[0] > 0,
                'a_ged_sb': data_i['ged_sb'].values[0],
                'b_ged_sb': data_j['ged_sb'].values[0],
            }
            # add rivalry data to dyad
            dyad.update(riv_data.to_dict(orient='records')[0])

            # exclude ged_sb from columns
            exclude_columns = ['year', 'gw_statename', 'gleditsch_ward', 'ccode', 'ged_sb']

            # Add data for country_i and country_j to dyad, prefix with 'a_' and 'b_'
            for feature in data_i.columns:
                if feature in exclude_columns:
                    continue
                dyad[f'a_{feature}'] = data_i[feature].values[0]  # Assuming one row per country per month
            for feature in data_j.columns:
                if feature in exclude_columns:
                    continue
                dyad[f'b_{feature}'] = data_j[feature].values[0]

            # Calculate and add ratios for specified features
            for feature in features_for_ratios:
                if feature in data_i.columns and feature in data_j.columns:
                    value_i = data_i[feature].values[0]  # Assuming one row per country per month
                    value_j = data_j[feature].values[0]
                    # Handle division by zero
                    if value_j == 0:
                        value_j = 1
                    ratio = value_i / value_j
                    dyad[f'ratio_{feature}'] = ratio

            # Append the completed dyad to the list
            dyadic_data.append(dyad)
        #     break
        # break

    return sort_dyadic_columns(pd.DataFrame(dyadic_data))


if __name__ == '__main__':
    total_months = len(cm_features['month_id'].unique())
    print(f"Processing {total_months} months...")

    # TEST CODE
    # monthly_dyadic_df: pd.DataFrame = process_month_data(130, cm_features, features_exclude, features_for_ratios,
    #                                                      majors)
    # pass
    # PRODUCTION CODE
    # Create the directory if it does not exist
    output_dir = 'data_dyad_monthly v0.6'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_month = {
            executor.submit(process_month_data, month, cm_features, features_exclude, features_for_ratios,
                            majors): month
            for month in cm_features['month_id'].unique()
        }

        for future in as_completed(future_to_month):
            month = future_to_month[future]
            try:
                monthly_dyadic_df: pd.DataFrame = future.result()
                # Save to CSV
                output_file_path = os.path.join(output_dir, f"month_id_{month}.csv")
                monthly_dyadic_df.to_csv(output_file_path, index=False)
                print(f"Month {month} processed and saved.")
            except Exception as exc:
                print(f"Month {month} generated an exception: {exc}")
                raise exc

    print("All months processed.")
