import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

cm_features = pd.read_csv('data/cm_features_with_name.csv')
# sort by month_id and country_id
cm_features = cm_features.sort_values(by=['month_id', 'country_id'])

# Features to exclude from the dyadic DataFrame
features_exclude = ['date', 'country_id', 'month_id', 'country']  # Excluding 'country_name' to handle separately

# Features to create ratios for
features_for_ratios = ['wdi_sp_pop_totl', 'ged_sb', 'wdi_sp_dyn_imrt_in', 'vdem_v2x_ex_military', 'wdi_ms_mil_xpnd_zs']


def process_month_data(month, cm_features, features_exclude, features_for_ratios) -> pd.DataFrame:
    print("Processing month: ", month)
    dyadic_data = []

    cm_features_month = cm_features[cm_features['month_id'] == month]
    countries = cm_features_month['country_id'].unique()
    date = cm_features_month['date'].values[0]

    for i in range(len(countries)):
        print(f"Processing month: {month}, country {i + 1}/{len(countries)}")

        for j in range(i + 1, len(countries)):
            country_i = countries[i]
            country_j = countries[j]

            # Filter data for each country and drop excluded features
            data_i = cm_features_month[cm_features_month['country_id'] == country_i]
            data_j = cm_features_month[cm_features_month['country_id'] == country_j]

            # Get country names for both countries
            country_name_i = data_i['country'].values[0]
            country_name_j = data_j['country'].values[0]

            # Drop excluded features
            data_i = data_i.drop(columns=features_exclude)
            data_j = data_j.drop(columns=features_exclude)

            # Initialize dyad dictionary with month, country IDs, and country names
            dyad = {
                'month_id': month,
                'date': date,
                'country_id_a': country_i,
                'country_id_b': country_j,
                'a_country_name': country_name_i,
                'b_country_name': country_name_j
            }

            # Add data for country_i and country_j to dyad, prefix with 'a_' and 'b_'
            for feature in data_i.columns:
                dyad[f'a_{feature}'] = data_i[feature].values[0]  # Assuming one row per country per month
            for feature in data_j.columns:
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
    return pd.DataFrame(dyadic_data)


if __name__ == '__main__':
    total_months = len(cm_features['month_id'].unique())
    print(f"Processing {total_months} months...")

    # Create the directory if it does not exist
    output_dir = 'data_dyad_monthly'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_month = {
            executor.submit(process_month_data, month, cm_features, features_exclude, features_for_ratios): month
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

    print("All months processed.")
