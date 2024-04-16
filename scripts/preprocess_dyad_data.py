import pandas as pd
import os

dyad_df = pd.read_parquet('../data_dyad_monthly/dyad_df.parquet')
# get only dyads from Jan 2016 to Jan 2019
# to date
dyad_df['date'] = pd.to_datetime(dyad_df['date'])
# dyad_df = dyad_df[
#     (dyad_df['date'] >= pd.Timestamp(year=2012, month=1, day=1))]  # cut everything before 2015

# dyad_df = dyad_df[
#     (dyad_df['date'] >= pd.Timestamp(year=2014, month=7, day=1)) &
#     (dyad_df['date'] <= pd.Timestamp(year=2019, month=1, day=1))]
# dyad_df.info()
# print("Shifting GED-SB variables by 15 months...")
#
# dyad_df.sort_values(by=['country_id_a', 'country_id_b', 'month_id'], inplace=True)
#
#
# # Function to shift the 'ged_sb' variable backwards by 15 months within each group
# def shift_ged_sb(group):
#     # Assuming 'ged_sb' is recorded per country in the dyad, adjust if you have a single 'ged_sb' variable
#     group['a_ged_sb_15_shifted'] = group['a_ged_sb'].shift(-15)  # 3 months gap + 12 months prediction
#     group['b_ged_sb_15_shifted'] = group['b_ged_sb'].shift(-15)  # 3 months gap + 12 months prediction
#     return group


# Apply the function to each dyad group
# dyad_df = dyad_df.groupby(['country_id_a', 'country_id_b']).apply(shift_ged_sb)
# dyad_df.reset_index(drop=True, inplace=True)
# cm_features['y_shifted'] = cm_features.groupby('country_id')['ged_sb'].shift(-15)  # 3 months gap + 12 months prediction
# show na for y_shifted
# cm_features[cm_features['y_shifted'].isna()]
# drop na
# dyad_df = dyad_df.dropna()
# dyad_df

print("One-hot encoding country identifiers...")
country_a_and_b_ids = dyad_df[['country_id_a', 'country_id_b']]
dyad_df = pd.get_dummies(dyad_df, columns=['country_id_a', 'country_id_b'], drop_first=False, dtype=int)
# merge back country_id_a and country_id_b
dyad_df = pd.concat([dyad_df, country_a_and_b_ids], axis=1)
# dyad_df

print("Cutting DF")
prediction_year = 2018  # Jan 2018 to Jan 2019
cut_year = prediction_year - 2  # 2016

features_to_oct = pd.Timestamp(year=cut_year, month=10, day=1)  # 2016-Oct-01

# Splitting the dataset
train_df = dyad_df[dyad_df['date'] < features_to_oct]
# test_df is one year from Oct 2016 to Oct 2017
test_df = dyad_df[
    (dyad_df['date'] >= pd.Timestamp(year=prediction_year - 2, month=10, day=1)) &  # oct 2016 predicts Jan 2018
    (dyad_df['date'] < pd.Timestamp(year=prediction_year - 1, month=10, day=1))]  # oct 2017 predicts Jan 2019

print("Scaling the numeric features...")
# Explicitly list columns to be dropped before scaling
columns_not_numeric = ['month_id', 'date', 'country_id_a', 'country_id_b', 'a_ged_sb', 'b_ged_sb', 'a_country_name',
                       'b_country_name', 'a_gleditsch_ward', 'b_gleditsch_ward', ]

# Also, drop one-hot encoded country identifiers if they are already in the dataframe
columns_not_numeric.extend(dyad_df.filter(regex='^country_id_a_').columns.tolist())
columns_not_numeric.extend(dyad_df.filter(regex='^country_id_b_').columns.tolist())

# Determine numeric columns by excluding the ones to drop from the dataframe
numeric_columns = dyad_df.drop(columns=columns_not_numeric).columns.tolist()
del dyad_df

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train_df[numeric_columns])

# Scale the numeric features
train_df_scaled = scaler.transform(train_df[numeric_columns])
test_df_scaled = scaler.transform(test_df[numeric_columns])

# Convert scaled features back to DataFrames, maintaining the original index for later recombination
train_df_scaled = pd.DataFrame(train_df_scaled, columns=numeric_columns, index=train_df.index)
test_df_scaled = pd.DataFrame(test_df_scaled, columns=numeric_columns, index=test_df.index)

print("Concatenating the dataframes...")
# Re-add the dropped columns to the scaled dataframe
train_df_final = pd.concat([train_df[columns_not_numeric], train_df_scaled], axis=1)
test_df_final = pd.concat([test_df[columns_not_numeric], test_df_scaled], axis=1)

print("Saving data to Parquet format...")
# save dir
output_dir = '../data_dyad_monthly_nn'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
train_df_final.to_parquet(f'{output_dir}/train_df_{prediction_year}.parquet')
test_df_final.to_parquet(f'{output_dir}/test_df_{prediction_year}.parquet')
