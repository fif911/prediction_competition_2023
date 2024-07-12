"""Implementation of single horizon NGBoost model for one year"""
# Configuration. TODO get from config file
SAVE_PREDICTIONS = True
INCLUDE_COUNTRY_ID = False
INCLUDE_MONTH_ID = False
DROP_0_ROWS_PERCENT = 20
DROP_35_LEAST_IMPORTANT = False

import pandas as pd
import numpy as np
from utilities import views_month_id_to_date

prediction_year = 2022

cm_features_version = '1.0'

if __name__ == '__main__':

    cm_features = pd.read_csv(f'data/cm_features_v{cm_features_version}_Y{prediction_year}.csv')

    # Load benchmark model
    model_names = {
        "bootstrap": "bm_cm_bootstrap_expanded_",
        "poisson": "bm_cm_last_historical_poisson_expanded_",
    }
    benchmark_model = pd.read_parquet(f'Benchmarks/{model_names["poisson"]}{prediction_year}.parquet')
    # Group by 'month_id' and 'country_id' and calculate mean and std for each group
    agg_funcs = {
        'outcome': ['mean', 'std']  # Assuming 'prediction' is the column to aggregate; adjust if necessary
    }
    # there is 20 draws per each country per each month. Get the mean of the draws and std for each month
    benchmark_model = benchmark_model.groupby(['month_id', 'country_id']).agg(agg_funcs).reset_index()
    # Flatten the multi-level columns resulting from aggregation
    benchmark_model.columns = ['_'.join(col).strip() if col[1] else col[0] for col in benchmark_model.columns.values]
    # Rename columns
    benchmark_model.rename(columns={'outcome_mean': 'outcome', 'outcome_std': 'outcome_std'}, inplace=True)

    # add date column
    benchmark_model['date'] = views_month_id_to_date(benchmark_model['month_id'])

    # load actuals
    actuals_model = pd.read_parquet(f'actuals/cm/window=Y{prediction_year}/cm_actuals_{prediction_year}.parquet')
    # actuals_model = actuals_model.groupby(['month_id', 'country_id']).mean().reset_index()
    actuals_model['date'] = views_month_id_to_date(actuals_model['month_id'])

    # get all rows that have ged_sb_y_ in the name
    all_targets = [col for col in cm_features.columns if col.startswith('ged_sb_y_')]
    prediction_window = 14
    target = f'ged_sb_{prediction_window}'
    try:
        all_targets.remove(target)
    except ValueError:
        pass
    cm_features.drop(columns=all_targets, inplace=True)
    cm_features = cm_features.dropna()  # drop all rows for which ged_sb_y_15 is NAN

    cm_features = cm_features.drop(columns=['country', 'gleditsch_ward'], errors='ignore')
    # drop if exists 'year', 'ccode'
    cm_features = cm_features.drop(columns=['year', 'ccode', 'region', 'region23'], errors='ignore')

    LEAST_IMPORTANT_FEATURES = ['general_efficiency_t48',
                                '_wdi_sp_dyn_imrt_in',
                                'vdem_v2x_egal',
                                'vdem_v2x_partipdem',
                                'vdem_v2x_partip',
                                'vdem_v2x_libdem',
                                'dam_cap_pcap_t48',
                                'vdem_v2xdd_dd',
                                'vdem_v2x_edcomp_thick',
                                'groundwater_export_t48',
                                'wdi_sh_sta_stnt_zs',
                                'region_Middle East & North Africa',
                                'vdem_v2x_execorr',
                                'region23_Western Asia',
                                'region23_Southern Europe',
                                'region23_Northern Africa',
                                'region_Sub-Saharan Africa',
                                'region23_Caribbean',
                                'region23_Eastern Europe',
                                'region23_Eastern Africa',
                                'region23_South-Eastern Asia',
                                'region23_Middle Africa',
                                'region23_Northern Europe',
                                'region23_Western Africa',
                                'region23_Southern Africa',
                                'region23_South America',
                                'region_Latin America & Caribbean',
                                'region23_Northern America',
                                'region_North America',
                                'region23_Melanesia',
                                'region23_Eastern Asia',
                                'region23_Central Asia',
                                'region23_Central America',
                                'region_Europe & Central Asia',
                                'region23_Western Europe']
    if DROP_35_LEAST_IMPORTANT:
        print("Current number of features:", len(cm_features.columns))
        cm_features = cm_features.drop(columns=LEAST_IMPORTANT_FEATURES)
        print("Number of features after dropping 35 least important:", len(cm_features.columns))

    cm_features['date'] = pd.to_datetime(cm_features['date'])
    cm_features['country_id'] = cm_features['country_id'].astype('category')

    # One-hot encode 'country_id'
    if INCLUDE_COUNTRY_ID:
        # TODO: try what changes if encode ccode
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit_transform(cm_features[['country_id']])
        countries_encoded = encoder.transform(cm_features[['country_id']])
        # rename the columns
        countries_encoded = pd.DataFrame(countries_encoded, columns=encoder.get_feature_names_out(['country_id']))
        countries_encoded = countries_encoded.drop(columns='country_id_1')  # drop country_id_1
        # drop na

        # countries_encoded
        # merge the encoded features with the original dataset
        cm_features = pd.concat([cm_features, countries_encoded], axis=1)
        cm_features = cm_features.dropna()

    last_month_id = cm_features['month_id'].max()
    train_features_to_oct = last_month_id - 11 - 3
    test_features_since_nov = last_month_id - 11
    print("features_to_oct:", train_features_to_oct)
    print("features_since_nov:", test_features_since_nov)

    train_df = cm_features[cm_features['month_id'] <= train_features_to_oct]  # train is till 476 inclusive
    # test_df is one year from Nov to Oct inclusive (479-490)
    test_df = cm_features[
        (cm_features['month_id'] >= test_features_since_nov)]

    if DROP_0_ROWS_PERCENT > 0:
        print(f"Initial count: {train_df[train_df[target] == 0].shape[0]}")
        indices = train_df[train_df[target] == 0].index.to_series()
        num_to_drop = int(len(indices) * DROP_0_ROWS_PERCENT / 100)
        indices_to_drop = indices.sample(n=num_to_drop, random_state=42)
        train_df = train_df.drop(indices_to_drop)
        print(f"Count after removal: {train_df[train_df[target] == 0].shape[0]}")
    test_df.reset_index(drop=True, inplace=True)

    # save date column for test_df
    test_df_date = test_df['date']
    train_df_date = train_df['date']
    test_df_country_name = test_df['gw_statename']
    train_df_country_name = train_df['gw_statename']
    train_df_country_id = train_df['country_id']
    test_df_country_id = test_df['country_id']
    train_df_month_id = train_df['month_id']
    test_df_month_id = test_df['month_id']

    test_df = test_df.drop('date', axis=1)
    test_df = test_df.drop("country_id", axis=1)
    test_df = test_df.drop("gw_statename", axis=1)

    train_df = train_df.drop('date', axis=1)
    train_df = train_df.drop("country_id", axis=1)
    train_df = train_df.drop("gw_statename", axis=1)

    if not INCLUDE_MONTH_ID:
        test_df = test_df.drop('month_id', axis=1)
        train_df = train_df.drop('month_id', axis=1)

    print(test_df_month_id.unique())
    print("Difference between bechmark and test month_id:")
    print(benchmark_model['month_id'].min() - test_df_month_id.max())
    print(benchmark_model['month_id'].min() - test_df_month_id.min())

    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]

    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    from sklearn.tree import DecisionTreeRegressor
    # Model tuning:
    # https://stanfordmlgroup.github.io/ngboost/2-tuning.html#Using-sklearn-Model-Selection
    from ngboost.scores import CRPScore, LogScore
    from ngboost.distns import Poisson, Normal, MultivariateNormal, Gamma
    from ngboost import NGBRegressor

    # supress RuntimeWarning for NGBRegressor
    import warnings

    normal_enabled = False
    if normal_enabled:
        n_estimators = 300
    else:
        n_estimators = 50
    score = LogScore

    bs_max_depth = 5
    minibatch_frac = 0.5
    base_learner = DecisionTreeRegressor(
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=bs_max_depth,
        splitter="best",
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"Training NGB with {n_estimators} estimators and {score} score...")
        ngb = NGBRegressor(n_estimators=n_estimators, verbose_eval=10, Dist=Normal if normal_enabled else Poisson,
                           learning_rate=0.01,
                           Score=score,
                           random_state=42,
                           Base=base_learner,
                           minibatch_frac=minibatch_frac,
                           # col_sample=1.0,
                           early_stopping_rounds=None).fit(
            X_train, y_train,
            X_test, y_test,  # be careful with this, not to use early stopping
        )
    ngb_train_predictions = ngb.predict(X_train)
    ngb_predictions = ngb.predict(X_test)
    ngb_predictions_dist = ngb.pred_dist(X_test)

    ngb_train_predictions = [max(0, pred) for pred in ngb_train_predictions]
    ngb_predictions = [max(0, pred) for pred in ngb_predictions]
    print("Done!")

    test_df['ngb_predictions'] = ngb_predictions
    train_df['ngb_predictions'] = ngb_train_predictions

    # add date column back to test_df and add to each date shift of 14 months
    test_df['date'] = test_df_date + pd.DateOffset(months=prediction_window)
    train_df['date'] = train_df_date
    test_df['country_id'] = test_df_country_id
    train_df['country_id'] = train_df_country_id
    test_df['month_id'] = test_df_month_id
    train_df['month_id'] = train_df_month_id
    test_df['country_name'] = test_df_country_name
    train_df['country_name'] = train_df_country_name

    from sklearn.metrics import mean_squared_error
    from math import sqrt

    # TODO: Improve metrics and use all metrics from the VIEWS competition
    # Calculate RMSE
    # train_rmse = sqrt(mean_squared_error(y_train, xg_lss_pred_train))
    ngb_train_rmse = sqrt(mean_squared_error(y_train, ngb_train_predictions))

    ngb_test_rmse = sqrt(mean_squared_error(y_test, ngb_predictions))
    all_zeros_rmse = sqrt(mean_squared_error(y_test, [0] * len(y_test)))
    # actuals_rmse = sqrt(mean_squared_error(actuals_model['ged_sb'], predictions))
    # benchmark_rmse = sqrt(mean_squared_error(y_test, benchmark_model['outcome']))
    actuals_bench_rmse = sqrt(mean_squared_error(actuals_model['ged_sb'], benchmark_model['outcome']))

    print("Cm features version:", cm_features_version)
    print(f"Prediction year: {prediction_year}")
    print(f"Include country_id: {INCLUDE_COUNTRY_ID}")
    print(f"Include month_id: {INCLUDE_MONTH_ID}")
    print(f"Drop train 0 rows: {DROP_0_ROWS_PERCENT}%")
    print(f"Normal distribution: {normal_enabled}")
    print(f"Number of estimators: {n_estimators}")
    print(f"Score: {str(score)}")

    print(f"\nNGB [train predictions] RMSE NGB: {ngb_train_rmse}")
    print(f"NGB [test predictions]  RMSE NGB: {ngb_test_rmse}")
    print(f"All Zeros: {all_zeros_rmse}")
    print(f"\nBenchmark: RMSE ACTUALS VS BENCHMARK: {actuals_bench_rmse}")

    PLOT_STD = True
    if PLOT_STD:
        import numpy as np

        if normal_enabled:
            ngb_predictions_std = np.sqrt(ngb_predictions_dist.var)
        else:
            sampled_dist = ngb_predictions_dist.sample(1000)
            ngb_predictions_std = sampled_dist.std(axis=0)
            ngb_predictions_max = sampled_dist.max(axis=0)
            ngb_predictions_min = sampled_dist.min(axis=0)
            test_df['ngb_predictions_max'] = ngb_predictions_max
            test_df['ngb_predictions_min'] = ngb_predictions_min

        # add std to test_df
        test_df['ngb_predictions_std'] = ngb_predictions_std

        import matplotlib.pyplot as plt

        # Assuming test_df is your DataFrame, and 'target' and 'predictions' are columns in it
        unique_months = test_df['month_id'].unique()
        n_months = len(unique_months)
        print("Unique months:", unique_months)

        # Calculate the grid size for the subplot (simple square root approximation for a square grid)
        grid_size_x = int(n_months ** 0.5) + (1 if n_months % int(n_months ** 0.5) else 0)
        grid_size_y = grid_size_x + 1

        # print(f'Grid size: {grid_size}')
        # Set overall figure size
        plt.figure(
            figsize=(grid_size_x * 6, grid_size_y * 3))  # Adjust the size factors (6, 4) based on your preference

        for index, month_id in enumerate(unique_months, start=1):
            this_month = test_df[test_df['month_id'] == month_id]
            mean_sq_error = sqrt(mean_squared_error(this_month[target], this_month['ngb_predictions']))
            current_date = this_month["date"].iloc[0]
            target_month = this_month[target]
            predictions_month = this_month['ngb_predictions']

            # Create subplot for current month
            plt.subplot(grid_size_x, grid_size_y, index)
            plt.scatter(target_month, predictions_month, color='blue', label='Actual vs Predicted', alpha=0.5)

            if PLOT_STD:
                predictions_std_month = this_month['ngb_predictions_std']
                plt.errorbar(target_month, predictions_month, yerr=predictions_std_month, fmt='o', color='blue',
                             alpha=0.5)

            # print current_date in YY/MM format
            print_date = current_date.strftime('%Y-%m')
            plt.title(f'Date {print_date} - NGB RMSE: {mean_sq_error:.2f}')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            # plt.xscale('log')
            # plt.yscale('log')
            line_length = 2000
            # if SCALE_DATAFRAME:
            #     line_length = 1

            plt.plot([0, line_length], [0, line_length], color='red', label='45 degree line')
            plt.legend()
            plt.xticks(rotation=45)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        dist_name = 'normal' if normal_enabled else 'poisson'
        folder_to_str = f"ng_boost_cm_v{cm_features_version}_pw_{prediction_window}_{dist_name}_d_{DROP_0_ROWS_PERCENT}_n_{n_estimators}_s_{score.__name__.lower()}_c_{str(INCLUDE_COUNTRY_ID)[0]}_m_{str(INCLUDE_MONTH_ID)[0]}_bsd_{bs_max_depth}_mbf_{minibatch_frac}_dli_{35 if DROP_35_LEAST_IMPORTANT else 0}"

        plt.savefig(f'predictions_{prediction_year}_ngb.png')