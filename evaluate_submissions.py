from pathlib import Path
import yaml
from CompetitionEvaluation import load_data, structure_data, calculate_metrics
from utilities import list_submissions, get_predictions, TargetType
from dataclasses import dataclass
import os
import xarray
import numpy as np
import numpy.typing as npt
from scipy.signal import resample
import argparse
import pandas as pd
import pyarrow

import logging
logging.getLogger(__name__)
logging.basicConfig(filename='evaluate_submission.log', encoding='utf-8', level=logging.INFO)

def evaluate_forecast(forecast: pd.DataFrame, 
                      actuals: pd.DataFrame, 
                      target: TargetType,
                      expected_samples: int,
                      save_to: str|os.PathLike,
                      draw_column_name: str ="draw", 
                      data_column_name: str = "outcome",
                      bins: list[float] = [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5]) -> None:

    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError(f'Target {target} must be either "pgm" or "cm".')

    # Cast to xarray
    observed, predictions = structure_data(actuals, forecast, draw_column_name=draw_column_name, data_column_name = data_column_name)

    if bool((predictions["outcome"] > 10e9).any()):
        logging.warning(f'Found predictions larger than earth population. These are censored at 10 billion.')
        predictions["outcome"] = xarray.where(predictions["outcome"]>10e9, 10e9, predictions["outcome"])
        
    crps = calculate_metrics(observed, predictions, metric = "crps", aggregate_over="nothing")
    mis = calculate_metrics(observed, predictions, metric = "mis", prediction_interval_level = 0.9, aggregate_over="nothing")

    if predictions.dims['member'] != expected_samples:
        logging.warning(f'Number of samples ({predictions.dims["member"]}) is not 1000. Using scipy.signal.resample to get {expected_samples} samples when calculating Ignorance Score.')
        np.random.seed(284975)
        arr: npt.ArrayLike = resample(predictions.to_array(), expected_samples, axis = 3)
        arr = np.where(arr<0, 0, arr) # For the time when resampling happens to go below zero.

        new_container = predictions.sel(member = 1)
        new_container = new_container.expand_dims({"member": range(0,expected_samples)}).to_array().transpose("variable", "month_id", unit, "member")
        predictions: xarray.Dataset = xarray.DataArray(data = arr, coords = new_container.coords).to_dataset(dim="variable")

    if bool((predictions["outcome"] < 0).any()):
        logging.warning(f'Found negative predictions. These are censored at 0 before calculating Ignorance Score.')
        predictions["outcome"] = xarray.where(predictions["outcome"]<0, 0, predictions["outcome"])


    ign = calculate_metrics(observed, predictions, metric = "ign", bins = bins, aggregate_over="nothing")
    
    # Long-format out
    crps.rename(columns={"ign": "value"}, inplace=True)
    ign.rename(columns={"ign": "value"}, inplace=True)
    mis.rename(columns={"ign": "value"}, inplace=True)
    
    (save_to / "metric=crps").mkdir(exist_ok=True, parents = True)
    (save_to / "metric=ign").mkdir(exist_ok=True, parents = True)
    (save_to / "metric=mis").mkdir(exist_ok=True, parents = True)
    crps.to_parquet(save_to / "metric=crps" / "crps.parquet")
    ign.to_parquet(save_to / "metric=ign" / "ign.parquet")
    mis.to_parquet(save_to / "metric=mis" / "mis.parquet")

def match_forecast_with_actuals(submission, actuals_folder, target: TargetType, year):

    filter = pyarrow.compute.field("year") == year
    actuals = get_predictions(actuals_folder, target = target, filters = filter)
    predictions = get_predictions(submission, target = target, filters = filter)

    predictions.drop(columns=["year"], inplace = True)
    actuals.drop(columns=["year"], inplace = True)

    return actuals, predictions

def main():
    parser = argparse.ArgumentParser(description="Method for evaluation of submissions to the ViEWS Prediction Challenge",
                                     epilog = "Example usage: python evaluate_submissions.py -s ./submissions -a ./actuals -e 100")
    parser.add_argument('-s', metavar='submissions', type=str, help='path to folder with submissions complying with submission_template')
    parser.add_argument('-a', metavar='actuals', type=str, help='path to folder with actuals')
    parser.add_argument('-t', metavar='targets', nargs = "+", type=str, help='pgm or cm or both', default = ["pgm", "cm"])
    parser.add_argument('-y', metavar='years', nargs = "+", type=int, help='yearly windows to evaluate', default = [2018, 2019, 2020, 2021])
    parser.add_argument('-e', metavar='expected', type=int, help='expected samples', default = 1000)
    parser.add_argument('-sc', metavar='sample-column-name', type=str, help='(Optional) name of column for the unique samples', default = "draw")
    parser.add_argument('-dc', metavar='data-column-name', type=str, help='(Optional) name of column with data, must be same in both observed and predictions data', default = "outcome")
    parser.add_argument('-ib', metavar = 'ign-bins', nargs = "+", type = float, help='Set a binning scheme for the ignorance score. List or integer (nbins). E.g., "--ib 0 0.5 1 5 10 100 1000". None also allowed.', default = [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5])    

    args = parser.parse_args()


    submission_path = Path(args.s)
    actuals_folder = Path(args.a)
    expected_samples = args.e
    targets = args.t
    years = args.y
    sample_column_name = args.sc
    data_column_name = args.dc
    bins = args.ib

    submissions = list_submissions(submission_path)

    for submission in submissions:
        try:
            logging.info(f'Evaluating {submission.name}')
            for target in targets:
                for year in years:
                    if any((submission / target).glob("**/*.parquet")): # test if there are prediction files in the target
                        actuals, predictions = match_forecast_with_actuals(submission, actuals_folder, target, year)
                        save_to = submission / "eval" / f'target={target}' / f'year={year}'
                        evaluate_forecast(forecast = predictions, 
                                        actuals = actuals, 
                                        target = target,
                                        expected_samples = expected_samples,
                                        draw_column_name=sample_column_name,
                                        data_column_name=data_column_name,
                                        bins = bins,
                                        save_to = save_to)
        except Exception as e:
            logging.error(f'{str(e)}')


if __name__ == "__main__":
    main()