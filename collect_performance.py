from pathlib import Path
import os
import pandas as pd
import argparse

from utilities import list_submissions, get_predictions, views_month_id_to_year, views_month_id_to_month, TargetType, get_submission_details


def get_eval(submission: str|os.PathLike, target: TargetType, groupby: str|list[str] = None) -> pd.DataFrame:
    """Convenience function to read and aggregate evaluation metrics from a submission.

    Parameters
    ----------
    submission : str | os.PathLike
        Path to a folder structured like a submission_template
    target : TargetType
        A string, either "pgm" for PRIO-GRID-months, or "cm" for country-months.
    groupby : str | list[str], optional
        A dimension to aggregate results across. Some options (all except None and "pooled" can be combined in a list):
        None: no aggregation
        "pooled": complete aggregation
        "year": aggregate by calendar year
        "month": aggregate by calendar month
        "month_id": aggregate by month_id (1 is January 1980)
        "country_id": aggregate by country (currently only works for target == "cm")
        "priogrid_gid": aggregate by PRIO-GRID id.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        Target must be "cm" or "pgm".
    """

    submission = Path(submission)
    df = get_predictions(submission / "eval", target = target)

    if target == "cm":
        unit = "country_id"
    elif target == "pgm":
        unit = "priogrid_gid"
    else:
        raise ValueError(f'Target must be either "cm" or "pgm".')
    
    df = df.pivot_table(values=["value"], index = ["month_id", unit], columns = "metric")
    df.columns = df.columns.get_level_values(1).to_list()

    if df.index.names != [None]:
        df = df.reset_index()

    sdetails = get_submission_details(submission)
    df["team"] = sdetails["team"]
    df["model"] = sdetails["even_shorter_identifier"]

    if groupby == None:    
        return df
    if groupby == "pooled":
        groupby = []
    if isinstance(groupby, str):
        groupby = [groupby]
    groupby.extend(["team", "model"])

    if "year" in groupby:
        df["year"] = views_month_id_to_year(df["month_id"])

    if "month" in groupby:
        df["month"] = views_month_id_to_month(df["month_id"])

    
    if "month_id" not in groupby:
        df = df.drop(columns="month_id")
    if unit not in groupby:
        df = df.drop(columns=unit)
    
    df = df.set_index(groupby)
    df = df.groupby(level=groupby).mean().reset_index()  

    return df

def evaluation_table(submissions: str|os.PathLike, target: TargetType, groupby: str, save_to: str|os.PathLike = None, across_submissions: bool = False):
    """Convenience function to make aggregated result tables of the evaulation metrics and store them to LaTeX, HTML, and excel format.

    Parameters
    ----------
    submissions : str | os.PathLike
        Path to a folder only containing folders structured like a submission_template
    target : TargetType
        A string, either "pgm" for PRIO-GRID-months, or "cm" for country-months.
    groupby : str | list[str], optional
        A dimension to aggregate results across. Some options (all except "pooled" can be combined in a list):
        "pooled": complete aggregation
        "year": aggregate by calendar year
        "month": aggregate by calendar month
        "month_id": aggregate by month_id (1 is January 1980)
        "country_id": aggregate by country (currently only works for target == "cm")
        "priogrid_gid": aggregate by PRIO-GRID id.
    save_to : str | os.PathLike, optional
        Folder to store evaulation tables in LaTeX, HTML, and excel format.
    across_submissions : bool
        Aggregate across submissions

        
    Returns
    -------
    pandas.DataFrame
        If save_to is None, or if groupby is a list or None, the function returns the dataframe. 
        It can be useful to collate all evaluation data into one dataframe, but not to write everything out to a table.

    """

    submissions = list_submissions(Path(submissions))

    eval_data = [get_eval(submission, target, groupby) for submission in submissions]
    df = pd.concat(eval_data)

    if groupby == None:    
        # Edge case if user specify a list of dimensions to groupby or no aggregation. Probably not something to plot tables for.
        return df
    if groupby == "pooled":
        groupby = []
    if isinstance(groupby, str):
        groupby = [groupby]
    
    if not across_submissions:
        groupby.extend(["team", "model"])
    
    if "year" in groupby:
        min_year = df["year"].min()
        df = df.pivot_table(values=["crps", "ign", "mis"], index = [g for g in groupby if g != "year"], aggfunc = {"crps": "mean", "ign": "mean", "mis": "mean"}, columns = "year")
        df = df.sort_values(("crps", min_year))
    else:
        if isinstance(df, pd.Series):
            # No need to sort single events.
            pass
        else:
            df = df.sort_values("crps")
            
    if save_to == None:
        return df 
    else:
        file_stem = f"metrics_{target}_by={groupby}"

        css_alt_rows = 'background-color: #e6e6e6; color: black;'
        highlight_props = "background-color: #00718f; color: #fafafa;"
        df = df.style \
                .format(decimal='.', thousands=' ', precision=3) \
                .highlight_min(axis=0, props=highlight_props) \
                .set_table_styles([
                    {'selector': 'tr:nth-child(even)', 'props': css_alt_rows}
                ])
        df.to_latex(save_to / f'{file_stem}.tex')
        df.to_html(save_to / f'{file_stem}.html')
        df.to_excel(save_to / f'{file_stem}.xlsx')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Method for collating evaluations from all submissions in the ViEWS Prediction Challenge",
                                     epilog = "Example usage: python collect_performance.py -s ./submissions")
    parser.add_argument('-s', metavar='submissions', type=str, help='path to folder with submissions complying with submission_template')
    parser.add_argument('-o', metavar='output_folder', type=str, help='path to folder to save result tables', default=None)
    parser.add_argument('-tt', metavar='target_type', type=str, help='target "pgm" or "cm"', default=None)
    parser.add_argument('-g', metavar='groupby', nargs = "+", type=str, help='string or list of strings of dimensions to aggregate over', default=None)
    
    args = parser.parse_args()

    evaluation_table(submissions = args.s, 
                     target = args.tt, 
                     groupby = args.g, 
                     save_to = args.o)