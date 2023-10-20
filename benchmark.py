# python3 your_script.py --feature_folder /path/to/features --target cm --year 2018 --benchmark_name boot

import argparse
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import pyarrow.compute as pac

def main():
    parser = argparse.ArgumentParser(description="Global Bootstrap Benchmark Script")
    parser.add_argument("--feature_folder", type=str, help="Path to the feature folder")
    parser.add_argument("--target", type=str, choices=["pgm", "cm"], help="Target type (pgm or cm)")
    parser.add_argument("--year", type=int, help="Year")
    #new argument
    parser.add_argument("--benchmark_name",type=str,help="boot - Bootstrap, hist - last historical")
    parser.add_argument("--month_lag",type=int,help="specify the month lag for prediction" )

    args = parser.parse_args()

    feature_folder = Path(args.feature_folder)
    target = args.target
    year = args.year
    benchmark_name = args.benchmark_name
    month_lag = args.month_lag
    result = global_bootstrap_benchmark(benchmark_name, feature_folder, target, year,month_lag)
    result.to_parquet(f'{feature_folder}/{benchmark_name}_{target}_{year}.parquet')

    # Do something with the result, e.g., print or save it.
    print(result)

def global_bootstrap_benchmark(benchmark_name,feature_folder, target, year,month_lag) -> pd.DataFrame:
    """
    Draws a random sample with replacement from the complete global pool of fatalities data from the year before.
    """
    
    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError('Target must be "pgm" or "cm".')
    
    if benchmark_name == "boot":
        
        filter = pac.field("year") == year
        df = pq.ParquetDataset(feature_folder / target, filters=filter).read(columns=[unit, "month_id"]).to_pandas()
        
        filter = (pac.field("month_id") <= df.month_id.min() - month_lag) & (pac.field("month_id") > df.month_id.min() - (12+month_lag))
        pool = pq.ParquetDataset(feature_folder / target, filters=filter).read(columns=["ged_sb"]).to_pandas()
        
        df["outcome"] = np.random.choice(pool["ged_sb"], size=(df.shape[0], 1000), replace=True).tolist()
        df = df.explode('outcome').astype('int32')
        df['draw'] = df.groupby(['month_id', unit]).cumcount()
        df.set_index(['month_id',unit,'draw'],inplace=True)        
        return df
    
    elif benchmark_name=="hist":
        filter = pac.field("year") == year
        df = pq.ParquetDataset(feature_folder / target, filters=filter).read(columns=[unit, "month_id"]).to_pandas()
        
        filter = (pac.field("month_id") <= df.month_id.min() - month_lag) & (pac.field("month_id") > df.month_id.min() - (12+month_lag))
        pool = pq.ParquetDataset(feature_folder / target, filters=filter).read(columns=["ged_sb"]).to_pandas()
        df["outcome"] = [np.random.poisson(value, 1000) for value in pool["ged_sb"]]
        df = df.explode('outcome').astype('int32')
        df['draw'] = df.groupby(['month_id', unit]).cumcount()
        df.set_index(['month_id',unit,'draw'],inplace=True)
        return df
    else:
        raise ValueError('Benchmark must be "boot" or "hist"')
    

if __name__ == "__main__":
    main()



# python3 your_script.py --feature_folder /path/to/features --target cm --year 2018 --benchmark_name boot
