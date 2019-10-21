import numpy as np
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from datetime import timedelta, datetime, date, time

# Start a Dask Client for cleaning.
# This can be monitored.
client = Client()

@dataclass
class dataset:
    feature_names: np.ndarray
    target_nanes: list
    target: np.ndarray
    data: np.ndarray

@dataclass
class splitdata:
    train: pd.DataFrame
    test: pd.DataFrame

class dataFormatter:
    def __init__(self, path):
        # Initialize Object
        self.df = dd.read_parquet(path)
        self.arrays = {}

    def write_month(self, year):
        def last_day_of_month(any_day):
            # From: https://stackoverflow.com/questions/42950/get-last-day-of-the-month-in-python
            next_month = any_day.replace(day=28) + timedelta(days=4)  # this will never fail
            return next_month - timedelta(days=next_month.day)

        for month in range(1,13):
            start = datetime.strptime(f'{year}-{month}-01', '%Y-%m-%d')
            end = datetime.combine(last_day_of_month(date(year,month, 1)), time(23, 59, 59))
            self.arrays[year].loc[start:end,'month'] = int(month)

    def assign_labels(self, year, col):
        conditions  = [self.arrays[year][col] >= (.41*3), (self.arrays[year][col] < (.41*3)) & (self.arrays[year][col] > .82) ,self.arrays[year][col] <= .82]
        scales = [2,1,0]
        self.arrays[year]['labels'] = np.select(conditions, scales)

    def get_year_frame(self, year):
        # Spilts the dask dataframe, sorted.
        # and spreads them across the dask
        def slice_dataframe(df, year):
            start = pd.to_datetime(f'{year}-01-01').tz_localize('US/Central')
            end = pd.to_datetime(f'{year}-12-31').tz_localize('US/Central')
            return df[start:end].compute()
        # cluster that is running
        self.arrays[year] = client.submit(slice_dataframe, self.df, year)

    def create_flag(self, year):
        self.arrays[year]['indicator'] = self.arrays[year]['trip_total'] / self.arrays[year]['trip_minutes']

    def clean(self):
        self.df = self.df[['trip_total', 'trip_seconds']].dropna()
        self.df['trip_minutes'] = self.df['trip_seconds'] / 60
        self.df = self.df.drop('trip_seconds', axis=1)
        self.df = self.df[self.df.trip_total >= 1]
        self.df = self.df[self.df.trip_minutes >= 1]

    def dump_dataset(self, year):
        print("Dumping Dataset")
        feature_names = self.arrays[year].columns
        target_names = ['high', 'average', 'low']
        target = self.arrays[year]['labels'].to_numpy()
        data = self.arrays[year].to_numpy()
        return dataset(feature_names, target_names, target, data)

    def gen_train_test(self, year):
        print("Generating train and test...")
        taxi_yr = self.dump_dataset(year)
        taxi_yr.feature_names = taxi_yr.feature_names[:-1]
        sampled_df = taxi_yr.data[np.random.choice(taxi_yr.data.shape[0], 1000, replace=False), :]
        train, test = train_test_split(sampled_df, test_size=0.25)
        return splitdata(train=train, test=test)
