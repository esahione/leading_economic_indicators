# IHS Markit Leading Economic Indicators
# Developed by Eduardo Sahione
import pandas as pd
import itertools


def resample_dataseries(series, original_frequency, column_name):
    return series.asfreq(original_frequency, method='ffill').dropna(axis=0, how='all').resample('M').last()[column_name]


def simple_parse_series(series_name, log=False):
    series = RAW_DATA[series_name]
    if log:
        series = np.log(series)
    
    series_frequency = DATASET_INFO[series_name]['resample_information']['frequency']
    series_column = DATASET_INFO[series_name]['resample_information']['column_name']
    series = resample_dataseries(series, series_frequency, series_column)

    return (series, DATASET_INFO[series_name])

def parse_series(series_name, log=False):
    series = RAW_DATA[series_name]
    if log:
        series = np.log(series)
    try:
        series_frequency = DATASET_INFO[series_name]['resample_information']['frequency']
        series_column = DATASET_INFO[series_name]['resample_information']['column_name']
        series = resample_dataseries(series, series_frequency, series_column)
    except:
        raise NotImplemented
    series_sa_m = pa.SeasonalAdjustment(series)
    series_sa = series_sa_m.fit()
    return series_sa['trend'], series_sa, series
