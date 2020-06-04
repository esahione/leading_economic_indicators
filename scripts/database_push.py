import pandas as pd
from pathlib import Path
from ihs_lei.data.push import *

def load_datasets(series_data_loc, series_metadata_loc):
    series_data = pd.read_csv(series_data_loc, names=['quandl_code', 'date', 'value'])
    series_metadata = pd.read_excel(series_metadata_loc, sheetname='all-data')
    list_indicators = pd.read_excel(series_metadata_loc, sheetname='list-indicators')
    sources = pd.read_excel(series_metadata_loc, sheetname='list-sources')
    series_metadata.columns = [x.lower().replace(' ', '_') for x in series_metadata.columns]
    series_metadata = series_metadata.reset_index(drop=True)
    list_indicators.columns = [x.lower().replace(' ', '_') for x in list_indicators.columns]
    list_indicators = list_indicators.reset_index(drop=True)
    sources.columns = [x.lower().replace(' ', '_') for x in sources.columns]
    sources = sources.reset_index(drop=True)
    result = {
        'economics': series_data,
        'series_metadata': series_metadata,
        'indicator_metadata': list_indicators,
        'sources': sources
    }

    return result


if __name__ == '__main__':
    print('[INFO] Loading datasets from csv and excel files.')
    series_data_loc = Path('.', 'assets', 'SGE', 'SGE_20160518.csv')
    series_metadata_loc = Path('.', 'assets', 'SGE', 'sge_metadata.xlsx')
    result = load_datasets(series_data_loc, series_metadata_loc)
    print('[INFO] Pushing dataset to database.')
    # print(result['indicator_metadata'])
    # for series, df in result.items():
    #     print(series)
    #     print(df)
    push_datasets_to_amazon_redshift(result, schema='ihs_lei')