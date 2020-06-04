import pandas as pd
from sortedcontainers import SortedSet
import re
from pprint import pprint
import pickle
import fuzzywuzzy
from fuzzywuzzy import process
from itertools import combinations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor as Pool
import asyncio
from functools import partial
import os
import numpy as np
import datetime
now = datetime.datetime.now()
import gzip
from io import StringIO, BytesIO
from functools import wraps
import boto3 as boto
from sqlalchemy import MetaData
from pandas import DataFrame
from pandas.io.sql import SQLTable, pandasSQL_builder, SQLDatabase
import psycopg2
import codecs
import tempfile
import datetime
import csv
import gzip
import zlib
from sqlalchemy_redshift import commands as redshift_commands
from sqlalchemy.schema import CreateTable
import sqlalchemy.types as sa_types
import logging

logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('nose').setLevel(logging.WARNING)

sortkeys = {
    'economics': ('quandl_code', 'date'),
    'series_metadata': ('quandl_code'), # All Data Spreadsheet
    'indicator_metadata': ('indicator_code'), # List-Indicators Spreadsheet
}

distkeys = {
    'economics': ('quandl_code'),
    'series_metadata': ('country_name'), # All Data Spreadsheet
}


class RedshiftTable(SQLTable):
    def __init__(self, table_name, schema, engine, dataframe, distkey=None, sortkey=None, column_override={}):
        self.sortkey = sortkey
        self.distkey = distkey
        self.keys = None
        self.df = dataframe
        self.engine = engine
        self.schema = schema
        self.name = table_name
        self.index = None
        self.dtype = None
        self.column_override = column_override
        self.pd_sql = SQLDatabase(engine, schema=schema)
        super().__init__(table_name, self.pd_sql, frame=self.df,
            index=False, schema=schema)
    

    def create_table_sql(self):
        from sqlalchemy import Table, Column, PrimaryKeyConstraint

        column_names_and_types_calc = \
            self._get_column_names_and_types(self._sqlalchemy_type)
        column_names_and_types = []
        for col_data in column_names_and_types_calc:
            col_name, col_type, etc = col_data
            if col_data[0] in self.column_override:
                col_type = self.column_override[col_data[0]]
            column_names_and_types.append((col_name, col_type, etc))

        # print(column_names_and_types)
        columns = [Column(name, typ, index=is_index)
                   for name, typ, is_index in column_names_and_types]

        if self.keys is not None:
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            pkc = PrimaryKeyConstraint(*keys, name=self.name + '_pk')
            columns.append(pkc)


        schema = self.schema

        # At this point, attach to new metadata, only attach to self.meta
        # once table is created.
        from sqlalchemy.schema import MetaData
        meta = MetaData(self.pd_sql, schema=schema)

        settings = {}
        if self.distkey is not None:
            settings['redshift_diststyle'] = 'KEY'
            settings['redshift_distkey'] = self.distkey

        if self.sortkey is not None:
            if isinstance(self.sortkey, tuple):
                settings['redshift_interleaved_sortkey'] = self.sortkey
            else:
                settings['redshift_sortkey'] = self.sortkey

        print('> Settings')
        print(settings)
        table = Table(self.name, meta, *columns, schema=schema, **settings)
        return str(CreateTable(table).compile(self.engine))


from sqlalchemy import create_engine
# 'jdbc:redshift://ihs-lake-doppler.cop6dfpxh7ta.us-west-2.redshift.amazonaws.com:5439/lake_one'
ENGINE = create_engine(None)

import tempfile
from contextlib import contextmanager
AWS_ACCESS_KEY_ID = None
AWS_SECRET_ACCESS_KEY = None

@contextmanager
def tempinput(data):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(data)
    temp.close()
    yield temp.name
    os.unlink(temp.name)

def monkeypatch_method(cls):
    @wraps(cls)
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator

def resolve_qualname(table_name, schema=None):
    name = '.'.join([schema, table_name]) if schema is not None else table_name
    return name

def does_table_exist(engine, schema, qualname):
    md = MetaData(engine, schema=schema, reflect=True)
    return qualname in md.tables.keys()


@monkeypatch_method(DataFrame)
def to_redshift(self, table_name, engine, bucket, folder, keypath=None,
                schema=None, if_exists='fail', index=True, index_label=None,
                aws_access_key_id=None, aws_secret_access_key=None,
                columns=None, null_as='', emptyasnull=True, sortkey=None, distkey=None,
                column_override={}):
    """
    Write a DataFrame to redshift via S3
    Parameters
    =========
    table_name : str. (unqualified) name in redshift
    engine : SQLA engine
    bucket : str; s3 bucket
    keypath : str; keypath in s3 (without bucket name)
    schema : redshift schema
    if_exits : str; {'fail', 'append', 'replace'}
    index : bool; include DataFrames index
    index_label : bool; label for the index
    aws_access_key_id / aws_secret_access_key : from ~/.boto by default
    columns : subset of columns to include
    null_as : treat these as null
    emptyasnull bool; whether '' is null
    """
    if keypath is None:
        keypath = table_name + '.gz'
    table_name = table_name.replace('/', '')
    dataset_name = table_name
    url = self.to_s3(keypath, bucket=bucket, index=index,
                     index_label=index_label, header=False, folder=folder)
    qualname = resolve_qualname(table_name, schema)

    table = SQLTable(table_name, pandasSQL_builder(engine, schema=schema),
                     self, if_exists=if_exists, index=index)

    table = RedshiftTable(table_name, schema, ENGINE, self, sortkey=sortkey, column_override=column_override, distkey=distkey)
    if columns is None:
        columns = ''
    else:
        columns = '()'.format(','.join(columns))
    print("Creating table {}".format(qualname))

    if table.exists():
        if if_exists == 'fail':
            raise ValueError("Table Exists")
        elif if_exists == 'append':
            queue = []
        elif if_exists == 'recreate':
            queue = ['drop table {} cascade'.format(qualname), table.create_table_sql()]
        elif if_exists == 'replace':
            queue = ['truncate table {}'.format(qualname)]
        else:
            raise ValueError("Bad option for `if_exists`")
    else:
        queue = [table.create_table_sql()]

    with engine.begin() as con:
        for stmt in queue:
            con.execute(stmt)

    conn = psycopg2.connect(database=engine.url.database,
                            user=engine.url.username,
                            password=engine.url.password,
                            host=engine.url.host,
                            port=engine.url.port,
                            sslmode='require')
    cur = conn.cursor()
    if null_as is not None:
        null_as = "NULL AS '{}'".format(null_as)
    else:
        null_as = ''

    if emptyasnull:
        emptyasnull = "EMPTYASNULL"
    else:
        emptyasnull = ''

    full_keypath = 's3://{}/{}'.format(bucket, url)
    print("COPYing")
    copy_command = redshift_commands.CopyCommand(table, full_keypath, access_key_id=AWS_ACCESS_KEY_ID,
        secret_access_key=AWS_SECRET_ACCESS_KEY, delimiter='|', blanks_as_null=True,
        empty_as_null=True, encoding='utf-8',
        remove_quotes=True, roundec=False,
        time_format=None, compression='GZIP',
        trim_blanks=True,
        comp_update=True, max_error=5, stat_update=True)

    with engine.begin() as con:
        con.execute(copy_command)
    # Now adding user 'analyst' to access
    if schema is None:
        schema = 'public'
    # stmt = ("grant all on all tables in schema {} to analyst ".format(schema))
    # cur.execute(stmt)
    # cur.execute(("grant all privileges on schema {} to analyst ".format(schema)))
    conn.commit()
    conn.close()


@monkeypatch_method(DataFrame)
def to_s3(self, keypath, bucket, folder, index=False, index_label='index',
                header=False):
    s3conn = boto.resource('s3')
    url = folder + '/' + keypath
    csv_file = StringIO()
    self.to_csv(csv_file,
        index=index,
        header=header,
        encoding='utf-8',
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True, sep='|')
    compressed = BytesIO()
    gz = gzip.GzipFile(filename=keypath, fileobj=compressed, mode='w')
    gz.write(csv_file.getvalue().encode('utf-8'))
    gz.close()
    content = BytesIO(compressed.getvalue())
    s3conn.Object(bucket, url).put(Body=content.getvalue())
    return url


def push_datasets_to_amazon_redshift(dataset, schema='ihs_lei', column_overrides={}, tables_to_recreate=['all']):
    for dataset_name, dataframe in dataset.items():
        print('[INFO] Pushing dataset named', dataset_name, 'to schema', schema)
        dataframe.columns = [x.lower().replace(' ', '_') for x in dataframe.columns]
        dataset_name = dataset_name.replace('/', '')
        try:
            column_override = column_overrides.get(dataset_name, {})
            print(dataset_name, 'override:', column_override)
            sortkey = None
            distkey = None
            if dataset_name in sortkeys:
                sortkey = sortkeys[dataset_name]
            if dataset_name in distkeys:
                distkey = distkeys[dataset_name]
            if dataset_name not in tables_to_recreate and 'all' not in tables_to_recreate:
                dataframe.to_redshift(dataset_name, ENGINE, 'plays-basins-database',
                    schema=schema, folder=str(datetime.date.today()),
                    if_exists='replace', index=False, emptyasnull=True, distkey=distkey,
                    sortkey=sortkey, column_override=column_override)
            else:
                dataframe.to_redshift(dataset_name, ENGINE, 'plays-basins-database',
                    schema=schema, folder=str(datetime.date.today()),
                    if_exists='recreate', index=False, emptyasnull=True, distkey=distkey,
                    sortkey=sortkey, column_override=column_override)
        except Exception as e:
            import traceback
            traceback.print_exc()
