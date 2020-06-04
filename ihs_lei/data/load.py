from sqlalchemy import *
from sqlalchemy.orm import *
import pandas as pd
import sqlalchemy
import arrow
from pathlib import Path
from ihs_lei.settings import CONNECTION_STR


class DatabaseReflector(object):
    engine = None
    meta = None
    Session = None

    def __init__(self, connection_string=CONNECTION_STR):
        self.connection_string = connection_string
        self.connect()
    
    def connect(self):
        if self.engine is None:
            self.engine = create_engine(self.connection_string, echo=False)
        if self.meta is None:
            self.meta = MetaData()
            self.meta.bind = self.engine
            self.meta.reflect(bind=self.engine)
        if self.Session is None:
            self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        self.connect()
        return self.Session()

    def reverse_table(self, table_name, pkfield=None, schema='ihs_lei'):
        self.connect()
        if table_name in self.meta.tables.keys():
            return self.meta.tables[table_name]
        return Table(table_name, self.meta, autoload=True, schema=schema)


class EconomicDataLoad(object):
    """docstring for EconomicDataLoad"""
    def __init__(self):
        self.db = DatabaseReflector()
        self.economic_table = self.db.reverse_table('t_economics')
        select = sqlalchemy.sql.select([self.economic_table])
        self.economic_data = pd.read_sql(select, self.db.engine)
        

class EconomicMetadataLoad(object):
    """docstring for EconomicMetadataLoad"""
    def __init__(self):
        self.db = DatabaseReflector()
        self.indicator_metadata_table = self.db.reverse_table('economics')
        self.series_metadata_table = self.db.reverse_table('series_metadata')
        self.sources_table = self.db.reverse_table('sources')
        select = sqlalchemy.sql.select([self.economic_table])
        self.economic_data = pd.read_sql(select, self.db.engine)

    def _create_metadata(self):
        pass