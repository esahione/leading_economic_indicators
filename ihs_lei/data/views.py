# IHS Markit Leading Economic Indicators
# Developed by Eduardo Sahione


ECONOMIC_DATA_VIEW = """
CREATE TABLE ihs_lei.t_economics
  distkey(country)
  sortkey (country, quandl_code, date)
  AS (
  SELECT ihs_lei.economics.quandl_code, ihs_lei.economics.date, ihs_lei.economics.value,
         ihs_lei.series_metadata.indicator_name, ihs_lei.series_metadata.country_name,
         ihs_lei.series_metadata.country_code, ihs_lei.series_metadata.indicator_code,
         ihs_lei.series_metadata.start_date, ihs_lei.series_metadata.frequency
  FROM ihs_lei.economics
  NATURAL JOIN ihs_lei.series_metadata
);
"""

