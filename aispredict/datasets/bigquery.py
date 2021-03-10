import geopandas as gpd
import pandas as pd
from google.cloud import bigquery, bigquery_storage
import google.auth
from shapely.geometry.polygon import Polygon
import shapely.wkt

def download(limit: int = 1000, within: Polygon=None, mmsi: list = None, project_id: str="master-thesis-305112" ,credentials=None) -> gpd.GeoDataFrame:
    """Creates a query job in Bigquery and downloades the result into a GeoPandas Dataframe
    

    Keyword Arguments:
    limit -- number of results to include. None returns all results.
    within -- coordinate filter, only points within the this polygon is included. None returns all results.
    mmsi -- list-like containing mmsi values to include in the result. None returns all.
    credentials -- google cloud credentials object. None use the google.auth.default.
    project_id -- google cloud project id to use for billing.
    """

    if credentials is None:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    
    # Make clients.
    bq = bigquery.Client(credentials=credentials, project=project_id,)
    bqstorage = bigquery_storage.BigQueryReadClient(credentials=credentials)

    limit = "LIMIT " + str(limit) if limit else ""
    within = f"AND ST_WITHIN(ST_GEOGPOINT(current_sample.long, current_sample.lat), ST_GEOGFROMTEXT('{str(within)}'))" if within else ""
    mmsi = "'" + "','".join(mmsi) + "'" if mmsi else None
    mmsi = f"AND CAST(mmsi AS STRING) IN ({mmsi})" if mmsi else ""
    query = f"""
    SELECT
    mmsi,
    UNIX_SECONDS(current_sample.timestamp) AS unix_timestamp,
    current_sample.cog AS cog,
    current_sample.sog AS sog,
    ST_GEOGPOINT(current_sample.long, current_sample.lat) AS geometry,
    UNIX_SECONDS(previous_sample.timestamp) AS prev_unix_timestamp,
    previous_sample.cog AS prev_cog,
    previous_sample.sog AS prev_sog,
    ST_GEOGPOINT(previous_sample.long, previous_sample.lat) AS prev_geometry,
    FROM `master-thesis-305112.ais.raw_with_prev`
    WHERE previous_sample.timestamp IS NOT NULL
    AND TIMESTAMP_DIFF(current_sample.timestamp, previous_sample.timestamp, MINUTE) < 30
    {within}
    {mmsi}
    ORDER BY mmsi, current_sample.timestamp
    {limit}
    """
    df = bq.query(query).result().to_dataframe(bqstorage_client=bqstorage)
    df.geometry = gpd.GeoSeries.from_wkt(df.geometry, crs="wgs84")
    df.prev_geometry = gpd.GeoSeries.from_wkt(df.prev_geometry, crs="wgs84")
    df = gpd.GeoDataFrame(df, geometry="geometry")
    return df