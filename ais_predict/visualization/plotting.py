import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import ais_predict.datasets.bigquery as bq


def plot(df: gpd.GeoDataFrame, *args, **kwargs):
    """ Plot geopandas dataframe
    
    This functions is intended as a unified plotting function. A map is by default added as a background.
    """
    ax = df.plot(*args, **kwargs)
    ctx.add_basemap(ax, crs=df.crs)
    return ax




