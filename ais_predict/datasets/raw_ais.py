from ast import Str
import pandas as pd
import geopandas as gpd


def convert_ais_df_to_trajectories(df: gpd.GeoDataFrame, max_time_between_samples: float = 60*15, max_duration: float = 30*60, min_samples: int = 4, min_duration: float = 15*60 , max_samples: int = 20, crs: str = "epsg:25832"):
	"""
	Converts a dataframe of AIS samples into trajectories

	Expects the following columns in df:
	- mmsi (int)
	- position (shapely.Point) with crs to correct coordinate projection
	- sog (float) in knots
	- cog (float) in degrees
	- timestamp (pandas.DateTimeIndex)
	"""
	d = df[["mmsi", "position", "timestamp", "cog", "sog"]].sort_values(by=["mmsi", "timestamp"])
	d["traj_length"] = 0
	d["duration"] = 0
	d.position = d.position.to_crs(crs)
	ix = d.duration < max_duration
	for i in range(1, max_samples):
		prev_suffix = f"_{i-1}" if i > 1 else ""
		lead = d[["mmsi", "timestamp", "position", "cog", "sog"]].shift(-i)

		length_ix = d.traj_length == i-1
		mmsi_ix = lead.mmsi == d.mmsi
		timestamp_ix = (lead.timestamp - d["timestamp" + prev_suffix]).dt.seconds < max_time_between_samples
		duration = (lead.timestamp - d.timestamp).dt.seconds
		duration_ix = duration < max_duration

		ix &= length_ix & mmsi_ix & timestamp_ix & duration_ix
		d = d.join(lead.loc[ix], rsuffix=f"_{i}")
		d.loc[ix, "traj_length"] += 1
		d.loc[ix, "duration"] = duration[ix]

		if i < min_samples:
			d = d.loc[ix]
			ix = ix[ix]
	
	d = d.loc[d.duration > min_duration]
	d = d.loc[d.traj_length > min_samples]
	return d