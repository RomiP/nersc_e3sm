import datetime as dt
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, polygonize
from typing import Union
import xarray as xr


MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

VARNAMES = {
	'maxMLD':'timeMonthlyMax_max_dThreshMLD',
	'bvfml':'timeMonthly_avg_bruntVaisalaFreqML',
	'qlat':'timeMonthly_avg_latentHeatFlux',
	'lwhfd':'timeMonthly_avg_longWaveHeatFluxDown',
	'lwhfu':'timeMonthly_avg_longWaveHeatFluxUp',
	'swhf':'timeMonthly_avg_shortWaveHeatFlux',
	'qsens':'timeMonthly_avg_sensibleHeatFlux',
}

def rotate(l: Union[list, np.array], k: int) -> Union[list, np.array]:
	'''
	Rotate list or array k steps to the left
	:param l: list or array
	:param k: number of rotational steps
	:return: rotated list or array
	>>> l = [1, 2, 3, 4, 5]
	>>> a = np.array(l)
	>>> rotate(l, 1)
	[2, 3, 4, 5, 1]
	>>> rotate(l, -1)
	[5, 1, 2, 3, 4]
	>>> rotate(a, 3)
	array([4, 5, 1, 2, 3])
	>>> rotate(a, -3)
	array([3, 4, 5, 1, 2])
	'''
	if isinstance(l, list):
		l_rot = l[k:] + l[:k]
	else:
		l_rot = np.concat((l[k:], l[:k]))
	return l_rot


def dt64_y_m_d(date: np.datetime64) -> (np.array, np.array, np.array):
	'''
	Extract year, month, and day components from a NumPy datetime64 object.

    This function converts a given `np.datetime64` (or array of datetime64 values)
    into three separate NumPy arrays representing the year, month, and day.

	:param date: np.datetime64 or np.ndarray
        A single datetime64 value or an array of datetime64 values.

	:return: tuple of np.ndarray
				A tuple containing three arrays:
				- years : np.ndarray of int
					The year component (e.g., 2025).
				- months : np.ndarray of int
					The month component (1–12).
				- days : np.ndarray of int
					The day component (1–31).
	'''
	years = date.astype('datetime64[Y]').astype(int) + 1970
	months = date.astype('datetime64[M]').astype(int) % 12 + 1
	days = date.astype('datetime64[D]') - date.astype('datetime64[M]') + 1
	return years, months, days


def normalize(x: np.ndarray) -> np.ndarray:
	norm = (x - np.mean(x)) / np.std(x)
	return norm


def make_monthly_date_list(startdate: dt.datetime, enddate: dt.datetime) -> list[dt.datetime]:
	dates = []
	while startdate < enddate:
		dates.append(startdate)
		startdate += relativedelta(months=1)
	return dates


def geopolygon_mask(geojson_polygon: str, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
	"""
	Determine whether each point in a 2D array of [longitude, latitude] coordinates
	lies inside a GeoJSON polygon using Shapely's vectorized API.

	Parameters
	----------
	geojson_polygon : string
		A GeoJSON dictionary representing a Polygon or MultiPolygon.
		*** Note when casting dict to str: make sure nested strings (i.e. dict keys) are noted with double quotes -> ""
	lons : np.ndarray
		A NumPy 1-D array of length N.
	lats : np.ndarray
		A NumPy 1-D array of length N.

	Returns
	-------
	np.ndarray
		A NumPy array of shape (N,) with boolean value True if the point is inside the polygon, else False.

	Examples
	--------
	>>> geojson_poly = '''
	... {
	...		"type": "Polygon",
	...		"coordinates": [
	...			[
	...				[-80.0, 25.0],
	...				[-80.0, 26.0],
	...				[-79.0, 26.0],
	...				[-79.0, 25.0],
	...				[-80.0, 25.0]
	...			]
	...		]
	...	}
	... '''
	>>> coords = np.array([[-79.5, 25.5], [-54.1, 58.6], [-80.1, 25.2]])
	>>> lons = np.array([-79.5, -54.1, -80.1])
	>>> lats = np.array([25.5, 58.6, 25.2])
	>>> print(geopolygon_mask(geojson_poly, lons, lats))
	[1 0 0]
	>>> poly_file = '/global/cfs/projectdirs/m1199/romina/PycharmProjects/nersc/test_files/labsea_test.geojson'
	>>> my_json = open(poly_file).read()
	>>> print(geopolygon_mask(my_json, lons, lats))
	[0 1 0]
	"""
	# Convert GeoJSON (as a string) to Shapely geometry
	polygon = shapely.from_geojson(geojson_polygon)

	# Create Shapely points from coordinates
	points = shapely.points(lons, lats)

	# Check containment
	return shapely.contains(polygon, points)


def make_geopoly_from_bath(bath, depth):
	crs = "EPSG:4326"
	fig, ax = plt.subplots()
	# One isobath level. If your dataset uses positive depths, set isobath_value accordingly.
	CS = ax.contour(bath.lon.values, bath.lat.values, bath.elevation.values, levels=[-depth])

	plt.close(fig)  # no need to render

	polygons = []

	# Convert contour paths to polygons (only closed rings)

	# Convert contour paths to segments
	segments = []
	for col in CS.collections:
		for path in col.get_paths():
			v = path.vertices  # Nx2 array of [lon, lat]
			# Remove NaNs and duplicates
			v = v[~np.isnan(v).any(axis=1)]
			if len(v) < 2:
				continue

			# If the path is closed, Matplotlib usually repeats the first point at the end.
			# We'll just create segments for consecutive pairs.
			for i in range(len(v) - 1):
				p1 = tuple(v[i])
				p2 = tuple(v[i + 1])
				if p1 != p2:
					segments.append(LineString([p1, p2]))

	if not segments:
		# Nothing extracted at this level
		return gpd.GeoDataFrame({"level": []}, geometry=[], crs=crs)

	# Merge and polygonize the segment network
	merged = unary_union(segments)
	polys = list(polygonize(merged))

	# Filter degenerate/super small polygons
	valid_polys = []
	for poly in polys:
		# reject invalid or empty
		if (poly is None) or poly.is_empty or (not poly.is_valid):
			continue
		# reject rings with too few vertices (outer boundary check)
		min_vertices = 3
		if hasattr(poly, "exterior") and len(poly.exterior.coords) < min_vertices:
			continue
		valid_polys.append(poly)

	if not valid_polys:
		return gpd.GeoDataFrame({"level": []}, geometry=[], crs=crs)

	gdf = gpd.GeoDataFrame({"level": [depth] * len(valid_polys)}, geometry=valid_polys, crs=crs)

	# Optional simplification (careful: can alter topology)
	simplify_tolerance = 10
	if simplify_tolerance and simplify_tolerance > 0:
		gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)

		return gdf


if __name__ == '__main__':

	maskgeojson = 'regional_masks/LabSeaWhole.geojson'
	lat, lon, ncells
	geopolygon_mask()