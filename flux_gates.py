import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmasher as cmr
from cartopy.feature import ShapelyFeature
import cartopy.mpl.ticker as ctk
from geopy.distance import geodesic
import heapq

# from analysis.plot_geometric_features import extent
from helpers import *
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shapely.geometry as geom
import numpy as np
from open_e3sm_files import *
from pyproj import Geod
import xarray as xr

from shapely.geometry import LineString

from plot_unstructured import unstructured_pcolor, _init_proj


def plot_fluxgate(gate, mesh_lats, mesh_lons):

	minx, miny, maxx, maxy = gate.total_bounds
	buffer = 1
	extent = [minx - buffer, maxx + buffer, miny - buffer, maxy + buffer]

	fig = plt.figure()
	proj = _init_proj('LambertConformal', extent=extent)
	ax = fig.add_subplot(1, 1, 1,
						 projection=ccrs.PlateCarree(),
						 )
	ax.set_extent(extent, crs=ccrs.PlateCarree())

	lat, lon, ncells = mpaso_mesh_latlon()
	z = mpaso_depth()
	buffer *= 1.1
	idx = ((lon < maxx + buffer) & (lon > minx - buffer)
		   & (lat > miny - buffer) & (lat < maxy + buffer))
	mesh = xr.load_dataset(MESHFILE_OCN).isel(Time=0)
	izmax = mesh.maxLevelCell.values[idx] - 1
	bath = z[izmax]
	lat = lat[idx]
	lon = lon[idx]

	cmap = cmr.ocean.reversed()
	cnorm = mcolors.Normalize(vmin=bath.min(), vmax=bath.max())

	polys = make_mpas_polygon(ncells[idx]+1)
	feature = ShapelyFeature(
		polys,
		ccrs.PlateCarree(),
		facecolor=cmap(cnorm(bath)),
		edgecolor='none',
		linewidth=1
	)

	ax.add_feature(feature)

	# fig, ax = unstructured_pcolor(lat, lon,
	# 							  # bath,
	# 							  extent=extent,
	# 							  projname='PlateCarree',
	# 							  gridlines=True,
	# 							  landmask=True,
	# 							  # dotsize=100,
	# 							  cmap=cmr.ocean.reversed(),)

	# plt.scatter(lon, lat, c=bath, cmap=cmr.ocean.reversed(),
	# 			s=80, alpha=0.5,
	# 			transform=ccrs.PlateCarree())

	# cbar = plt.colorbar()
	# cbar.set_label('Bathymetry (m)')

	# # Put a background image on for nice sea rendering.
	ax.add_feature(cfeature.LAND, facecolor='grey')

	geoms = gate.geometry.values

	coords = np.array(geoms[0].coords)
	feat = ShapelyFeature(geoms, ccrs.PlateCarree(),
						  edgecolor='red',
						  facecolor='none',
						  linewidth=1,)
	# Add to cartopy
	ax.add_feature(feat)

	# plot normal vector
	norm = normal_vector(gate)
	if all(norm) > 0:
		norm = -norm
	mid = np.mean(coords, axis=0)
	# ax.quiver(mid[0], mid[1], norm[0], norm[1], angles='xy', scale_units='xy',
	# 		  scale=0.5, color='r' , transform=ccrs.PlateCarree())
	ax.quiver(
		np.array([mid[0]]), np.array([mid[1]]),
		np.array([norm[0]]), np.array([norm[1]]),
		angles='xy',
		scale_units='xy',
		scale=0.5,
		linewidths=5,
		color='r',
		zorder=99,
		transform=ccrs.PlateCarree()
	)

	ax.scatter(mesh_lons, mesh_lats, c='k', marker='o', alpha=0.5, zorder=98,
			   transform=ccrs.PlateCarree())

	return fig, ax

def buildCellPoly(cell, latV, lonV):
	verts = cell.verticesOnCell.values

	poly = geom.Polygon([
		(lonV[v-1], latV[v-1]) for v in verts if v > 0
	])
	return poly

def djikstra(start, stop, mesh, mask=None):
	lons = np.degrees(mesh.lonCell.values)
	lats = np.degrees(mesh.latCell.values)
	lons[lons > 180] -= 360
	ncells = mesh.nCells.values

	gateline = geom.LineString((start, stop))

	if mask is None:
		mask = np.ones(ncells, dtype=bool)


	dist = {i:np.inf for i in ncells[mask]}

	istart = arg_nearest_geo(start, lons, lats)
	istop = arg_nearest_geo(stop, lons, lats)

	prev = {istart:None}

	dist[istart] = 0

	pq = [(0.0, istart)]

	while pq:
		d, cell = pq.pop(0)
		if cell == istop:
			break
		currcell = mesh.sel(nCells=cell)
		ineighbours = currcell.cellsOnCell.values
		ineighbours = [i - 1 for i in ineighbours if i > 0 and mask[i-1]]

		point1 = (lats[cell], lons[cell])  # current cell
		for n in ineighbours:
			point2 = (lats[n], lons[n])  # dist to neighbour
			# Calculate distance in kilometers
			dgate = shapely.distance(gateline, geom.Point([lons[n], lats[n]]))/360*6000
			x = geodesic(point1, point2).km + d + dgate
			if x < dist[n]:
				dist[n] = x
				prev[n] = cell
				heapq.heappush(pq, (x, n))

	path = [istop]
	while prev[path[-1]] is not None:
		path.append(prev[path[-1]])


	return path

def make_flux_gate_mask(root, fname):
	# root = 'regional_masks/flux_gates/'
	# fname = 'denmark_strait'
	gate = gpd.read_file(root + fname + '.geojson')
	line = gate.geometry.values[0]
	coords = list(line.coords)

	bounds = gate.total_bounds
	lat, lon, ncells = mpaso_mesh_latlon()
	idx = (lon > bounds[0]) & (lon < bounds[2]) & (lat > bounds[1]) & (lat < bounds[3])
	mesh = xr.load_dataset(MESHFILE_OCN)  # .sel(nCells=idx)

	mesh_gate = djikstra(coords[0], coords[1], mesh, idx)

	gate_mask = list(np.zeros(len(ncells)))
	for i in mesh_gate:
		gate_mask[i] = 1

	mesh_gate = {
		'cellnums': [int(i) for i in mesh_gate],
		'mask': gate_mask,
	}

	with open(root + fname + '.json', 'w') as f:
		json.dump(mesh_gate, f, indent=4)

def normal_vector(line):
	line = line.geometry.values[0]
	coords = list(line.coords)


	dx = coords[1][0] - coords[0][0]
	dy = coords[1][1] - coords[0][1]

	dx, dy = -dy, dx

	vec = np.array([dx, dy])
	vec /= np.linalg.norm(vec)

	return vec

def calculate_flux(data, gateline, mask):

	coords = gateline.get_coordinates().values[:, ::-1]
	gatelen = geodesic(coords[0], coords[1]).km
	n = np.sum(mask)
	dx = gatelen / n


	data = get_mpaso_file_by_date(2000, 1, 'historical0101').isel(Time=0)
	mesh = xr.open_dataset(MESHFILE_OCN).isel(Time=0)
	cellIDs = (mesh.nCells.values + 1)[mask]

	dz = mesh.layerThickness.values[mask, :]

	#
	# # cellIDs = {cellIDs[i]:i for i in range(len(cellIDs))}
	#
	# cellsOnEdge = mesh.cellsOnEdge.values - 1  # to 0-based
	#
	# cell_set = set(cellIDs - 1)
	#
	# edge_mask = np.array([
	# 	(c0 in cell_set) ^ (c1 in cell_set) # xor for outer perim edges
	# 	for c0, c1 in cellsOnEdge
	# ])
	#
	# edge_ids = np.where(edge_mask)[0]
	#
	# gatemesh = mesh.sel(nCells=mask, nEdges=edge_mask).isel(Time=0)
	# bottomdepth = gatemesh.bottomDepth.values
	# dz = gatemesh.layerThickness.values
	# dx = gatemesh.dvEdge.values
	#
	# data = data.sel(nCells=mask, nEdges=edge_mask).isel(Time=0)


	u = data[VARNAMES['vzonal']].values[mask,:]
	v = data[VARNAMES['vmeridional']].values[mask,:]

	da = dx * dz
	norm = normal_vector(gateline)
	if all(norm) > 0:
		# I want norm to point
		norm = -norm
	flux = (u*norm[0] + v*norm[1]) * da

	return flux.T

def plot_flux(flux, gateline, mask):

	mesh = xr.open_dataset(MESHFILE_OCN).isel(Time=0)
	z = mpaso_depth()
	dz = mesh.layerThickness.values[mask, :].T
	bathmask = dz > 0

	coords = gateline.get_coordinates().values[:, ::-1]
	gatelen = geodesic(coords[0], coords[1]).km
	n = np.sum(mask)
	dx = gatelen / n
	x = np.arange(n) * dx

	cmap = plt.get_cmap('coolwarm').copy()
	flux[~bathmask] = np.nan
	plt.pcolormesh(x, z, flux, cmap=cmap)
	cbar = plt.colorbar()
	cbar.set_label('Volume flux ($m^3/s$)')
	# Get the current color limits
	vmin, vmax = plt.gci().get_clim()
	cmax = max(abs(vmin), abs(vmax))
	plt.clim(-cmax, cmax)

	# 2. Set the color for NaN values (e.g., 'gray' or 'red')
	cmap.set_bad(color='tab:gray')

	plt.ylim([0, 2000])
	ax = plt.gca()
	ax.invert_yaxis()





if __name__ == '__main__':
	root = 'regional_masks/flux_gates/'
	fname = 'denmark_strait2'
	gate_line = gpd.read_file(root + fname + '.geojson')
	if not os.path.exists(root + fname + '.json'):
		make_flux_gate_mask(root, fname)

	mesh_gate = json.load(open(root + fname + '.json'))
	mask = np.array(mesh_gate['mask']).astype(bool)

	# %% calculate and plot flux

	# flux = calculate_flux(1,gate_line, mask)
	# plot_flux(flux, gate_line, mask)
	#
	# plt.show()

	# %% Create mask and plot fluxgate

	lat, lon, ncells = mpaso_mesh_latlon()

	mesh_gate = json.load(open(root + fname + '.json'))
	mask = np.array(mesh_gate['mask']).astype(bool)
	gate_lats = lat[mask]
	gate_lons = lon[mask]

	plot_fluxgate(gate_line, gate_lats, gate_lons)
	# todo: add bathymetry to flux gate plot

	plt.show()