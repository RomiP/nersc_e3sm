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

	# cmap = cmr.ocean.reversed()
	# cnorm = mcolors.Normalize(vmin=bath.min(), vmax=bath.max())
	#
	# polys = make_mpas_polygon(ncells[idx]+1)
	# feature = ShapelyFeature(
	# 	polys,
	# 	ccrs.PlateCarree(),
	# 	facecolor=cmap(cnorm(bath)),
	# 	edgecolor='none',
	# 	linewidth=1
	# )
	#
	# ax.add_feature(feature)
	fig, ax = unstructured_pcolor(lat, lon,
								  bath,
								  extent=extent,
								  interp='mosaic',
								  cellnums=ncells[idx]+1,
								  projname='PlateCarree',
								  gridlines=True,
								  landmask=True,
								  clabel='Bathymetry (m)',
								  cmap=cmr.ocean.reversed(),)

	# plt.scatter(lon, lat, c=bath, cmap=cmr.ocean.reversed(),
	# 			s=80, alpha=0.5,
	# 			transform=ccrs.PlateCarree())

	# cbar = plt.colorbar()
	# cbar.set_label('Bathymetry (m)')

	# # Put a background image on for nice sea rendering.
	# ax.add_feature(cfeature.LAND, facecolor='grey')

	# draw the flux gate
	geoms = gate.geometry.values
	coords = np.array(geoms[0].coords)
	feat = ShapelyFeature(geoms, ccrs.PlateCarree(),
						  edgecolor='red',
						  facecolor='none',
						  linewidth=1,)
	ax.add_feature(feat)

	# plot normal vector
	norm = normal_vector(gate)
	if all(norm) > 0:
		norm = -norm
	mid = np.mean(coords, axis=0)
	print('quiver')
	ax.quiver(
		np.array([mid[0]]), np.array([mid[1]]),
		np.array([norm[0]]), np.array([norm[1]]),
		angles='xy',
		scale_units='inches',
		scale=1.25,
		linewidths=5,
		color='r',
		zorder=99,
		transform=ccrs.PlateCarree()
	)

	# plot the cells that make up the flux gate
	print('scatter')
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
	istart = arg_nearest_geo(start, lons, lats)
	istop = arg_nearest_geo(stop, lons, lats)

	'''
	Create a distance array dist[] of size V and initialize all values to infinity
	since no paths are known yet. Set the distance of the source vertex to 0 and 
	insert it into the priority queue.
	'''
	dist = {i:np.inf for i in ncells[mask]}
	prev = {istart:None}
	dist[istart] = 0.0
	pq = []
	heapq.heappush(pq, (0, istart))

	# pq = [(0.0, istart)]

	# pend = (lats[istop], lons[istop])
	# pstart = (lats[istart], lons[istart])
	# d = geodesic(pstart, pend).km
	# pq = [(d, istart)]

	while pq:
		'''
		While the priority queue is not empty, remove the vertex 
		with the smallest distance value.
		'''
		d, cell = heapq.heappop(pq)

		if cell == istop:
			break

		'''
		Check if the popped distance is greater than the recorded distance for this 
		vertex, it means this vertex has already been processed with a smaller 
		distance, so skip it
		'''
		if d > dist[cell]:
			continue

		# get neighbours of current cell
		currcell = mesh.sel(nCells=cell)
		ineighbours = currcell.cellsOnCell.values
		ineighbours = [i - 1 for i in ineighbours if i > 0 and mask[i-1]]

		point1 = (lats[cell], lons[cell])  # current cell
		for n in ineighbours:
			point2 = (lats[n], lons[n])  # neighbour

			'''
			For each neighbor n of currcell, check if the path through currcell gives a smaller 
			distance than the current dist[n]. If it does, update dist[n] = dist[currcell] + edge weight(d) 
			and push (dist[n], n) into the priority queue.	
			'''

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
	buffer = 1
	idx = ((lon > bounds[0] - buffer) & (lon < bounds[2] + buffer) &
		   (lat > bounds[1] - buffer) & (lat < bounds[3] + buffer))
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

def calculate_flux(data, gateline, mask, scalearea=True):

	u = data[VARNAMES['vzonal']].values[mask,:]
	v = data[VARNAMES['vmeridional']].values[mask,:]

	norm = normal_vector(gateline)
	if all(norm) > 0:
		# I want norm to point
		norm = -norm
	flux = (u*norm[0] + v*norm[1])

	if scalearea:
		coords = gateline.get_coordinates().values[:, ::-1]
		gatelen = geodesic(coords[0], coords[1]).km
		n = np.sum(mask)
		dx = gatelen / n
		mesh = xr.open_dataset(MESHFILE_OCN).isel(Time=0)
		dz = mesh.layerThickness.values[mask, :]
		da = dx * dz
		flux *= da

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

	# sort by distance along the gate (not cellID)
	lats = np.degrees(mesh.latCell.values[mask])
	lons = np.degrees(mesh.lonCell.values[mask]) - 360
	dist = (lats - coords[0, 0]) ** 2 + (lons - coords[0, 1]) ** 2
	idx = np.argsort(dist)

	cmap = plt.get_cmap('coolwarm').copy()
	flux[~bathmask] = np.nan
	plt.pcolormesh(x, z, flux[:,idx], cmap=cmap)
	cbar = plt.colorbar()
	cbar.set_label('Across-gate Velocity ($m/s$)')
	# Get the current color limits
	vmin, vmax = plt.gci().get_clim()
	cmax = max(abs(vmin), abs(vmax))
	plt.clim(-cmax, cmax)

	# 2. Set the color for NaN values (e.g., 'gray' or 'red')
	cmap.set_bad(color='tab:gray')

	plt.ylim([0, 2000])
	ax = plt.gca()
	ax.invert_yaxis()
	plt.ylabel('Depth (m)')
	plt.xlabel('Distance along gate (km)')




if __name__ == '__main__':
	print('1234')
	root = 'regional_masks/flux_gates/'
	fname = 'denmark_strait_sill'
	gate_line = gpd.read_file(root + fname + '.geojson')
	if not os.path.exists(root + fname + '.json') or True:
		make_flux_gate_mask(root, fname)

	mesh_gate = json.load(open(root + fname + '.json'))
	mask = np.array(mesh_gate['mask']).astype(bool)
	cellnums = np.array(mesh_gate['cellnums'])

	# %% calculate and plot flux
	# dates = make_monthly_date_list(dt.datetime(1950,1,1),
	# 							   dt.datetime(1960,1,1))
	#
	# flux = 0
	# for d in tqdm(dates):
	# 	data = get_mpaso_file_by_date(2000, 1, 'historical0201').isel(Time=0)
	# 	flux += calculate_flux(data,gate_line, mask, scalearea=False)
	#
	# flux /= len(dates)
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

	plt.show()