import json

import matplotlib.pyplot as plt
import numpy as np
import shapely
from datashader.colors import viridis
from bootstrap import *
from mesh_tools import djikstra_edges, sign_flip, edge_sign_for_direction
from open_e3sm_files import *
from plot_unstructured import *


def plot_fluxgate(gate, mesh_lats, mesh_lons, parallel=False, posquad=None):

	minx, miny, maxx, maxy = gate.total_bounds
	buffer = 1
	extent = [minx - buffer, maxx + buffer, miny - buffer, maxy + buffer]

	# fig = plt.figure()
	# proj = _init_proj('LambertConformal', extent=extent)
	# ax = fig.add_subplot(1, 1, 1,
	# 					 projection=ccrs.PlateCarree(),
	# 					 )
	# ax.set_extent(extent, crs=ccrs.PlateCarree())

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
	# plt.scatter(lon, lat, c=bath, cmap=cmr.ocean.reversed(),
	# 			s=80, alpha=0.5,
	# 			transform=ccrs.PlateCarree())

	# cbar = plt.colorbar()
	# cbar.set_label('Bathymetry (m)')

	# # Put a background image on for nice sea rendering.
	# ax.add_feature(cfeature.LAND, facecolor='grey')

	# draw the flux gate
	geoms = gate.geometry.values
	# coords = np.array(geoms[0].coords)
	coords = gate.get_coordinates().values
	feat = ShapelyFeature(geoms, ccrs.PlateCarree(),
						  edgecolor='red',
						  facecolor='none',
						  linewidth=1,)
	ax.add_feature(feat)

	# # plot normal vector
	# norm = normal_vector(gate, parallel)
	# if posquad is not None:
	# 	quad = quadrant(norm)
	# 	if not quad in posquad:
	# 		norm = - norm
	# mid = np.mean(coords, axis=0)
	# print('quiver')
	# ax.quiver(
	# 	np.array([mid[0]]), np.array([mid[1]]),
	# 	np.array([norm[0]]), np.array([norm[1]]),
	# 	angles='xy',
	# 	scale_units='inches',
	# 	scale=1.25,
	# 	linewidths=5,
	# 	color='r',
	# 	zorder=99,
	# 	transform=ccrs.PlateCarree()
	# )

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
	import heapq
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

def make_flux_gate_mask(root, fname, edges=False):
	from mesh_tools import djikstra_edges
	# root = 'regional_masks/flux_gates/'
	# fname = 'denmark_strait'
	gate = gpd.read_file(root + fname + '.geojson')
	# line = gate.geometry.values[0]
	# coords = list(line.coords)
	coords = gate.get_coordinates().values

	bounds = gate.total_bounds

	lat, lon, ncells = mpaso_mesh_latlon()
	mesh = xr.load_dataset(MESHFILE_OCN)
	if edges:
		lon = np.degrees(mesh.lonEdge.values)
		lat = np.degrees(mesh.latEdge.values)
		lon[lon > 180] -= 360
		ncells = mesh.nEdges.values

	buffer = 1
	idx = ((lon > bounds[0] - buffer) & (lon < bounds[2] + buffer) &
		   (lat > bounds[1] - buffer) & (lat < bounds[3] + buffer))

	# mesh_gate = djikstra(coords[0], coords[1], mesh, idx)
	mesh_gate = []
	for i in range(len(coords) - 1):
		if edges:
			mesh_gate += djikstra_edges(coords[i], coords[i + 1], mesh, idx)
		else:
			mesh_gate += djikstra(coords[i], coords[i + 1], mesh, idx)

	gate_mask = list(np.zeros(len(ncells)))
	for i in mesh_gate:
		gate_mask[i] = 1

	nametype = 'cell' if not edges else 'edge'
	mesh_gate = {
		f'{nametype}nums': [int(i) for i in mesh_gate],
		'mask': gate_mask,
	}

	fname += '_edges'*edges
	with open(root + fname + '.json', 'w') as f:
		json.dump(mesh_gate, f, indent=4)

def normal_vector(line, parallel=True):
	# line = line.geometry.values[0]
	# coords = list(line.coords)

	if isinstance(line, list) or isinstance(line, tuple):
		coords = line
	else:
		coords = line.get_coordinates().values


	dx = coords[1][0] - coords[0][0]
	dy = coords[1][1] - coords[0][1]

	if not parallel:
		dx, dy = -dy, dx

	vec = np.array([dx, dy])
	vec /= np.linalg.norm(vec)

	return vec


def calculate_flux(data, gateline, mask, scalearea=True, parallel=False, posquad=None, **kwargs):

	u = data[VARNAMES['vzonal']].values[mask,:]
	v = data[VARNAMES['vmeridional']].values[mask,:]

	norm = normal_vector(gateline.boundary, parallel)

	if posquad is not None:
		quad = quadrant(norm)
		if not quad in posquad:
			norm = - norm

	flux = (u*norm[0] + v*norm[1])

	if scalearea:
		coords = gateline.get_coordinates().values[:, ::-1]
		gatelen = geodesic(coords[0], coords[1]).km * 1000
		n = np.sum(mask)
		dx = gatelen / n
		mesh = xr.open_dataset(MESHFILE_OCN).isel(Time=0)
		dz = mesh.layerThickness.values[mask, :]
		da = dx * dz
		flux *= da

	return flux.T

def calculate_flux_polygon(data, polygon, mask, scalearea=True):
	u = data[VARNAMES['vzonal']]
	v = data[VARNAMES['vmeridional']]

	lat, lon, ncells = mpaso_mesh_latlon()
	gridcells = np.argwhere(mask).squeeze()
	coords = polygon.get_coordinates().values
	centre = polygon.centroid.get_coordinates().values.squeeze()

	flux = []
	for i in gridcells:
		tail = np.array([lon[i], lat[i]])
		norm = normal_vector([tail, centre])
		flux.append(u[i,:].values * norm[0] + v[i,:].values * norm[1])

	coords = np.array([lon[gridcells], lat[gridcells]]).T
	angles = np.arctan2(coords[:, 1] - centre[1], coords[:, 0] - centre[0])
	ccw_indices = np.argsort(angles).squeeze()

	flux = np.array(flux)
	flux = flux[ccw_indices]

	coords = coords[ccw_indices]


	if scalearea:
		mesh = xr.open_dataset(MESHFILE_OCN).isel(Time=0)
		coords = np.append(coords, [coords[0]], axis=0)
		dx = np.array([geodesic(coords[i], coords[i + 1]).m for i in range(len(coords) - 1)])
		dx = np.repeat([dx], 80, axis=0).T
		dz = mesh.layerThickness.values[mask, :]
		da = dx * dz
		flux *= da
		coords = coords[:-1]

	return flux.T, coords

def plot_flux(flux, gateline, mask, cmapname='coolwarm'):

	mesh = xr.open_dataset(MESHFILE_OCN).isel(Time=0)
	z = mpaso_depth()
	dz = mesh.layerThickness.values[mask, :].T
	bathmask = dz > 0

	coords = gateline.get_coordinates().values[:, ::-1]
	if coords[0][0] > coords[0][0]:
		coords[0], coords[1] = coords[1], coords[0]
	gatelen = geodesic(coords[0], coords[1]).km
	n = np.sum(mask)
	dx = gatelen / n
	x = np.arange(n) * dx

	# sort by distance along the gate (not cellID)
	lats = np.degrees(mesh.latCell.values[mask])
	lons = np.degrees(mesh.lonCell.values[mask]) - 360
	dist = (lats - coords[0, 0]) ** 2 + (lons - coords[0, 1]) ** 2
	idx = np.argsort(dist)

	cmap = plt.get_cmap(cmapname).copy()
	flux[~bathmask] = np.nan
	plt.pcolormesh(x, z, flux[:,idx], cmap=cmap)
	cbar = plt.colorbar()

	ax = plt.gca()
	# cs = plt.contour(x, z, flux[:,idx], c='k', levels=[35, 35.1, 35.2, 35.3])
	# ax.clabel(cs, cs.levels, fontsize=10)


	# Set the color for NaN values (e.g., 'gray' or 'red')
	cmap.set_bad(color='tab:gray')

	maxdepth = np.argwhere(np.any(bathmask, axis=1))[-1, 0]
	# print(z[maxdepth + 1])
	plt.ylim([0, z[maxdepth + 1]])

	ax.invert_yaxis()
	plt.ylabel('Depth (m)')
	plt.xlabel('Distance along gate (km)')

	return plt.gcf(), ax, cbar

def plot_normal_velocity(data, gate_line, mask, **kwargs):

	data = data.mean(dim='Time', skipna=True)
	flux = calculate_flux(data, gate_line, mask, scalearea=True, **kwargs) * 1e-6 # Sv
	fig, ax, cbar = plot_flux(flux, gate_line, mask)

	print(f'Total flux: {np.nansum(flux)} Sv')


	# cbar.set_label('Along-gate Velocity ($m/s$)')
	cbar.set_label('Across-gate Transport (Sv)')
	# Get the current color limits
	# plt.clim(-0.4, 0.4)
	vmin, vmax = plt.gci().get_clim()
	cmax = max(abs(vmin), abs(vmax))
	plt.clim(-cmax, cmax)

	# Add text at ends
	plt.text(1.1, 1.02, "North", va='bottom', ha='left', transform=ax.transAxes)
	plt.text(1.1, -0.02, "South", va='top', ha='left', transform=ax.transAxes)

	# plt.text(0.01, 0, "Greenland", va='bottom', ha='left', transform=ax.transAxes)
	# plt.text(0.99, 0, "Iceland", va='bottom', ha='right', transform=ax.transAxes)

	# plt.text(0.01, 0, "Greenland", va='bottom', ha='left', transform=ax.transAxes)
	plt.text(0.99, 0, "Greenland", va='bottom', ha='right', transform=ax.transAxes)

	# plt.text(0.01, 0, "To Labrador", va='bottom', ha='left', transform=ax.transAxes)
	# plt.text(0.99, 0, "To Fram", va='bottom', ha='right', transform=ax.transAxes)

	# plt.title(f'Denmark Strait Normal Velocity ({dates[0].year} - {dates[-1].year})')
	if 'title' in kwargs:
		plt.title(kwargs['title'])
	if 'saveas' in kwargs:
		plt.savefig(kwargs['saveas'])
	plt.show()

def plot_crosssection(data, runnum, varname, mask):

	data = data.mean(dim='Time')
	if varname == 'dens':
		flux = rho_e3sm(data, mask).T
	else:
		flux = data[VARNAMES[varname]].values[mask, :].T
	fig, ax, cbar = plot_flux(flux, gate_line, mask, 'turbo')

	if varname == 'sal':
		cbar.set_label('Salinity (PSU)')
		plt.clim(35, 35.3)

		plt.title(f'AR7 Line Salinity ({dates[0].year} - {dates[-1].year})')
		# plt.title(f'Denmark Strait Salinity ({dates[0].year} - {dates[-1].year})')

	elif varname == 'ocntemp':
		cbar.set_label('Temperature ($^\circ$C)')
		plt.clim(-2, 8)
		plt.title(f'Denmark Strait Temperature ({dates[0].year} - {dates[-1].year})')

	plt.savefig(f'figs/flux_gates/{fname}_{varname}_{runnum}_y{dates[0].year}-{dates[-1].year}.png')
	plt.show()

def flux_index_dataset(gatename, posquad=[1,2]):
	root = 'regional_masks/flux_gates/'

	outfile = f'/global/cfs/cdirs/m1199/romina/data/timeseries/flowrate_{gatename}_ts_historical.nc'
	gate_line = gpd.read_file(root + gatename + '.geojson')
	mesh_gate = json.load(open(root + gatename + '.json'))
	mask = np.array(mesh_gate['mask']).astype(bool)

	startdate = dt.datetime(1950, 1, 1)
	enddate = dt.datetime(2015, 1, 1)
	dates = make_monthly_date_list(startdate, enddate)
	runs = ['historical0101', 'historical0151', 'historical0201', 'historical0251', 'historical0301']


	for runname in runs:
		print(runname)
		flux = np.array([])
		for year in range(startdate.year, enddate.year):
			print(year)
			dates_1year = make_monthly_date_list(
				dt.datetime(year, 1,1),
				dt.datetime(year + 1, 1,1)
			)
			data = zip_subset_by_time(dates_1year, get_mpaso_file_by_date,
									  varnames=['vzonal', 'vmeridional'],
									  runname=runname)

			ts = flux_ts(data, mask, gate_line)
			flux = np.concat([flux, flux_ts(data, mask, gate_line, posquad=posquad)])



		# Create DataArray for this run
		da_new = xr.DataArray(
			flux.reshape(-1, 1),
			dims=('Time', 'runname'),
			coords={'Time': dates, 'runname': [runname]},
			name='flowrate'
		)

		ds_new = xr.Dataset({'flowrate': da_new})

		# --- Save / Append logic ---
		if os.path.exists(outfile):
			print("Appending to existing file...")

			# ds_existing = xr.open_dataset(outfile)

			with xr.open_dataset(outfile) as ds_existing:
				ds_combined = xr.concat([ds_existing, ds_new], dim='runname')

				# Combine along runname dimension
				ds_combined = xr.concat([ds_existing, ds_new], dim='runname')

				# Optional: remove duplicate runnames if rerunning
				_, index = np.unique(ds_combined['runname'], return_index=True)
				ds_combined = ds_combined.isel(runname=index)

			ds_combined.to_netcdf(outfile, mode='w')

		else:
			print("Creating new file...")
			ds_new.attrs['units'] = 'sverdrup'
			ds_new.attrs['positive_direction'] = 'inward',
			ds_new.attrs['description'] = 'Volume transport into the central labrador sea'
			ds_new.to_netcdf(outfile)

def flux_ts(data, gatemask, gateline, depth=None, posquad=None):
	if depth is None:
		mesh = xr.open_dataset(MESHFILE_OCN).isel(Time=0)
		depth = mesh.layerThickness.values[gatemask, :] > 0
		depth = depth.T

	flux = []
	for i in range(len(data.Time.values)):
		if all(gateline.geom_type == 'Polygon'):
			flow_rate, coords = calculate_flux_polygon(data.isel(Time=i), gateline, gatemask)
		else:
			flow_rate = calculate_flux(data.isel(Time=i), gateline, gatemask, posquad=posquad)
		flux.append(np.nansum(flow_rate[depth])*1e-6)

	flux = np.array(flux)
	return flux

def plot_TS_diagram(data, runnum, depth, mask):
	data = data.mean(dim='Time')
	z = mpaso_depth()
	n = len(z)
	zidx = np.argmin(np.abs(z-depth))

	mesh = xr.open_dataset(MESHFILE_OCN).isel(Time=0)
	dz = mesh.layerThickness.values[mask, :].T
	bathmask = dz > 0

	s = data[VARNAMES['sal']].values[mask, :].T
	t = data[VARNAMES['ocntemp']].values[mask, :].T

	s[~bathmask] = np.nan
	t[~bathmask] = np.nan

	plt.scatter(np.ravel(s), np.ravel(t), c=z.repeat(np.sum(mask)),
				cmap='viridis', alpha=0.2)
	plt.colorbar()

	s = np.linspace(np.nanmin(s), np.nanmax(s), 20, endpoint=True)
	t = np.linspace(np.nanmin(t), np.nanmax(t), 20, endpoint=True)
	s, t = np.meshgrid(s, t)
	dens = rho(s, t, depth)

	cs = plt.contour(s, t, dens, colors='k', linestyles='--')
	# Add labels to the contours
	plt.clabel(cs, inline=True, fontsize=10)

	plt.xlabel('Salinity (PSU)')
	plt.ylabel('Temperature ($^\circ$C)')
	plt.show()

# 	todo: add freezing point to TS diagram

def linesegment_normal(line, **kwargs):
	pass

def edgeflux_dataset(mask_name, polygon=False, normalvec=[0,1], cmapname='coolwarm'):
	# get edge mask
	root = 'regional_masks/flux_gates/'
	mesh_gate = json.load(open(root + mask_name + '_edges.json'))
	mask = np.array(mesh_gate['mask']).astype(bool)
	mesh = xr.open_dataset(MESHFILE_OCN)
	z = mpaso_depth(mesh)

	# get gate normal vector, calculate sign convention
	gate_line = gpd.read_file(root + '../'*polygon + mask_name + '.geojson')

	cells = mesh.cellsOnEdge.isel(nEdges=mesh_gate['edgenums'], TWO=0).values - 1
	dz = mesh.layerThickness.values.squeeze().T[:, cells]
	bathmask = dz > 0

	if polygon:
		coords = gate_line.get_coordinates().values
		centre = gate_line.centroid.get_coordinates().values.squeeze()
		sign, dot = edge_sign_for_direction(mesh_gate['edgenums'], mesh, centre, polygon=True)
		sign = np.tile(sign, (80, 1))
	else:

		# norm = normal_vector(gate_line.boundary, parallel=False)
		# if posquad is not None:
		# 	quad = quadrant(norm)
		# 	if not quad in posquad:
		# 		norm = - norm

		sign, dot = edge_sign_for_direction(mesh_gate['edgenums'], mesh, normalvec)
		sign = np.tile(sign, (80, 1))


	# add 180 deg to angleEdge to get consistent direction
	mesh = mesh.isel(nEdges=mesh_gate['edgenums'])
	# angleedge = mesh.angleEdge.values
	# angleedge[angleedge < 0] += np.pi
	# angleedge = np.tile(angleedge, (80, 1))

	# determine distance of edge centre along gate - sort
	lons = np.degrees(mesh.lonEdge.values)
	lats = np.degrees(mesh.latEdge.values)
	lons[lons > 180] -= 360
	coords = gate_line.get_coordinates().values[:, ::-1]
	if polygon:
		dx = lons - centre[0]
		dy = lats - centre[1]
		dist = np.atan2(dy, dx)
	else:
		dist = (lats - coords[0, 0]) ** 2 + (lons - coords[0, 1]) ** 2
	idx = np.argsort(dist)
	edgenums = np.array(mesh_gate['edgenums'])[idx]

	# calculate edge area
	dv = mesh.dvEdge.values[idx]
	dx = np.cumsum(dv)
	dv = np.tile(dv, (80, 1))
	dA = dv * dz[:,idx]

	startdate = dt.datetime(1950, 1, 1)
	enddate = dt.datetime(2015, 1, 1)
	dates = make_monthly_date_list(startdate, enddate)
	outfile = f'/global/cfs/cdirs/m1199/romina/data/timeseries/mleflowrate_{mask_name}_edges_ts_historical.nc'
	runs = ['historical0101', 'historical0151', 'historical0201', 'historical0251', 'historical0301']

	for runname in runs:
		print(runname)
		da_new = None
		for date in tqdm(dates):
			data = get_mpaso_file_by_date(1950, 1, 'historical0101').isel(Time=0)
			f = data['timeMonthly_avg_normalMLEvelocity'].isel(nEdges=edgenums)
			y = f.values.squeeze().T * sign[:, idx]
			y[~bathmask[:, idx]] = np.nan
			f.data = y.T

			if da_new is not None:
				da_new = xr.concat([da_new, f], dim='Time')
			else:
				da_new = f
			f.close()

		ds_new = xr.Dataset({'vmlenormedge': da_new})
		# --- Save / Append logic ---
		if os.path.exists(outfile):
			print("Appending to existing file...")

			# ds_existing = xr.open_dataset(outfile)

			with xr.open_dataset(outfile) as ds_existing:
				ds_combined = xr.concat([ds_existing, ds_new], dim='runname')

				# Combine along runname dimension
				ds_combined = xr.concat([ds_existing, ds_new], dim='runname')

				# Optional: remove duplicate runnames if rerunning
				_, index = np.unique(ds_combined['runname'], return_index=True)
				ds_combined = ds_combined.isel(runname=index)

			ds_combined.to_netcdf(outfile, mode='w')

		else:
			print("Creating new file...")
			# ds_new.assign_coords(runname=)
			ds_new.attrs['units'] = 'sverdrup'
			ds_new.attrs['positive_direction'] = 'inward',
			ds_new.attrs['description'] = 'Volume transport into central Lab Sea'
			ds_new["edgeLength"] = xr.DataArray(mesh.dvEdge.values[idx], dims='nEdges')
			ds_new["edgeArea"] = xr.DataArray(dA, dims=['nVertLevels', 'nEdges'])
			ds_new["z"] = xr.DataArray(dz[:,idx], dims=['nVertLevels', 'nEdges'])
			ds_new = ds_new.assign_coords(nEdges=("nEdges", edgenums[idx]))
			ds_new.to_netcdf(outfile)

	# plotting
	cmap = plt.get_cmap(cmapname).copy()
	plt.pcolormesh(dx/1000, z, y*dA/1e6, cmap=cmap)
	plt.clim(-0.3, 0.3)
	cbar = plt.colorbar()
	ax = plt.gca()
	# cs = plt.contour(x, z, flux[:,idx], c='k', levels=[35, 35.1, 35.2, 35.3])
	# ax.clabel(cs, cs.levels, fontsize=10)

	# Set the color for NaN values (e.g., 'gray' or 'red')
	cmap.set_bad(color='tab:gray')

	maxdepth = np.argwhere(np.any(bathmask, axis=1))[-1, 0]
	plt.ylim([0, z[maxdepth + 1]])

	ax.invert_yaxis()
	plt.ylabel('Depth (m)')
	cbar.set_label('Transport (Sv)')
	plt.xlabel('Distance along gate (km)')
	plt.show()

if __name__ == '__main__':
	print('1')

	root = 'regional_masks/flux_gates/'
	# fname, poly = 'ar7_approx', False
	fname, poly = 'LabSea_central2', True
	# gate_line = gpd.read_file(root + fname + '.geojson')
	# if not os.path.exists(root + fname + '_edges.json'):
	# 	make_flux_gate_mask(root, fname, edges=True)

	edgeflux_dataset(fname, poly)

	# mesh_gate = json.load(open(root + fname + '.json'))
	# mask = np.array(mesh_gate['mask']).astype(bool)

	# cellnums = np.array(mesh_gate['cellnums'])
	# positive_quadrant = [1,2]
	# par = False

	# mesh = xr.open_dataset(MESHFILE_OCN)
	# z = mpaso_depth(mesh)
	# iz = np.argmax(z>200)
	#
	# edgenums = np.array(mesh_gate['edgenums'])
	# posvec = np.array([-52.257738 + 53.8168085, 56.8198172 - 57.61676])
	# from mesh_tools import sign_flip
	#
	# sign = sign_flip(posvec, edgenums)
	#
	# data = get_mpaso_file_by_date(1950, 1, 'historical0101').isel(Time=0)
	# fluxparam = data['timeMonthly_avg_normalMLEvelocity'][mask,:]
	# fluxparam = fluxparam.isel(nVertLevels=slice(0, iz)).mean(dim='nVertLevels') * sign
	# fluxparam = fluxparam.mean()





	# %% calculate and plot flux
	# runnum = 'historical0101'
	# dates = make_monthly_date_list(dt.datetime(1950,1,1),
	# 							   dt.datetime(1950,3,1))
	# # data = zip_subset_by_time(dates, get_mpaso_file_by_date,
	# # 						  varnames=['vzonal', 'vmeridional'],
	# # 						  # varnames=['sal', 'ocntemp', 'dens'],
	# # 						  runname=runnum)
	# data = get_mpaso_file_by_date(1950, 1, runnum)
	# # plot_TS_diagram(data, runnum, 100, mask)
	# # plot_normal_velocity(dates, runnum)
	# # plot_crosssection(data, runnum, 'sal', mask)
	# plot_normal_velocity(data, gate_line, mask,
	# 					 title=f'GWBC Transport {runnum} ({dates[0].year} - {dates[-1].year})',
	# 					 # saveas=f'figs/flux_gates/{fname}_transport_{runnum}_{dates[0].year}-{dates[-1].year}.png'
	# 					 )
	# plt.show()

	# # flux_ts(data, mask, gate_line)
	# # flux_index_dataset('model_dczone')
	# #
	# # ystep = 10
	# # for runnum in ['historical0101', 'historical0151', 'historical0201', 'historical0251', 'historical0301']:
	# # 	print('\t' + runnum)
	# #
	# # 	for i in range(1950, 2015, ystep):
	# # 		print(i, min(2014, i+ystep))
	# # 		dates = make_monthly_date_list(dt.datetime(i, 1, 1),
	# # 									   dt.datetime(min(2015, i + ystep), 1, 1))
	# #
	# # 		data = zip_subset_by_time(dates, get_mpaso_file_by_date, ['sal', 'ocntemp'], runname=runnum)
	# #
	# # 		plot_normal_velocity(dates, runnum, par=par)
	# # 		plot_crosssection(data, runnum, 'ocntemp', mask)
	# # 		plot_crosssection(data, runnum, 'sal', mask)
	# # 		plot_TS_diagram(data, runnum, mask)
	#


	# %% Create mask and plot fluxgate

	# lat, lon, ncells = mpaso_mesh_latlon()
	# gate_lats = lat[mask]
	# gate_lons = lon[mask]
	#
	# plot_fluxgate(gate_line, gate_lats, gate_lons,
	# 			  parallel=par, posquad=positive_quadrant)
	#
	# plt.show()

