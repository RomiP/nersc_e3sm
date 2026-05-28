import matplotlib.pyplot as plt
import numpy as np
from datashader.colors import viridis

from bootstrap import *
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
	coords = np.array(geoms[0].coords)
	feat = ShapelyFeature(geoms, ccrs.PlateCarree(),
						  edgecolor='red',
						  facecolor='none',
						  linewidth=1,)
	ax.add_feature(feat)

	# plot normal vector
	norm = normal_vector(gate, parallel)
	if posquad is not None:
		quad = quadrant(norm)
		if not quad in posquad:
			norm = - norm
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

def normal_vector(line, parallel=True):
	line = line.geometry.values[0]
	coords = list(line.coords)


	dx = coords[1][0] - coords[0][0]
	dy = coords[1][1] - coords[0][1]

	if not parallel:
		dx, dy = -dy, dx

	vec = np.array([dx, dy])
	vec /= np.linalg.norm(vec)

	return vec


def calculate_flux(data, gateline, mask, scalearea=True, parallel=False, posquad=None):

	u = data[VARNAMES['vzonal']].values[mask,:]
	v = data[VARNAMES['vmeridional']].values[mask,:]


	norm = normal_vector(gateline, parallel)

	if posquad is not None:
		quad = quadrant(norm)
		if not quad in posquad:
			norm = - norm

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


def plot_flux(flux, gateline, mask, cmapname='coolwarm'):

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

	cmap = plt.get_cmap(cmapname).copy()
	flux[~bathmask] = np.nan
	plt.pcolormesh(x, z, flux[:,idx], cmap=cmap)
	cbar = plt.colorbar()


	# Set the color for NaN values (e.g., 'gray' or 'red')
	cmap.set_bad(color='tab:gray')

	maxdepth = np.argwhere(np.any(bathmask, axis=1))[-1, 0]
	# print(z[maxdepth + 1])
	plt.ylim([0, z[maxdepth + 1]])
	ax = plt.gca()
	ax.invert_yaxis()
	plt.ylabel('Depth (m)')
	plt.xlabel('Distance along gate (km)')

	return plt.gcf(), ax, cbar

def plot_normal_velocity(data, runnum, **kwargs):
	# flux = 0
	# for d in tqdm(dates):
	# 	data = get_mpaso_file_by_date(d.year, d.month, runnum,
	# 								  varname='timeSeriesStatsMonthly').isel(Time=0)
	# 	flux += calculate_flux(data, gate_line, mask, scalearea=False,
	# 						   parallel=par, posquad=positive_quadrant)
	#
	#
	#
	# flux /= len(dates)

	flux = data.mean(dim='Time', skipna=True)
	fig, ax, cbar = plot_flux(flux, gate_line, mask)


	cbar.set_label('Along-gate Velocity ($m/s$)')
	# Get the current color limits
	plt.clim(-0.4, 0.4)
	# vmin, vmax = plt.gci().get_clim()
	# cmax = max(abs(vmin), abs(vmax))
	# plt.clim(-cmax, cmax)

	# Add text at ends
	plt.text(1.1, 1.02, "South West", va='bottom', ha='left', transform=ax.transAxes)
	plt.text(1.1, -0.02, "North East", va='top', ha='left', transform=ax.transAxes)

	# plt.text(0.01, 0, "Greenland", va='bottom', ha='left', transform=ax.transAxes)
	# plt.text(0.99, 0, "Iceland", va='bottom', ha='right', transform=ax.transAxes)

	plt.text(0.01, 0, "To Labrador", va='bottom', ha='left', transform=ax.transAxes)
	plt.text(0.99, 0, "To Fram", va='bottom', ha='right', transform=ax.transAxes)

	plt.title(f'Denmark Strait Parallel Velocity ({dates[0].year} - {dates[-1].year})')
	plt.savefig(f'figs/flux_gates/{fname}_parallelvelocity_{runnum}_y{dates[0].year}-{dates[-1].year}.png')
	plt.show()

def plot_crosssection(data, runnum, varname, mask):

	data = data.mean(dim='Time')
	if varname == 'dens':
		flux = rho_e3sm(data, mask).T
	else:
		flux = data[VARNAMES[varname]].values[mask, :].T
	fig, ax, cbar = plot_flux(flux, gate_line, mask, 'viridis')

	if varname == 'sal':
		cbar.set_label('Salinity (PSU)')
		plt.clim(30, 35.5)
		plt.title(f'Denmark Strait Salinity ({dates[0].year} - {dates[-1].year})')

	elif varname == 'ocntemp':
		cbar.set_label('Temperature ($^\circ$C)')
		plt.clim(-2, 8)
		plt.title(f'Denmark Strait Temperature ({dates[0].year} - {dates[-1].year})')

	plt.savefig(f'figs/flux_gates/{fname}_{varname}_{runnum}_y{dates[0].year}-{dates[-1].year}.png')
	plt.show()



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

if __name__ == '__main__':
	print('')

	root = 'regional_masks/flux_gates/'
	fname = 'denmark_strait_sill'
	gate_line = gpd.read_file(root + fname + '.geojson')
	if not os.path.exists(root + fname + '.json'):
		make_flux_gate_mask(root, fname)

	mesh_gate = json.load(open(root + fname + '.json'))
	mask = np.array(mesh_gate['mask']).astype(bool)
	cellnums = np.array(mesh_gate['cellnums'])


	positive_quadrant = [3,4]
	par = True
	# %% calculate and plot flux
	runnum = 'historical0101'
	dates = make_monthly_date_list(dt.datetime(1950,11,1),
								   dt.datetime(1951,1,1))
	# data = zip_subset_by_time(dates, get_mpaso_file_by_date, ['sal', 'ocntemp', 'dens'], runname=runnum)
	data = get_mpaso_file_by_date(1950, 1, runnum)
	plot_TS_diagram(data, runnum, 100, mask)
	# plot_normal_velocity(dates, runnum)
	# plot_crosssection(data, runnum, 'dens', mask)
	# plot_normal_velocity(data, runnum)

	# ystep = 10
	# for runnum in ['historical0101', 'historical0151', 'historical0201', 'historical0251', 'historical0301']:
	# 	print('\t' + runnum)
	#
	# 	for i in range(1950, 2015, ystep):
	# 		print(i, min(2014, i+ystep))
	# 		dates = make_monthly_date_list(dt.datetime(i, 1, 1),
	# 									   dt.datetime(min(2015, i + ystep), 1, 1))
	#
	# 		data = zip_subset_by_time(dates, get_mpaso_file_by_date, ['sal', 'ocntemp'], runname=runnum)
	#
	# 		# plot_normal_velocity(dates, runnum, par=par)
	# 		# plot_crosssection(data, runnum, 'ocntemp', mask)
	# 		# plot_crosssection(data, runnum, 'sal', mask)
	# 		plot_TS_diagram(data, runnum, mask)



	# %% Create mask and plot fluxgate

	# lat, lon, ncells = mpaso_mesh_latlon()
	# gate_lats = lat[mask]
	# gate_lons = lon[mask]
	#
	# plot_fluxgate(gate_line, gate_lats, gate_lons,
	# 			  parallel=par, posquad=positive_quadrant)
	#
	# plt.show()