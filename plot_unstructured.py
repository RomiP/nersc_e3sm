import cartopy
import cartopy.crs as ccrs
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.mpl.ticker as ctk
import numpy as np
from scipy.interpolate import griddata


def _init_proj(proj_name, **kwargs):
	if proj_name == 'LambertConformal':
		extent = kwargs['extent']
		thirds = (extent[3] - extent[2]) / 3.
		return ccrs.LambertConformal(
			central_longitude=np.mean(extent[:2]),
			standard_parallels=(extent[2] + thirds, extent[3] - thirds)
		)
	else:
		raise NotImplementedError


def unstructured_pcolor(lat, lon, dat, **kwargs):
	'''
	Create pseudocolour plot from unstructured data either
	by coloured scatter plot points or interpolating to a
	regular grid.

	:param lat: latitudes [-180, 180]
	:param lon: longitudes [-90, 90]
	:param dat: scalar data to colour
	:param kwargs:
		extent = [longitude_min, longitude_max, latitude_min, latitude_max]
			extenttype = 'tight' or 'rect'
		cmap = str: name of colourmap
		clim = colourmap limits [smallest, largest]
		clabel = str: colourbar label
		landmask = boolean: filled land mask, else just coastlines
		interp = boolean: interp to regular grid, scatter plot if false
			dotsize = size of scatter plot points
			res = grid resolution (deg) for interp to regular grid
		projname = str: name of map projection
	:return: figure, axes
	'''
	fig = plt.figure(dpi=300)

	# set defaults
	if not 'extent' in kwargs:
		extent = [-65, 0, 30, 80]
	else:
		extent = kwargs['extent']
	if not 'extenttype' in kwargs:
		kwargs['extenttype'] = 'rect'
	if not 'cmap' in kwargs:
		kwargs['cmap'] = 'turbo'
	if not 'dotsize' in kwargs:
		kwargs['dotsize'] = 0.25
	if not 'clim' in kwargs:
		kwargs['clim'] = [None, None]
	if not 'landmask' in kwargs:
		kwargs['landmask'] = False
	if not 'interp' in kwargs:
		kwargs['interp'] = False

	# init axes with geo proj
	projname = 'LambertConformal'
	if 'projname' in kwargs:
		projname = kwargs['projname']
	my_proj = _init_proj(projname, extent=extent)
	ax = fig.add_subplot(1, 1, 1, projection=my_proj)

	if kwargs['landmask']:
		ax.add_feature(cartopy.feature.NaturalEarthFeature(
			'physical', 'land', '110m', facecolor='grey'), zorder=99)
	else:
		ax.coastlines()

	if kwargs['interp']:

		step = 0.1
		if 'res' in kwargs:
			step = kwargs['res']
		X = np.arange(extent[0], extent[1], step) # lons
		X[X<0] += 360
		Y = np.arange(extent[2], extent[3], step) # lats
		X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
		coords = np.array([lon, lat]).T
		Z = griddata(coords, np.squeeze(dat), (X, Y), method='nearest')
		sc = ax.pcolormesh(X, Y, Z, cmap=kwargs['cmap'], transform=ccrs.PlateCarree())


	else:
		sc = ax.scatter(lon, lat, c=dat,
						s=kwargs['dotsize'], cmap=kwargs['cmap'],
						marker='o', transform=ccrs.PlateCarree())


	if kwargs['extenttype'] == 'tight':
		xlim = [min(extent[:2]), max(extent[:2])]
		ylim = [min(extent[2:]), max(extent[2:])]

		rect = mpath.Path([[xlim[0], ylim[0]],
						   [xlim[1], ylim[0]],
						   [xlim[1], ylim[1]],
						   [xlim[0], ylim[1]],
						   [xlim[0], ylim[0]],
						   ]).interpolated(20)

		proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
		rect_in_target = proj_to_data.transform_path(rect)

		ax.set_boundary(rect_in_target)

	else:

		# Set the extent of the map (longitude_min, longitude_max, latitude_min, latitude_max)
		ax.set_extent(extent, crs=ccrs.PlateCarree())



	cbar = plt.colorbar(sc)
	if 'clabel' in kwargs:
		cbar.set_label(kwargs['clabel'])


	if 'gridlines' in kwargs and kwargs['gridlines']:
		gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linestyle='dashed')
		gl.top_labels = False
		gl.right_labels = False
		gl.rotate_labels = False
		gl.xlocator = ctk.LongitudeLocator(4)
		gl.ylocator = ctk.LatitudeLocator(6)
		gl.xformatter = ctk.LongitudeFormatter(zero_direction_label=True)
		gl.yformatter = ctk.LatitudeFormatter()

	return fig, ax
