import bokeh.palettes
import cartopy
import cartopy.crs as ccrs
import cartopy.mpl.ticker as ctk
import holoviews as hv
import matplotlib.colors as cm
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.ticker import FixedLocator
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import mosaic
import numpy as np
from open_e3sm_files import MESHFILE_OCN
from scipy.interpolate import griddata
import xarray as xr


class MidpointNormalize(cm.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		cm.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))


def _init_proj(proj_name, **kwargs):
	if proj_name == 'LambertConformal':
		extent = kwargs['extent']
		thirds = (extent[3] - extent[2]) / 3.
		projection = ccrs.LambertConformal(
			central_longitude=np.mean(extent[:2]),
			standard_parallels=(extent[2] + thirds, extent[3] - thirds)
		)
	elif proj_name == 'NorthPolarStereo':
		projection = ccrs.NorthPolarStereo()
	elif proj_name == 'SouthPolarStereo':
		projection = ccrs.SouthPolarStereo()
	elif proj_name == 'Miller':
		projection = ccrs.Miller()
	else:
		projection = ccrs.Robinson()

	return projection




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
	if not 'norm' in kwargs:
		kwargs['norm'] = None
	else:
		norm = kwargs['norm']
		vmin, vmax = kwargs['clim']
		kwargs['norm'] = None
		# kwargs['norm'] = MidpointNormalize(midpoint=kwargs['norm'],
		# 								   vmin=vmin, vmax=vmax)

		if vmax is not None and vmin is not None:
			rng = max(abs(vmax), abs(vmin))
			norm = cm.Normalize(vmin=-rng + norm, vmax=rng + norm)

			# Truncate colormap
			cmap_full = plt.colormaps[kwargs['cmap']]
			cmap_trunc = ListedColormap(cmap_full(np.linspace(norm(vmin), norm(vmax), 256)))
			kwargs['cmap'] = cmap_trunc
	#
	# kwargs['norm'] = TwoSlopeNorm(vcenter=kwargs['norm'],
	# 							  vmin=kwargs['clim'][0],
	# 							  vmax=kwargs['clim'][1])

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

	if kwargs['interp']=='grid':

		step = 0.1
		if 'res' in kwargs:
			step = kwargs['res']
		X = np.arange(extent[0], extent[1], step)  # lons
		# X[X < 0] += 360
		Y = np.arange(extent[2], extent[3], step)  # lats
		X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
		coords = np.array([lon, lat]).T
		Z = griddata(coords, np.squeeze(dat), (X, Y), method='nearest')
		colplot = ax.pcolormesh(X, Y, Z, cmap=kwargs['cmap'], clim=kwargs['clim'],
						   norm=kwargs['norm'],
						   transform=ccrs.PlateCarree())
	elif kwargs['interp']=='mosaic':
		# todo: this expects dat to be dataArray with
		if not 'dsMesh' in kwargs:
			kwargs['dsMesh'] = xr.open_dataset(MESHFILE_OCN)
			kwargs['dsMesh'].attrs['is_periodic'] = 'NO'
		descriptor = mosaic.Descriptor(kwargs['dsMesh'], my_proj, ccrs.Geodetic(), use_latlon=True)

		mosaic_arg = {}
		if 'mosaic_edges' in kwargs and kwargs['mosaic_edges']:
			mosaic_arg['edgecolors'] = 'grey'
			mosaic_arg['antialiaseds'] = True
		else:
			mosaic_arg['antialiaseds'] = False

		colplot = mosaic.polypcolor(ax, descriptor, dat, cmap=kwargs['cmap'], norm=kwargs['norm'], **mosaic_arg)

	else:
		colplot = ax.scatter(lon, lat, c=dat,
						s=kwargs['dotsize'], cmap=kwargs['cmap'], clim=kwargs['clim'],
						norm=kwargs['norm'],
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

	cbar = plt.colorbar(colplot)
	if 'clabel' in kwargs:
		cbar.set_label(kwargs['clabel'])

	if 'title' in kwargs:
		plt.title(kwargs['title'])

	# if kwargs['norm'] is not None:
	# 	vmin, vmax = sc.get_clim()
	# 	# Proportional ticks(linear in data	space)
	# 	# ticks = np.linspace(vmin, vmax, 6)
	# 	ticks = np.concat([np.linspace(vmin, 0, 6), np.linspace(0, vmax, 2)])
	# 	print(ticks)
	# 	cbar.set_ticks(ticks)
	# 	cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))

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
