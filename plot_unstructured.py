import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np


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
	fig = plt.figure(dpi=300)

	# set defaults
	if not 'extent' in kwargs:
		extent = [-65, 0, 30, 80]
	else:
		extent = kwargs['extent']
	if not 'cmap' in kwargs:
		kwargs['cmap'] = 'turbo'
	if not 'dotsize' in kwargs:
		kwargs['dotsize'] = 0.25
	if not 'clim' in kwargs:
		kwargs['clim'] = [None, None]
	if not 'landmask' in kwargs:
		kwargs['landmask'] = False

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
		ax.add_feature(cartopy.feature.LAND, edgecolor='black')

	# todo: add option to interp to grid (for nicer publication figs)
	sc = ax.scatter(lon, lat, c=dat,
					s=kwargs['dotsize'], cmap=kwargs['cmap'],
					marker='o', transform=ccrs.PlateCarree())

	cbar = plt.colorbar(sc)
	if 'clabel' in kwargs:
		cbar.set_label(kwargs['clabel'])

	return fig, ax
