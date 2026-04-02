from climos_and_extrema import get_climatology
import cmasher as cmr
from helpers import *
from open_e3sm_files import *
from plot_unstructured import *
from cartopy.feature import ShapelyFeature


def plot_geospatial_MLD_climo_convexhull():
	runnum = 'avg'
	runtype = 'historical'
	varname = 'maxMLD'
	month = [1, 2, 3]
	ovr_write = False
	if isinstance(month, int):
		climo = get_climatology(varname, month, runtype, overwrite=ovr_write)
		mstr = f'm{month:02}'
	else:
		climo = get_climatology(varname, month[0], runtype, overwrite=ovr_write)
		for m in month[1:]:
			climo[varname].data += get_climatology(varname, m, runtype, overwrite=ovr_write)[varname].data

		climo[varname].data /= len(month)
		mstr = f'{MONTHS[month[0] - 1]}-{MONTHS[month[-1] - 1]}'

	if varname in ['slp']:
		lat = climo['lat'].data
		lon = climo['lon'].data
		lon[lon > 180] -= 360
	else:
		lat, lon, cellnum = mpaso_mesh_latlon()

	if runnum == 'avg':
		climo = climo.mean(dim='runname')
	else:
		climo = climo.sel(runname=runtype + runnum)

	fig, ax = unstructured_pcolor(lat, lon, climo[varname].values,
								  extent=[-70, -30, 50, 70],
								  extenttype='tight',
								  cmap='viridis',
								  clim=[0, 2000],
								  clabel='MDL (m)',
								  gridlines=True,
								  landmask=True,
								  interp='grid')
	plt.title('Winter max MLD from E3SM ensemble mean')

	gdf = gpd.read_file('regional_masks/model_dczone.geojson')
	# 3. Create a ShapelyFeature
	geoms = gdf.geometry.values
	feat = ShapelyFeature(geoms, ccrs.PlateCarree(), edgecolor='red', facecolor='none', linewidth=2)

	# 4. Add to cartopy
	ax.add_feature(feat)

	plt.savefig(f'figs/pubs/{varname}_climo_{mstr.lower()}_{runtype + runnum}.png')
	plt.show()

def plot_E3SM_spatial_resolution():
	pass

def plot_mld_ts(runnum):
	# runnum = '0151'
	tsroot = '/global/cfs/cdirs/m1199/romina/data/timeseries/'

	files = [
		'maxMLD_dcmean_ts_historical.nc',
	]
	yr_range = [1950, 2015]
	ensemble_mean = runnum == 'avg'
	title = f'run number: {runnum}'

	for f in files:
		dat = xr.open_dataset(tsroot + f)
		if ensemble_mean:
			dat = dat.mean(dim='runname')
			title = 'ensemble mean'
		else:
			type = 'historical'
			run = type + runnum

			if run in dat.runname:
				dat = dat.sel(runname=run)
			else:
				continue

		mld_annual = np.reshape(dat['maxMLD'].values, (-1, 12))
		mld_annual = np.max(mld_annual, axis=1)
		years, months, d = dt64_y_m_d(dat.time.values)
		years = np.reshape(years, (-1, 12))
		years = years[:, 0]

		# plt.plot(dat['time'], dat['maxMLD'])
		plt.plot(years, mld_annual)

	plt.xlabel('Year')
	plt.ylabel('Max MLD (m)')
	plt.title(f'Labrador Sea Annual Max MLD ({title})')

	plt.xlim(yr_range[0], yr_range[-1])
	plt.ylim([0,3500])

	ax = plt.gca()
	ax.invert_yaxis()
	plt.ylabel('Max MLD (m)')

	plt.savefig(f'figs/pubs/maxmld_annual_ts_{runnum}.png')
	plt.show()


if __name__ == '__main__':
	# plot_geospatial_MLD_climo_convexhull()
	for run in ['0101', '0151', '0201', '0251', '0301']:
		plot_mld_ts(run)
	# plot_E3SM_spatial_resolution()
