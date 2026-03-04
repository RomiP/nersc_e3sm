import numpy as np
import geopandas as gpd
from helpers import *
from open_e3sm_files import *
from plot_unstructured import *
from cartopy.feature import ShapelyFeature


def plot_climo(runnum='avg'):
	runtype = 'historical'
	varname = 'maxMLD'
	month = [1,2,3]
	ovr_write = False
	if isinstance(month, int):
		climo = get_climatology(varname, month, runtype, overwrite=ovr_write)
		mstr = f'm{month:02}'
	else:
		climo = get_climatology(varname, month[0], runtype, overwrite=ovr_write)
		for m in month[1:]:
			climo[varname].data += get_climatology(varname, m, runtype, overwrite=ovr_write)[varname].data

		climo[varname].data /= len(month)
		mstr = f'{MONTHS[month[0]-1]}-{MONTHS[month[-1]-1]}'


	if varname in ['slp']:
		lat = climo['lat'].data
		lon = climo['lon'].data
		lon[lon>180] -= 360
	else:
		lat, lon, cellnum = mpaso_mesh_latlon()


	if runnum == 'avg':
		climo = climo.mean(dim='runname')
	else:
		climo = climo.sel(runname=runtype + runnum)

	fig, ax = unstructured_pcolor(lat, lon, climo[varname].values,
								  extent=[-70, -30, 50, 70],
								  # extent=[-70, 10, 30, 70],
								  extenttype='tight',
								  cmap='viridis',
								  # clim=[99500, 102500],
								  clim=[0, 2500],
								  # dotsize=5,
								  gridlines=True,
								  landmask=True,
								  interp='grid')
	plt.title(f'{varname} Climatology {mstr} {runtype + runnum}')

	gdf = gpd.read_file('regional_masks/map.geojson')
	# ax.add_geometries([geo_polygon], crs=ccrs.PlateCarree(),
	# 				  facecolor='blue', alpha=0.5, edgecolor='black')
	# 3. Create a ShapelyFeature
	geoms = gdf.geometry.values
	feat = ShapelyFeature(geoms, ccrs.PlateCarree(), edgecolor='red', facecolor='none')

	# 4. Add to cartopy
	ax.add_feature(feat)

	plt.savefig(f'figs/{varname}_climo_{mstr.lower()}_{runtype+runnum}.png')
	plt.show()

def get_extrema_ts(runnum, k, **kwargs):

	if not 'varname' in kwargs:
		varname = 'maxMLD'
	else:
		varname = kwargs['varname']
	if not 'runtype' in kwargs:
		runtype = 'historical'
	else:
		runtype = kwargs['runtype']

	if not 'm_start' in kwargs:
		kwargs['m_start'] = 10
	if not 'm_end' in kwargs:
		kwargs['m_end'] = 3
	if not 'method' in kwargs:
		kwargs['method'] = 'mean'

	f = f'/global/cfs/projectdirs/m1199/romina/data/timeseries/{varname}_ts_{runtype}.nc'
	var_dat = xr.open_dataset(f)

	if runnum == 'avg':
		var_dat = var_dat.mean(dim='runname')
	else:
		var_dat = var_dat.sel(runname=runtype + runnum)

	avg, year = get_seasonal_mean(var_dat[varname].data, var_dat.time.values, **kwargs)
	idx_sorted = np.argsort(avg)

	idx_min = idx_sorted[:k]
	idx_max = idx_sorted[-k:]

	max_vals = {
		'data':avg[idx_max],
		'years':year[idx_max],
		'idx':idx_max
	}

	min_vals = {
		'data':avg[idx_min],
		'years':year[idx_min],
		'idx':idx_min
	}

	return max_vals, min_vals

def get_seasonal_mean(data, dates, m_start, m_end, method='mean', **kwargs):


	y, m, d = dt64_y_m_d(dates)

	years = np.reshape(y, (-1, 12))[:,0]
	if m_start > m_end:
		k = 12 - m_start + 1
		i = k + m_end
		# m = np.roll(m, k)[12:]
		data = np.roll(data, k)[12:]
		# m = np.reshape(m, (-1, 12))
		years = years[1:]
	else:
		i = m_end - m_start + 1

	data = np.reshape(data, (-1, 12))

	if method == 'mean':
		avg = np.nanmean(data[:, :i], axis=1)
	elif method == 'max':
		avg = np.nanmax(data[:, :i], axis=1)
	elif method == 'min':
		avg = np.nanmin(data[:, :i], axis=1)
	else:
		raise ValueError('Method must be one of "mean", "max", or "min"')


	return avg, years

def plot_qnet_extrema_comps(runnum):

	index_var = 'maxMLD'
	runtype = 'historical'

	maxvals, minvals = get_extrema_ts(runnum, 5, varname=index_var, runtype=runtype, m_start=1, m_end=12, method='max')

	print(maxvals)
	print(minvals)


	qnetdat = xr.open_dataset(f'/global/cfs/cdirs/m1199/romina/data/composite_fields/netHeatFlux_{runtype}.nc')
	if runnum == 'avg':
		qnetdat = qnetdat.mean(dim='runname')
	else:
		qnetdat = qnetdat.sel(runname=runtype + runnum)


	y, m, d = dt64_y_m_d(qnetdat.Time.values)


	m_start = 10
	m_end = 3
	idx = None
	for year in minvals['years']:
		if m_start > m_end:
			x = np.argwhere(((y == year-1) & (m>= m_start)) |
							((y == year) & (m<= m_end)))
		else:
			x = np.argwhere((y == year) & ((m>= m_start) & (m<= m_end)) )

		if idx is None:
			idx = np.squeeze(x)
		else:
			idx = np.concatenate((idx, np.squeeze(x)))

	data = qnetdat['netHeatFlux'].isel(Time=idx).mean(dim='Time').values
	lat, lon, ncells = mpaso_mesh_latlon()

	unstructured_pcolor(lat, lon, data,
						extent=[-70, -30, 50, 70],
						clim=[-300, 100],
						cmap='coolwarm',
						norm=0,
						clabel='Net Heat Flux ($W/m^2$)',
						# extenttype='tight',
						gridlines=True,
						interp=False,
						title=runtype + runnum + ' Oct-Mar',
						)
	plt.show()


def plot_maxMLD_extrema(runnum):
	index_var = 'maxMLD'
	runtype = 'historical'
	varname = 'timeMonthlyMax_max_dThreshMLD'

	lat, lon, ncells = mpaso_mesh_latlon()

	maxvals, minvals = get_extrema_ts(runnum, 5, varname=index_var, runtype=runtype, m_start=10, m_end=4, method='max')

	print(maxvals)
	print(minvals)

	mlddat = None
	for year in maxvals['years']:
		for month in [1,2,3]:
			ocn_dat = get_mpaso_file_by_date(year, month, runtype+runnum, 'timeSeriesStatsMonthlyMax')
			if mlddat is None:
				mlddat = ocn_dat[varname].values
			else:
				mlddat += ocn_dat[varname].values


	mlddat /= 15
	unstructured_pcolor(lat, lon, mlddat,
						extent=[-70, -30, 50, 70],
						clim=[100, 3500],
						title=runtype + runnum + ' Oct-Mar')
	plt.show()


if __name__ == '__main__':




	for run in ['0101', '0151', '0201', '0251', '0301', 'avg']:
		print(run)
		# get_extrema_ts(run, 3)
		# plot_qnet_extrema_comps(run)
		# plot_maxMLD_extrema(run)

		plot_climo(run)
		break
