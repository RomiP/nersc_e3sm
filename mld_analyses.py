import datetime as dt
from helpers import *
import matplotlib
matplotlib.use('module://backend_interagg')
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from open_e3sm_files import *
from tqdm import tqdm
import xarray as xr

def max_mld_percentile_ts(runname, startdate, enddate):
	lat, lon, ncell = mpaso_mesh_latlon()
	mask = get_arctic_ocn_region_mask('Labrador Sea')

	lon = (mask * lon).where(mask, drop=True)
	lat = (mask * lat).where(mask, drop=True)
	ncell = (mask * ncell).where(mask, drop=True)
	ncell = ncell.values.astype(int)

	fileroot = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/E3SM-Arcticv2.1_{runname}/archive/ocn/hist/'
	varname = 'timeMonthlyMax_max_dThreshMLD'

	dates = []
	while startdate < enddate:
		dates.append(startdate)
		startdate += relativedelta(months=1)

	mld = []
	for date in tqdm(dates):
		fname = f'E3SM-Arcticv2.1_{runname}.mpaso.hist.am.timeSeriesStatsMonthlyMax.{date.strftime("%Y-%m-%d")}.nc'
		modeldat = xr.open_dataset(fileroot + fname, engine='netcdf4')
		modeldat = modeldat.sel(nCells=ncell)
		mldmax = np.percentile(modeldat[varname], 95)
		mld.append(mldmax)




	mld = np.array(mld)
	dates = np.array(dates)
	# plt.plot(dates, mld)
	# plt.show()

	return dates, mld


def make_maxMLD_percentile_ts_dataset():
	startdate = dt.datetime(1950, 1, 1)
	enddate = dt.datetime(2015, 1, 1)
	runs = ['historical0101', 'historical0151', 'historical0201', 'historical0251', 'historical0301']

	startdate = dt.datetime(2015, 1, 1)
	enddate = dt.datetime(2096, 1, 1)
	runs = ['ssp370_0201']


	ds = {}
	for runname in runs:
		print(runname)
		dates, mld = max_mld_percentile_ts(runname, startdate, enddate)

		da = xr.DataArray(mld.reshape(-1, 1), dims=('time', 'runname'), coords={'time': dates, 'runname': [runname]},
						  name='maxMLD')
		ds[runname] = da

	da = xr.concat(ds.values(), dim='runname')

	ds = xr.Dataset({'maxMLD': da})
	ds.attrs['units'] = 'm'
	ds.attrs['description'] = 'Labrador Sea region 95th percentile of maxMLD recorded over monthly time period.'

	ds.to_netcdf('/global/cfs/cdirs/m1199/romina/data/maxMLD_ts_forecast.nc', mode='a')

def plot_mld_climo():

	run = 'historical0301'
	tsroot = '/global/cfs/cdirs/m1199/romina/data/'

	files = [
		'maxMLD_ts_historical.nc',
		'maxMLD_ts_forecast.nc'
	]
	yr_range = [1950, 2100]

	# Get a specific colormap, e.g., 'viridis'
	cmap = cm.get_cmap('viridis')
	# need to normalize because color maps are defined in [0, 1]
	norm = colors.Normalize(yr_range[0], yr_range[1])# Get a specific colormap, e.g., 'viridis'
	cmap = cm.get_cmap('viridis')
	# need to normalize because color maps are defined in [0, 1]
	norm = colors.Normalize(yr_range[0], yr_range[-1])

	for f in files:
		dat = xr.open_dataset(tsroot + f)
		# if run in dat.runname:
		# 	dat = dat.sel(runname=run)
		dat = dat.mean(dim='runname')
		years, months, d = dt64_y_m_d(dat.time.values)

		mld = np.reshape(dat['maxMLD'].values, (-1, 12))
		m = np.reshape(months, (-1, 12))

		for year in tqdm(range(years[0], years[-1])):
			c = cmap(norm(year))
			mldts = mld[year - years[0], :]
			plt.plot(rotate(MONTHS, -3), rotate(mldts, -3), c=c, alpha=0.8, lw=1)

	ax = plt.gca()
	ax.invert_yaxis()

	plt.ylabel('Max MLD (m)')
	plt.title(f'Labrador Sea Max MLD (historical avg)')

	# plot colorbar
	plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
	plt.show()


if __name__ == '__main__':
	runnum = 'historical0201'
	# make_maxMLD_percentile_ts_dataset()
	plot_mld_climo()

	# max_mld_percentile_ts(runnum)
