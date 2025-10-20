import datetime as dt
import matplotlib
import xarray

matplotlib.use('module://backend_interagg')
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from open_e3sm_files import *
from tqdm import tqdm
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
	mld = []
	while startdate < enddate:
		fname = f'E3SM-Arcticv2.1_{runname}.mpaso.hist.am.timeSeriesStatsMonthlyMax.{startdate.strftime("%Y-%m-%d")}.nc'
		modeldat = xr.open_dataset(fileroot + fname, engine='netcdf4')
		modeldat = modeldat.sel(nCells=ncell)

		mldmax = np.percentile(modeldat[varname], 95)

		dates.append(startdate)
		mld.append(mldmax)

		startdate += relativedelta(months=1)


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
	runs = ['ssp370_0201', '4xCO2_0201']


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

	ds.to_netcdf('/global/cfs/cdirs/m1199/romina/data/maxMLD_ts_forecast.nc')

if __name__ == '__main__':
	runnum = 'historical0201'
	make_maxMLD_percentile_ts_dataset()

	# max_mld_percentile_ts(runnum)



