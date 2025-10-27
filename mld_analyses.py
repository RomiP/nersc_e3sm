from atmo_analyses import ts_seasonal_avg
from cftime import datetime as cdt
import datetime as dt
from helpers import *
import matplotlib
matplotlib.use('module://backend_interagg')
import matplotlib.pyplot as plt
import numpy as np
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

		if startdate.year < 1000:
			d = cdt(startdate.year, startdate.month, startdate.day, calendar='noleap')
			dates.append(d)
		else:
			dates.append(startdate)
		startdate += relativedelta(months=1)


	mld = []
	for date in tqdm(dates):
		# fname = f'E3SM-Arcticv2.1_{runname}.mpaso.hist.am.timeSeriesStatsMonthlyMax.{date.strftime("%Y-%m-%d")}.nc'
		# modeldat = xr.open_dataset(fileroot + fname, engine='netcdf4')
		if runname == 'control':
			fname = (get_e3sm_run_path(runname, 'ocn')[0] +
					 '../singleVarFiles/maxMLD/'
					 f'dThreshMLD.E3SMv2.1B60to10rA02.mpaso.hist.am.timeSeriesStatsMonthlyMax.{date.year:04}-{date.month:02}-01.nc')
			modeldat = xr.open_dataset(fname)
		else:
			modeldat = get_mpaso_file_by_date(date.year, date.month, runname, 'timeSeriesStatsMonthlyMax')

		modeldat = modeldat.sel(nCells=ncell)
		mldmax = np.percentile(modeldat[varname], 95)
		mld.append(mldmax)




	mld = np.array(mld)
	dates = np.array(dates)
	# plt.plot(dates, mld)
	# plt.show()

	return dates, mld


def make_maxMLD_percentile_ts_dataset():
	startdate = dt.datetime(1, 1, 1)
	enddate = dt.datetime(387, 1, 1)
	runs = ['control']

	# startdate = dt.datetime(1950, 1, 1)
	# enddate = dt.datetime(2015, 1, 1)
	# runs = ['historical0101', 'historical0151', 'historical0201', 'historical0251', 'historical0301']

	# startdate = dt.datetime(2015, 1, 1)
	# enddate = dt.datetime(2096, 1, 1)
	# runs = ['ssp370_0201']


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

	ds.to_netcdf('/global/cfs/cdirs/m1199/romina/data/maxMLD_ts_control.nc', mode='a')

def plot_mld_climo():

	run = 'historical0301'
	tsroot = '/global/cfs/cdirs/m1199/romina/data/'

	files = [
		'maxMLD_ts_historical.nc',
		# 'maxMLD_ts_forecast.nc'
	]
	yr_range = [1950, 2100]

	# Get a specific colormap, e.g., 'viridis'
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


def plot_mld_ts():
	run = 'control'
	# run = 'avg'
	tsroot = '/global/cfs/cdirs/m1199/romina/data/'

	# files = [
	# 	'maxMLD_ts_historical.nc',
	# 	'maxMLD_ts_forecast.nc'
	# ]
	# yr_range = [1950, 2100]

	files = ['maxMLD_ts_control.nc']
	yr_range = [1, 2]

	# Get a specific colormap, e.g., 'viridis'
	cmap = cm.get_cmap('viridis')
	# need to normalize because color maps are defined in [0, 1]
	norm = colors.Normalize(yr_range[0], yr_range[-1])

	for f in files:
		dat = xr.open_dataset(tsroot + f)
		if run in dat.runname:
			dat = dat.sel(runname=run)
		else:
			dat = dat.mean(dim='runname')

		mld_annual = np.reshape(dat['maxMLD'].values, (-1, 12))
		mld_annual = np.max(mld_annual, axis=1)
		years, months, d = dt64_y_m_d(dat.time.values)
		years = np.reshape(years, (-1, 12))
		years = years[:, 0]

		# plt.plot(dat['time'], dat['maxMLD'])
		plt.plot(years, mld_annual)


	plt.xlabel('Date')
	plt.ylabel('Max MLD (m)')
	plt.title(f'Labrador Sea Annual Max MLD ({run})')

	ax = plt.gca()
	ax.invert_yaxis()

	plt.savefig('figs/maxmld_annual_ts_control.png')
	plt.show()

def plot_mld_vs_nao():
	run = 'historical0301'
	monthrange = [10, 3]

	root = '/global/cfs/cdirs/m1199/romina/data/'
	mldfile = 'maxMLD_ts_historical.nc'
	naofile = 'nao_historical.nc'

	mlddata = xr.open_dataset(root + mldfile)
	naodata = xr.open_dataset(root + naofile)

	if 'avg' in run:
		mlddata = mlddata.mean(dim='runname')
		naodata = naodata.mean(dim='runname')
	else:
		naodata = naodata.sel(runname=run)
		mlddata = mlddata.sel(runname=run)

	# nao = naodata['nao'].values
	# nao, years = ts_seasonal_avg(nao, naodata.time.values, monthrange)

	il = naodata['icelo'].values
	il, years = ts_seasonal_avg(il, naodata.time.values, monthrange)

	ah = naodata['azohi'].values
	ah, years = ts_seasonal_avg(ah, naodata.time.values, monthrange)

	nao = normalize(ah - il)

	mld = mlddata['maxMLD'].values
	mld = mld.reshape(-1, 12)
	mld = np.max(mld, axis=1)[:-1]

	plt.scatter(nao, mld)
	plt.xlabel('NAO index')
	plt.ylabel('Max MLD (m)')
	plt.title(f'E3SM Mean Winter NAO Effect on Max MLD ({run})')

	# plt.savefig(f'figs/nao_vs_maxMLD_{run}.png')

	plt.show()





if __name__ == '__main__':
	runnum = 'historical0201'
	# max_mld_percentile_ts(runnum)
	# make_maxMLD_percentile_ts_dataset()

	# plot_mld_climo()
	plot_mld_ts()
	# plot_mld_vs_nao()


