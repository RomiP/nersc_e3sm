from cftime import datetime as cdt
import datetime as dt
from helpers import *
import matplotlib.pyplot as plt
import numpy as np
from dateutil.relativedelta import relativedelta
from open_e3sm_files import *
from tqdm import tqdm

def produce_NAO_ts(runname, startdate, enddate):
	# Icelow (Akureyri)
	lat_ice = 65.7
	lon_ice = -18.1

	# Azohigh (Ponta Delgada)
	lat_azo = 37.7
	lon_azo = -25.7

	dates = []
	while startdate < enddate:

		if startdate.year < 1000:
			d = cdt(startdate.year, startdate.month, startdate.day, calendar='noleap')
			dates.append(d)
		else:
			dates.append(startdate)
		startdate += relativedelta(months=1)


	il = []
	ah = []
	for date in tqdm(dates):
		atmodata = get_atmo_file_by_date(date.year, date.month, runname)
		lat = atmodata.lat.values
		lon = atmodata.lon.values

		idx_il = get_nearest_coord_idx(lon, lat, [lon_ice, lat_ice])
		slp = atmodata.PSL.isel(ncol=idx_il)
		il.append(slp.values)

		idx_ah = get_nearest_coord_idx(lon, lat, [lon_azo, lat_azo])
		slp = atmodata.PSL.isel(ncol=idx_ah)
		ah.append(slp.values)

	il = np.array(il)
	ah = np.array(ah)

	nao = ah - il
	nao = (nao - np.mean(nao)) / np.std(nao)

	# plt.plot(dates, nao)
	# plt.show()

	return il, ah, nao, dates

def make_nao_dataset():
	# startdate = dt.datetime(1, 1, 1)
	# enddate = dt.datetime(387, 1, 1)
	# runs = ['control']

	# startdate = dt.datetime(1950, 1, 1)
	# enddate = dt.datetime(2015, 1, 1)
	# runs = ['historical0101', 'historical0151', 'historical0201', 'historical0251', 'historical0301']

	startdate = dt.datetime(2015, 1, 1)
	enddate = dt.datetime(2097, 1, 1)
	runs = ['ssp370_0101', 'ssp370_0201']

	ds_runs = []
	for runname in runs:
		print(runname)
		il, ah, nao, dates = produce_NAO_ts(runname, startdate, enddate)

		da_il = xr.DataArray(il.reshape(-1, 1), dims=('time', 'runname'), coords={'time': dates, 'runname': [runname]},
						  name='icelo')
		da_ah = xr.DataArray(ah.reshape(-1, 1), dims=('time', 'runname'), coords={'time': dates, 'runname': [runname]},
						  name='azohi')
		da_nao = xr.DataArray(nao.reshape(-1, 1), dims=('time', 'runname'), coords={'time': dates, 'runname': [runname]},
						  name='nao')
		ds = xr.Dataset(
			{'icelo': da_il, 'azohi': da_ah, 'nao': da_nao},
		)
		ds_runs.append(ds)

	ds = xr.concat(ds_runs, dim='runname')

	# ds = xr.Dataset({'maxMLD': da})
	ds.icelo.attrs['units'] = 'Pa'
	ds.icelo.attrs['description'] = 'Icelandic low. Sea Level Pressure at Akureyri. coords: [65.7, -18.1]'

	ds.azohi.attrs['units'] = 'Pa'
	ds.azohi.attrs['description'] = 'Azores High. Sea Level Pressure at Ponta Delgada. coords: [37.7, -25.7]'

	ds.nao.attrs['units'] = 'unitless'
	ds.nao.attrs['description'] = ('Monthly NAO index calculated from the normalized SLP difference between the azores'
								   ' high and the icelandic low.')

	ds.to_netcdf('/global/cfs/cdirs/m1199/romina/data/nao_forecast.nc', mode='a')

def ts_seasonal_avg(ts, time, monthrange):
	'''

	:param ts: Monthly time series
	:param time: list of numpy datetime64 objects
	:param monthrange: [month_start, month_end] as integers (i.e. 1 = Jan)
	:return: seasonally averaged time series, years
	'''

	y, m, d = dt64_y_m_d(time)
	# m = np.reshape(m, (-1, 12))

	if monthrange[0] > monthrange[1]:
		k1 = monthrange[0] - 1
		k2 = 12 - k1 + monthrange[1]
		# m = np.roll(m, -k1, axis=1)
		y = np.roll(y, -k1, axis=0)
		y = y[:-12]

		ts = np.roll(ts, -k1, axis=0)
		ts = ts[:-12]

		k1 = 0

	else:
		k1 = monthrange[0] - 1
		k2 = monthrange[1]

	y = np.reshape(y, (-1, 12))
	ts = np.reshape(ts, (-1, 12))

	years = y[:,-1]
	seasonal = np.nanmean(ts[:, k1:k2], axis=1)

	return seasonal, years


def plot_nao_ts():
	type = 'control'
	run = ''
	monthrange = [10, 3]

	root = '/global/cfs/cdirs/m1199/romina/data/'
	naofile = f'nao_{type}.nc'

	naodata = xr.open_dataset(root + naofile)

	if 'avg' in run:
		naodata = naodata.mean(dim='runname')
	else:
		naodata = naodata.sel(runname=type+run)

	# nao = naodata['nao'].values
	# nao, years = ts_seasonal_avg(nao, naodata.time.values, monthrange)

	il = naodata['icelo'].values
	il, years = ts_seasonal_avg(il, naodata.time.values, monthrange)

	ah = naodata['azohi'].values
	ah, years = ts_seasonal_avg(ah, naodata.time.values, monthrange)

	nao = normalize(ah - il)

	sigma2 = np.nanstd(nao)

	plt.plot(years, nao, linewidth=1)
	# plt.scatter(years, nao)

	plt.hlines(0, min(years), max(years), color='tab:red')
	plt.hlines([sigma2, -sigma2], min(years), max(years), color='tab:grey')
	plt.hlines([2*sigma2, -2*sigma2], min(years), max(years), color='tab:grey', linestyle='--')
	plt.xlabel('Year')
	plt.ylabel('NAO Index')
	plt.title('Seasonal Average (Oct-Mar) NAO Index')
	plt.show()

if __name__ == '__main__':
	# pass
	# runname = 'historical0101'
	# startdate = dt.datetime(1950, 1, 1)
	# enddate = dt.datetime(2015, 1 ,1)
	# produce_NAO_ts(runname, startdate, enddate)

	make_nao_dataset()
	# plot_nao_ts()

