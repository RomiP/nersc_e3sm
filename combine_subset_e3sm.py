from cftime import datetime as cdt
import datetime as dt
from helpers import *
import matplotlib.pyplot as plt
import numpy as np
from dateutil.relativedelta import relativedelta
from open_e3sm_files import *
from tqdm import tqdm

def make_qnet_field(startdate, enddate, runs, type):

	# units = W/m^2, positive into ocean
	dates = make_monthly_date_list(startdate, enddate)
	# ocn_dat  = get_mpaso_file_by_date(1950, 1, runs[0])
	prefix = 'timeMonthly_avg_'
	heat_varnames = ['latentHeatFlux', 'longWaveHeatFluxDown', 'longWaveHeatFluxUp', 'sensibleHeatFlux', 'shortWaveHeatFlux']
	heat_varnames = [prefix + i for i in heat_varnames]
	print()

	def _sum_vars(ds, vars, newname=None):
		mysum = ds[vars[0]]
		for var in vars[1:]:
			mysum.data += ds[var].data

		return mysum.rename(newname)

	ds = {}
	for runname in runs:
		print(runname)
		da = None
		for d in tqdm(dates):
			if runname == 'control':
				# todo: deal with single var files
				p = get_e3sm_run_path(runname) + '../'
			ocn_dat = get_mpaso_file_by_date(d.year, d.month, runname)
			if da is None:
				da = _sum_vars(ocn_dat, heat_varnames, newname='netHeatFlux')
			else:
				da = xr.concat([da, _sum_vars(ocn_dat, heat_varnames, newname='netHeatFlux')], dim='Time')

		da['runname'] = runname
		ds[runname] = da

	da = xr.concat(ds.values(), dim='runname')
	ds = xr.Dataset({'netHeatFlux': da})
	ds.attrs['units'] = 'W/m^2'
	ds.attrs['description'] = 'Net heat flux into the ocean'
	ds.attrs['Summed variables'] = heat_varnames

	ds.to_netcdf(f'/global/cfs/cdirs/m1199/romina/data/netHeatFlux_{type}.nc')

if __name__ == '__main__':
	# %% date ranges and params

	# startdate = dt.datetime(1, 1, 1)
	# enddate = dt.datetime(387, 1, 1)
	# type = 'control'
	# runs = ['control']

	# startdate = dt.datetime(1950, 1, 1)
	# enddate = dt.datetime(2015, 1, 1)
	# type = 'historical'
	# runs = ['historical0101', 'historical0151', 'historical0201', 'historical0251', 'historical0301']

	startdate = dt.datetime(2015, 1, 1)
	enddate = dt.datetime(2097, 1, 1)
	type = 'forecast'
	runs = ['ssp370_0101', 'ssp370_0201']

	# %% Do stuff

	make_qnet_field(startdate, enddate, runs, type)

