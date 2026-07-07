from cftime import datetime as cdt
import datetime as dt
from helpers import *
import matplotlib.pyplot as plt
import numpy as np
from dateutil.relativedelta import relativedelta
from open_e3sm_files import *
from tqdm import tqdm

from plot_unstructured import unstructured_pcolor


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

def extract_ts_from_composite_file():
	type = 'historical'
	dat_file = f'/global/cfs/cdirs/m1199/romina/data/composite_fields/netHeatFlux_{type}.nc'

	# dc_mask = xr.open_dataset(f'/global/cfs/projectdirs/m1199/romina/data/misc/maxMLDmask_{type}.nc')
	# mask = dc_mask['maxMLDmask']

	geopoly_file = 'regional_masks/model_dczone.geojson'
	with open(geopoly_file, 'r') as f:
		geopoly = f.read()
	lat, lon, ncells = mpaso_mesh_latlon()
	mask = geopolygon_mask(geopoly, lon, lat)

	# ncells *= mask
	dat = xr.open_dataset(dat_file)
	print('masking')
	# dat = dat.where(dat.nCells * mask, drop=False).mean(dim='nCells')
	dat = dat.where(mask, drop=False).mean(dim='nCells')

	dat.to_netcdf(f'/global/cfs/projectdirs/m1199/romina/data/timeseries/netHeatFlux_LabSeaDC_{type}.nc')


def regionally_averaged_ts(runname, startdate, enddate, fieldnames, mask=None, mesh=None):
	if mesh is None:
		mesh = xr.open_dataset(MESHFILE_OCN)

	if mask is None:
		poly_file = 'regional_masks/model_dczone.geojson'
		my_json = open(poly_file).read()
		lat, lon, _ = mpaso_mesh_latlon(mesh)
		mask = geopolygon_mask(my_json, lon, lat)
	elif isinstance(mask, str) and mask.endswith('.geojson'):
		my_json = open(mask).read()
		lat, lon, _ = mpaso_mesh_latlon(mesh)
		mask = geopolygon_mask(my_json, lon, lat)
	elif isinstance(mask, str) and mask.endswith('.json'):
		mask = json.load(open(mask))
		mask = np.array(mask['mask']).astype(bool)
	imask = np.argwhere(mask).squeeze()


	mesh = mesh.isel(Time=0, nCells=imask)
	z = mpaso_depth(mesh)
	dz_mask = mesh['layerThickness'].values > 0
	izmax = np.argwhere(np.isnan(z)).squeeze()[0]
	dz_mask = dz_mask[:, :izmax]

	dates = []
	while startdate < enddate:

		if startdate.year < 1000:
			d = cdt(startdate.year, startdate.month, startdate.day, calendar='noleap')
			dates.append(d)
		else:
			dates.append(startdate)
		startdate += relativedelta(months=1)


	regional_dat = {f:[] for f in fieldnames}

	for date in tqdm(dates):
		# fname = f'E3SM-Arcticv2.1_{runname}.mpaso.hist.am.timeSeriesStatsMonthlyMax.{date.strftime("%Y-%m-%d")}.nc'
		# modeldat = xr.open_dataset(fileroot + fname, engine='netcdf4')
		if runname == 'control':
			raise NotImplementedError
			fname = (get_e3sm_run_path(runname, 'ocn')[0] +
					 '../singleVarFiles/maxMLD/'
					 f'dThreshMLD.E3SMv2.1B60to10rA02.mpaso.hist.am.timeSeriesStatsMonthlyMax.{date.year:04}-{date.month:02}-01.nc')
			modeldat = xr.open_dataset(fname)
		else:
			if fieldnames[0] in COMPONENTS['ocn']:
				modeldat = get_mpaso_file_by_date(date.year, date.month, runname)
			elif fieldnames[0] in COMPONENTS['ice']:
				modeldat = get_mpassi_file_by_date(date.year, date.month, runname)
			elif fieldnames[0] in COMPONENTS['atm']:
				modeldat = get_atmo_file_by_date(date.year, date.month, runname)
			else:
				raise NotImplementedError

		modeldat = modeldat.isel(nCells=imask, Time=0)
		modeldat = modeldat.sel(nVertLevels=slice(0, izmax))

		for field in fieldnames:
			varname = VARNAMES[field]
			singlevar = modeldat[varname]

			if singlevar.ndim == 2:
				singlevar.data[~dz_mask] = np.nan

			regional_dat[field].append(singlevar.mean(dim='nCells', skipna=True))

		# modeldat = modeldat.sel(nCells=ncell)
	# 	modeldat = modeldat[varname].data.squeeze()[mask]
	# 	ts.append(np.mean(modeldat))
	#
	#
	# ts = np.array(ts)
	# dates = np.array(dates)
	#
	# return dates, ts

	for field in fieldnames:
		regional_dat[field] = xr.concat(regional_dat[field], dim='Time')
		regional_dat[field].attrs = modeldat[VARNAMES[field]].attrs
		if 'nVertLevels' in regional_dat[field].dims:
			regional_dat[field] = regional_dat[field].rename({'nVertLevels':'depth'})
			regional_dat[field] = regional_dat[field].assign_coords(depth=z[:izmax])

	return regional_dat

def make_regionalavg_ts_dataset(fieldname, units):
	# startdate = dt.datetime(1, 1, 1)
	# enddate = dt.datetime(387, 1, 1)
	# runs = ['control']

	startdate = dt.datetime(1950, 1, 1)
	enddate = dt.datetime(2015, 1, 1)
	runs = ['historical0101', 'historical0151', 'historical0201', 'historical0251', 'historical0301']

	# startdate = dt.datetime(2015, 1, 1)
	# enddate = dt.datetime(2097, 1, 1)
	# runs = ['ssp370_0101', 'ssp370_0201']

	outfile = f'/global/cfs/cdirs/m1199/romina/data/timeseries/{fieldname}_dcmean_ts_historical.nc'
	# ds = {}
	for runname in runs:
		print(runname)
		dates, field = regionally_averaged_ts(runname, startdate, enddate, fieldname)
		# dates2, field2 = regionally_averaged_ts(runname, startdate, enddate, fieldname+'u')
		# field+=field2

		da_new = xr.DataArray(field.reshape(-1, 1),
						  dims=('Time', 'runname'),
						  coords={'Time': dates, 'runname': [runname]},
						  name=fieldname)

		ds_new = xr.Dataset({fieldname: da_new})

		# --- Save / Append logic ---
		if os.path.exists(outfile):
			print("Appending to existing file...")

			with xr.open_dataset(outfile) as ds_existing:
				ds_combined = xr.concat([ds_existing, ds_new], dim='runname')

				# Combine along runname dimension
				ds_combined = xr.concat([ds_existing, ds_new], dim='runname')

				# Optional: remove duplicate runnames if rerunning
				_, index = np.unique(ds_combined['runname'], return_index=True)
				ds_combined = ds_combined.isel(runname=index)

			ds_combined.to_netcdf(outfile, mode='w')

		else:
			print("Creating new file...")
			ds_new.attrs['units'] = units
			ds_new.attrs['description'] = f'Labrador Sea {fieldname} mean over deep convection zone recorded over monthly time period.'
			ds_new.to_netcdf(outfile)


def regional_avg_dataset(fieldnames, region):


	startdate = dt.datetime(1950, 1, 1)
	enddate = dt.datetime(2015, 1, 1)
	runs = ['historical0101', 'historical0151', 'historical0201', 'historical0251', 'historical0301']
	mesh = xr.open_dataset(MESHFILE_OCN)
	for run in runs[2:]:
		print(run)
		all_fields = regionally_averaged_ts(run, startdate, enddate, fieldnames, mask=region, mesh=mesh)

		for field in fieldnames:
			outfile = f'/global/cfs/cdirs/m1199/romina/data/timeseries/{field}_{region.split("/")[-1].split(".")[0]}_ts_historical.nc'
			ds_new = xr.Dataset({field: all_fields[field]}).assign_coords({'runname':[run]})


			# --- Save / Append logic ---
			if os.path.exists(outfile):
				print("Appending to existing file...")

				with xr.open_dataset(outfile) as ds_existing:
					# ds_combined = xr.concat([ds_existing, ds_new], dim='runname')

					# Combine along runname dimension
					ds_combined = xr.concat([ds_existing, ds_new], dim='runname')

					# Optional: remove duplicate runnames if rerunning
					_, index = np.unique(ds_combined['runname'], return_index=True)
					ds_combined = ds_combined.isel(runname=index)

				ds_combined.to_netcdf(outfile, mode='w')

			else:
				print("Creating new file...")
				# ds_new.attrs['units'] = units
				# ds_new.attrs[
				# 	'description'] = f'Labrador Sea {fieldname} mean over deep convection zone recorded over monthly time period.'
				ds_new.to_netcdf(outfile)








def make_CDT_prof_dataset(runnum):

	lat, lon, ncell = mpaso_mesh_latlon()
	mask = geopolygon_mask('regional_masks/model_dczone.geojson', lon, lat)

	runnum = 'historical' + runnum
	startdate = dt.datetime(1950, 1, 1)
	enddate = dt.datetime(2015, 1, 1)
	dates = make_monthly_date_list(startdate, enddate)

	prof = []
	for d in tqdm(dates):
		ds = get_mpaso_file_by_date(d.year, d.month, runnum)
		ctd = ds[['timeMonthly_avg_density', 'timeMonthly_avg_activeTracers_salinity',
				  'timeMonthly_avg_activeTracers_temperature']]
		p = ctd.where(ctd.nCells * mask).mean(dim='nCells')
		prof.append(p)
		# if prof is None:
		# 	prof = p
		# else:
		# 	prof = xr.concat([prof, p], dim='Time')

	print('concatenating')
	prof = xr.concat(prof, dim='Time')
	prof['z'] = ('nVertLevels', mpaso_depth())
	prof['z'].attrs['units'] = 'm'
	prof.to_netcdf(f'/global/cfs/projectdirs/m1199/romina/data/profiles/LabSea_ctd_dczone_{runnum}.nc')
	print()




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

	# startdate = dt.datetime(2015, 1, 1)
	# enddate = dt.datetime(2097, 1, 1)
	# type = 'forecast'
	# runs = ['ssp370_0101', 'ssp370_0201']

	# %% Do stuff

	# make_qnet_field(startdate, enddate, runs, type)

	# extract_ts_from_composite_file()
	print('1')
	# make_regionalavg_ts_dataset('sal', 'PSU')
	regional_avg_dataset(['sal', 'ocntemp', 'pdens', 'brn'], 'regional_masks/model_dczone.geojson')
	# for run in ['0101', '0151', '0251', '0301']:
	# 	make_CDT_prof_dataset(run)








