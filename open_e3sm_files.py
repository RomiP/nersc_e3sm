from cftime import datetime	as cdt
from helpers import MONTHS
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import xarray as xr

MESHFILE_OCN = ('/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/'
				'mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc')

E3SM_SIM_PATH = '/global/cfs/cdirs/m1199/e3sm-arrm-simulations/'

def get_arctic_ocn_region_mask(regions):
	'''
	Load cell mask for union of Arctic Ocean regions.
	Valid region names
	- Baffin Bay
	- Baltic Sea
	- Barents Sea
	- Canada Basin
	- Canadian Archipelago
	- Central Arctic
	- Chukchi Sea
	- East Siberian Sea
	- Greenland Sea
	- Hudson Bay
	- Irminger Sea
	- Kara Sea
	- Labrador Sea
	- Laptev Sea
	- North Sea
	- Norwegian Sea
	- Beaufort Gyre
	- Beaufort Gyre Shelf
	- Arctic Ocean (no Barents/Kara Seas)

	:param regions: Name of Arctic ocean region(s) [str or lst]
	:return: cell mask
	'''
	regionMaskDir = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arcticRegions.nc'
	if not isinstance(regions, list):
		regions = [regions]

	dsRegionMask = xr.open_dataset(regionMaskDir)
	all_regions = dsRegionMask.regionNames.values.astype(str)

	cellMask = None
	for reg in regions:
		# print(reg)
		# regionIndex = all_regions.index(reg)
		i = np.argwhere(all_regions == reg)
		dsMask = dsRegionMask.isel(nRegions=int(i))

		if cellMask is not None:
			cellMask = np.logical_or(cellMask, dsMask.regionCellMasks == 1)
		else:
			cellMask = dsMask.regionCellMasks == 1

	return cellMask

def get_nearest_coord_idx(lons, lats, targetcoord):
	'''
	determine the index of the nearest point to a target from an
	unstructured list of coodinate pairs.
	:param lats: iterable of latitudes
	:param lons: iterable of longitudes
	:param targetcoord: [lon, lat]
	:return:  index of  [lon, lat] coordinate pair nearest the target coordinate

	>>> lons = np.linspace(-20, 0, 20)
	>>> lats = np.linspace(30, 80, 20)
	>>> get_nearest_coord_idx(lons, lats, [-19.8, 30.2])
	0
	>>> get_nearest_coord_idx(lons, lats, [-7, 61])
	12
	'''
	coords = np.array([lons, lats]).T
	idx = np.argmin(np.linalg.norm(coords - targetcoord, axis=1))
	return int(idx)

def mpaso_mesh_latlon():
	mesh = xr.open_dataset(MESHFILE_OCN)
	lonCell = mesh.lonCell.values
	latCell = mesh.latCell.values
	lonCell = 180 / np.pi * lonCell
	latCell = 180 / np.pi * latCell
	lonCell[lonCell > 180] -= 360
	return latCell, lonCell, mesh.nCells.values

def max_mld_ts_dir(runnum='0151'):
	return f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/E3SM-Arcticv2.1_historical{runnum}/timeseries_data/maxMLD/'


def subset_region_from_ds(region_names, ds):
	if not isinstance(region_names, list):
		region_names = [region_names]

	all_regions = ds.regionNames.data
	idx = []
	for reg in region_names:
		i = np.argwhere(all_regions == reg)
		idx.append(int(i))

	return ds.isel(nRegions=idx)

def get_e3sm_run_path(runname, component):
	if runname == 'control':
		runname = 'E3SMv2.1B60to10rA02'
		path = E3SM_SIM_PATH + runname
	elif runname == 'ssp370_0101':
		runname = 'E3SM-Arcticv2.1_ssp370_0101'
		path = f'/pscratch/sd/d/dcomeau/e3sm_scratch/pm-cpu/{runname}/'
	else:
		runname = f'E3SM-Arcticv2.1_{runname}'
		path = E3SM_SIM_PATH + runname

	path += f'/archive/{component}/hist/'
	return path, runname

def get_control_singlevar_file(varname, component):
	# todo: make this work
	path = get_e3sm_run_path(runname='control', component=component)[0] + '../{varname}'

def get_mpaso_file_by_date(year, month, runname, varname='timeSeriesStatsMonthly'):
	'''
	Open and return the netcdf file containing atmospheric data for the specified run number
	:param year: year [int]
	:param month: month [int]
	:param runname: run name [string], use 'control' for 400 yr control spin-up
	:param varname: output variable type str
	:return: xarray dataset
	'''
	path, runname = get_e3sm_run_path(runname, 'ocn')
	path += f'{runname}.mpaso.hist.am.{varname}.{year:04}-{month:02}-01.nc'

	if not os.path.exists(path):
		print(path, ' does not exist')
		raise FileNotFoundError
	else:
		f = xr.open_dataset(path)
		return f

def get_atmo_file_by_date(year, month, runname, timestep='h0'):
	'''
	Open and return the netcdf file containing atmospheric data for the specified run number
	:param year: year [int]
	:param month: month [int]
	:param runname: run name [string], use 'control' for 400 yr control spin-up
	:param timestep: h0 - h4
		h0: monthly averages
		h1: daily averages
		h2: 6-hourly averages
		h3: 6-hourly instantaneous fields
		h4: other 6-hourly instantaneous fields
	:return: xarray dataset
	'''

	path, runname = get_e3sm_run_path(runname, 'atm')
	path += f'{runname}.eam.{timestep}.{year:04}-{month:02}.nc'

	if not os.path.exists(path):
		print(path, ' does not exist')
		raise FileNotFoundError
	else:
		f = xr.open_dataset(path)
		return f

def get_climatology(varname, month, runtype):
	fname = '/global/cfs/cdirs/m1199/romina/data/climos/'
	fname += f'{varname}_climo_m{month:02}_{runtype}.nc'

	if os.path.exists(fname):
		print('File exists. Returning it.')
		return xr.open_dataset(fname)

	runnames = ['control']
	yearrange = range(1, 387)

	if runtype == 'historical':
		runnames = ['historical0101', 'historical0151', 'historical0201', 'historical0251', 'historical0301']
		yearrange = range(1950, 2015)

	elif runtype == 'forecast':
		runnames = ['ssp370_0101', 'ssp370_0201']
		yearrange = range(2015, 2097)

	ocn_fields = {
		'maxMLD':'timeMonthlyMax_max_dThreshMLD',
		'filename':'timeSeriesStatsMonthlyMax',
	}


	ds = {}
	for runname in runnames:
		print(runname)
		times_included = []
		climodat = None
		for year in tqdm(yearrange):
			if varname in ocn_fields:

				if runname == 'control' and varname == 'maxMLD':
					fpath = (get_e3sm_run_path(runname, 'ocn')[0] +
							 '../singleVarFiles/maxMLD/'
							 f'dThreshMLD.E3SMv2.1B60to10rA02.mpaso.hist.am.timeSeriesStatsMonthlyMax.{year:04}-{month:02}-01.nc')
					modeldat = xr.open_dataset(fpath)
				else:
					modeldat = get_mpaso_file_by_date(year, month, runnames[0], varname=ocn_fields['filename'])

				if climodat is None:
					climodat = modeldat[ocn_fields[varname]].rename(varname)
					climodat['runname'] = runname
				else:
					climodat.data += modeldat[ocn_fields[varname]].values

				times_included.append(modeldat.Time.values[0])

		climodat.data /= len(times_included)

		ds[runname] = climodat

	da = xr.concat(ds.values(), dim='runname')
	ds = xr.Dataset({varname:da})
	ds['dates_included'] = times_included
	ds.to_netcdf(fname)
	return ds









if __name__ == '__main__':
	# get_arctic_ocn_region_mask('Labrador Sea')

	for i in range(12):
		print(i + 1)
		get_climatology('maxMLD', i+1, 'control')
	# mpaso_mesh_latlon()

	# f = xr.open_dataset('/global/cfs/cdirs/m1199/e3sm-arrm-simulations/E3SM-Arcticv2.1_historical0151/archive/ocn/hist/E3SM-Arcticv2.1_historical0151.mpaso.hist.am.timeSeriesStatsMonthlyMax.1960-01-01.nc')
	# print()
	# year = 1950
	# max_mld_file = max_mld_ts_dir() + f'arcticRegions_max_year{year}.nc'
	#
	# max_mld = xr.open_dataset(max_mld_file)
	#
	# roi = subset_region(['Labrador Sea', 'Irminger Sea'], max_mld)
	# print()
