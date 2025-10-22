import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
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

def mpaso_mesh_latlon():
	mesh = xr.open_dataset(MESHFILE_OCN)
	lonCell = mesh.lonCell.values
	latCell = mesh.latCell.values
	lonCell = 180 / np.pi * lonCell
	latCell = 180 / np.pi * latCell
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

def get_mpaso_var_by_date(varname, year, month, runname):

	historical = True
	if year > 2014:
		historical = False

	path = E3SM_SIM_PATH + f'E3SM-Arcticv2.1_{runname}/archive/ocn/hist/'


def get_atmo_file_by_date(year, month, runname, timestep='h0'):
	'''

	:param year:
	:param month:
	:param runname:
	:param timestep: h0 - h4
		h0: monthly averages
		h1: daily averages
		h2: 6-hourly averages
		h3: 6-hourly instantaneous fields
		h4: other 6-hourly instantaneous fields
	:return:
	'''


	fileroot = E3SM_SIM_PATH + f'E3SM-Arcticv2.1_{runname}/archive/atm/hist/'
	fname = f'E3SM-Arcticv2.1_{runname}.eam.{timestep}.{year}-{month:02}.nc'

	return xr.open_dataset(fileroot+fname)




if __name__ == '__main__':
	get_arctic_ocn_region_mask('Labrador Sea')
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
