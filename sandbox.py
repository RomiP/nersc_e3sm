from helpers import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from open_e3sm_files import *
import os
import xarray as xr

from analysis.make_plots import make_scatter_plot
from mpas_analysis.shared.io.utility import get_files_year_month, decode_strings

from plot_unstructured import *

colorIndices = [0, 14, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255]

def open_some_data():
	# Settings for nersc
	meshName = 'ARRM10to60E2r1'
	meshfile = ('/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/'
				'mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc')
	runname = 'E3SM-Arcticv2.1_historical0151'
	modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runname}/archive'
	regionMaskDir = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/'
	isShortTermArchive = True


	year = 1960
	month = 1

	############ MPAS-Ocean fields
	modelname = 'ocn'
	modelnameOut = 'ocean'
	modelcomp = 'mpaso'

	# mpasFile = 'highFrequencyOutput'
	# mpasFileDayformat = '01_00.00.00'
	# timeIndex = 0
	# variables = [
	# 			{'name': 'temperatureAtSurface',
	# 			 'mpasvarname': 'temperatureAtSurface',
	# 			 'title': 'SST',
	# 			 'units': '$^\circ$C',
	# 			 'factor': 1,
	# 			 'colormap': plt.get_cmap('RdBu_r'),
	# 			 'clevels':   [-1.8, -1.6, -0.5, 0.0, 0.5, 2.0, 4.0, 8.0, 12., 16., 22.],
	# 			 'clevelsNH': [-1.8, -1.6, -1.0, -0.5, 0.0, 0.5, 2.0, 4.0, 6.0, 8.0, 12.],
	# 			 'clevelsSH': [-1.8, -1.6, -1.0, -0.5, 0.0, 0.5, 2.0, 4.0, 6.0, 8.0, 12.],
	# 			 'is3d': False},
	# ]
	dlevels = [0., 50.] # depth levels

	mpasFile = 'timeSeriesStatsMonthlyMax'
	mpasFileDayformat = '01'
	timeIndex = 0
	variables = [
	            {'name': 'maxMLD',
	             'mpasvarname': 'timeMonthlyMax_max_dThreshMLD',
	             'title': 'Maximum MLD',
	             'units': 'm',
	             'factor': 1,
	             'colormap': plt.get_cmap('viridis'),
	             'clevels': [10., 50., 80., 100., 150., 200., 300., 400., 800., 1200., 2000.],
	             'clevelsNH': [50., 100., 150., 200., 300., 500., 800., 1200., 1500., 2000., 3000.],
	             'clevelsSH': [10., 50., 80., 100., 150., 200., 300., 400., 800., 1200., 2000.],
	             'is3d': False}
	           ]

	# Info about MPAS mesh
	dsMesh = xr.open_dataset(meshfile)
	lonCell = dsMesh.lonCell.values
	latCell = dsMesh.latCell.values
	lonCell = 180 / np.pi * lonCell
	latCell = 180 / np.pi * latCell
	lonEdge = dsMesh.lonEdge.values
	latEdge = dsMesh.latEdge.values
	lonEdge = 180 / np.pi * lonEdge
	latEdge = 180 / np.pi * latEdge
	lonVertex = dsMesh.lonVertex.values
	latVertex = dsMesh.latVertex.values
	lonVertex = 180 / np.pi * lonVertex
	latVertex = 180 / np.pi * latVertex
	depth = dsMesh.bottomDepth
	# Find model levels for each depth level (relevant if plotDepthAvg = False)
	z = dsMesh.refBottomDepth
	zlevels = np.zeros(np.shape(dlevels), dtype=np.int64)
	for id in range(len(dlevels)):
		dz = np.abs(z.values - dlevels[id])
		zlevels[id] = np.argmin(dz)

	modelfile = f'{modeldir}/ocn/hist/{runname}.{modelcomp}.hist.am.{mpasFile}.{year:04d}-{month:02d}-{mpasFileDayformat}.nc'
	print(modelfile)
	# modelfile = f'{modeldir}/{modelcomp}.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-{mpasFileDayformat}.nc' # old (v1) filename format
	ds = xr.open_dataset(modelfile).isel(Time=timeIndex)

	# groupName = 'arctic_atlantic_budget_regions'
	# regionMaskFile = f'{regionMaskDir}/{meshName}_{groupName}.nc'
	regionMaskFile = f'{regionMaskDir}/ARRM10to60E2r1_arcticRegions.nc'
	if os.path.exists(regionMaskFile):
		dsRegionMask = xr.open_dataset(regionMaskFile)
		allRegions = decode_strings(dsRegionMask.regionNames)

	areaCell = dsMesh.areaCell

	regionIndices = []
	cellMask = None
	regionNames = ['Labrador Sea', 'Irminger Sea']
	for regionName in regionNames:
		print(regionName)
		regionIndex = allRegions.index(regionName)
		# regionIndices.append(regionIndex)

		dsMask = dsRegionMask.isel(nRegions=regionIndex)

		if cellMask is not None:
			cellMask = np.logical_or(cellMask, dsMask.regionCellMasks == 1)
		else:
			cellMask = dsMask.regionCellMasks == 1
	#
	# 	localArea = areaCell.where(cellMask, drop=True)
	# 	regionalArea = localArea.sum()

	# regionIndex = regionNames.index('North Atlantic subpolar gyre')
	# dsMask = dsRegionMask.isel(nRegions=regionIndex)
	# cellMask = dsMask.regionCellMasks == 1

	figtitle = 'Jan 1960'
	dotSize = 0.25
	# make_scatter_plot(lon, lat, dotSize, figtitle, fld=ds['temperatureAtSurface'])

	var = variables[0]
	varname = var['name']
	mpasvarname = var['mpasvarname']
	factor = var['factor']
	clevels = var['clevels']
	clevelsNH = var['clevelsNH']
	clevelsSH = var['clevelsSH']
	colormap = var['colormap']
	vartitle = var['title']
	varunits = var['units']
	fld = ds[mpasvarname]
	cindices = colorIndices

	# fld = (localArea * fld.where(cellMask, drop=True)).sum(dim='nCells') / regionalArea
	lon = (cellMask * lonCell).where(cellMask, drop=True)
	lat = (cellMask * latCell).where(cellMask, drop=True)
	fld = (cellMask * fld).where(cellMask, drop=True)

	print('starting image')
	fig, ax = unstructured_pcolor(lat, lon, fld, landmask=True, dotsize=0.3, clabel=varunits)
	print('done')

	plt.title(vartitle + ' ' + figtitle)
	plt.show()

	# make_scatter_plot(lon, lat, dotSize, figtitle,
	# 				  fld=fld, cmap=colormap, clevels=clevels, cindices=cindices, cbarLabel=varunits)

def plot_regional_avg_max_mld():
	months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
	ystart = 1950
	yend = 2015  # historical only goes until 2014

	# Get a specific colormap, e.g., 'viridis'
	cmap = cm.get_cmap('viridis')
	# need to normalize because color maps are defined in [0, 1]
	norm = colors.Normalize(ystart, yend)

	for year in range(ystart, yend):
		print(year)
		max_mld_file = max_mld_ts_dir() + f'arcticRegions_max_year{year}.nc'
		max_mld = xr.open_dataset(max_mld_file)
		roi = subset_region(['Labrador Sea'], max_mld)
		mld = np.squeeze(roi.maxMLD.data)
		c = cmap(norm(year))
		plt.plot(rotate(months, -3), rotate(mld, -3), c=c, alpha=0.8, lw=1)

	ax = plt.gca()
	ax.invert_yaxis()

	plt.ylabel('Max MLD (m)')
	plt.title('Labrador Sea Max MLD')

	# plot colorbar
	plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
	plt.show()

if __name__ == '__main__':
	print('hello world')

	# supported values are ['gtk3agg', 'gtk3cairo', 'gtk4agg', 'gtk4cairo', 'macosx', 'nbagg', 'notebook', 'qtagg', 'qtcairo', 'qt5agg', 'qt5cairo', 'tkagg', 'tkcairo', 'webagg', 'wx', 'wxagg', 'wxcairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template', 'module://backend_interagg', 'inline']
	matplotlib.use('module://backend_interagg')
	print(matplotlib.get_backend())

	# x = np.array([1,2,3,4,5,6,7,8,9])
	# plt.plot(x,x)
	# plt.show()

	# unstructured_pcolor(0,0,0)
	open_some_data()