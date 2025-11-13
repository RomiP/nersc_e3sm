from helpers import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from open_e3sm_files import *
import os
import xarray as xr

# from analysis.make_plots import make_scatter_plot
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
	dlevels = [0., 50.]  # depth levels

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

	# # fld = (localArea * fld.where(cellMask, drop=True)).sum(dim='nCells') / regionalArea
	# lon = (cellMask * lonCell).where(cellMask, drop=True)
	# lat = (cellMask * latCell).where(cellMask, drop=True)
	# fld = (cellMask * fld).where(cellMask, drop=True)

	lat = latCell
	lon = lonCell

	print('starting image')
	fig, ax = unstructured_pcolor(lat, lon, fld, landmask=True, dotsize=dotSize, clabel=varunits,
								  extent=[-70, -30, 50, 70],
								  interp=False)
	print('done')

	plt.title(vartitle + ' ' + figtitle)
	plt.show()


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
		roi = subset_region_from_ds(['Labrador Sea'], max_mld)
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


def plot_atm_data():
	y = 1960
	m = 1
	runname = 'historical0101'
	root = '/global/cfs/cdirs/m1199/e3sm-arrm-simulations/E3SM-Arcticv2.1_historical0101/archive/atm/hist/'
	file = f'E3SM-Arcticv2.1_{runname}.eam.h0.{y}-{m:02}.nc'
	atm = xr.open_dataset(root + file)

	fig, ax = unstructured_pcolor(atm['lat'].values, atm['lon'].values, atm['PSL'].values,
								  extent=[-100, 0, 30, 80],
								  # extenttype='tight',
								  gridlines=True,
								  interp=True)
	plt.show()


def playing_with_plotly():
	import plotly.graph_objects as go
	fig = go.Figure(data=
	go.Contour(
		z=[[10, 10.625, 12.5, 15.625, 20],
		   [5.625, 6.25, 8.125, 11.25, 15.625],
		   [2.5, 3.125, 5., 8.125, 12.5],
		   [0.625, 1.25, 3.125, 6.25, 10.625],
		   [0, 0.625, 2.5, 5.625, 10]],
		dx=10,
		x0=5,
		dy=10,
		y0=10,
	)
	)

	fig.show()


def plot_climo(runnum='avg'):
	runtype = 'historical'
	varname = 'slp'
	month = [1,2,3]
	if isinstance(month, int):
		climo = get_climatology(varname, month, runtype)
		mstr = f'm{month:02}'
	else:
		climo = get_climatology(varname, month[0], runtype)
		for m in month[1:]:
			climo[varname].data += get_climatology(varname, m, runtype)[varname].data

		climo[varname].data /= len(month)
		mstr = f'{MONTHS[month[0]-1]}-{MONTHS[month[-1]-1]}'


	if varname in ['slp']:
		lat = climo['lat'].data
		lon = climo['lon'].data
		lon[lon>180] -= 360
	else:
		lat, lon, cellnum = mpaso_mesh_latlon()
	# file = '/global/cfs/projectdirs/m1199/romina/data/maxMLD_climo_m01_historical.nc'
	# climo = xr.open_dataset(file)

	if runnum == 'avg':
		climo = climo.mean(dim='runname')
	else:
		climo = climo.sel(runname=runtype + runnum)

	fig, ax = unstructured_pcolor(lat, lon, climo[varname].values,
								  # extent=[-70, -30, 50, 70],
								  extent=[-70, 10, 30, 70],
								  extenttype='tight',
								  cmap='viridis',
								  clim=[99500, 102500],
								  # dotsize=5,
								  gridlines=True,
								  interp='grid')
	plt.title(f'{varname} Climatology {mstr} {runtype + runnum}')

	plt.savefig(f'figs/{varname}_climo_{mstr.lower()}_{runtype+runnum}.png')
	plt.show()


def plot_qnet_data():
	runtype = 'historical'
	runnum = 'avg'
	qnet_file = f'/global/cfs/cdirs/m1199/romina/data/composite_fields/netHeatFlux_{runtype}.nc'
	qnetds = xr.open_dataset(qnet_file)

	if runnum == 'avg':
		qnetds = qnetds.mean(dim='runname')
	else:
		qnetds = qnetds.sel(runname=runtype + runnum)

	# climo for specific months
	y, m, d = dt64_y_m_d(qnetds.Time.values)
	idx = np.squeeze(np.argwhere((m <= 3) | (m >= 10)))
	qnetds = qnetds.isel(Time=idx)

	#  for full year climo
	qnetds = qnetds.mean(dim='Time')

	lat, lon, cellnum = mpaso_mesh_latlon()

	fig, ax = unstructured_pcolor(lat, lon, qnetds['netHeatFlux'].values,
								  projname='Miller',
								  extent=[-70, -30, 50, 70],
								  clim=[-300, 100],
								  cmap='coolwarm',
								  norm=0,
								  clabel='Net Heat Flux ($W/m^2$)',
								  # extenttype='tight',
								  gridlines=True,
								  interp='mosaic',
								  title=runtype + runnum + ' Oct-Mar',
								  )

	plt.savefig(f'figs/qnet_climo_oct-mar_{runtype}{runnum}.png')
	plt.show()


def plot_heatmap(runnum):
	runtype = 'historical'
	maskfile = f'/global/cfs/projectdirs/m1199/romina/data/misc/maxMLDmask_{runtype}.nc'

	mask = xr.open_dataset(maskfile)

	if runnum == 'avg':
		mask = mask.mean(dim='runname')
	else:
		mask = mask.sel(runname=runtype + runnum)

	# climo for specific months
	y, m, d = dt64_y_m_d(mask.Time.values)
	idx = np.squeeze(np.argwhere((m <= 3) | (m >= 10)))
	mask = mask.isel(Time=idx)

	#  for full year climo
	mask = mask.mean(dim='Time')

	lat, lon, cellnum = mpaso_mesh_latlon()

	fig, ax = unstructured_pcolor(lat, lon, mask['maxMLDmask'].values,
								  extent=[-70, -30, 50, 70],
								  # clim=[-300, 100],
								  # cmap='coolwarm',
								  # norm=0,
								  clabel='max MLD location frequency',
								  # extenttype='tight',
								  gridlines=True,
								  # interp='mosaic',
								  title=runtype + runnum + ' Oct-Mar',
								  )

	# plt.savefig(f'figs/qnet_climo_oct-mar_{runtype}{runnum}.png')
	plt.show()


if __name__ == '__main__':
	# supported values are ['gtk3agg', 'gtk3cairo', 'gtk4agg', 'gtk4cairo', 'macosx', 'nbagg', 'notebook', 'qtagg',
	# 'qtcairo', 'qt5agg', 'qt5cairo', 'tkagg', 'tkcairo', 'webagg', 'wx', 'wxagg', 'wxcairo', 'agg', 'cairo', 'pdf',
	# 'pgf', 'ps', 'svg', 'template', 'module://backend_interagg', 'inline']
	matplotlib.use('module://backend_interagg')
	print(matplotlib.get_backend())

	# x = np.array([1,2,3,4,5,6,7,8,9])
	# plt.plot(x,x)
	# plt.show()

	# unstructured_pcolor(0,0,0)
	# open_some_data()
	# plot_atm_data()
	for run in ['0101', '0151', '0201', '0251', '0301', 'avg']:
		# plot_climo(run)
		plot_heatmap(run)
# plot_qnet_data()

# path = '/global/cfs/projectdirs/m1199/e3sm-arrm-simulations/TL319_r05_ARRM10to60E2r1.JRA-MOSART-Phys/archive/ocn/hist/'
# fname = 'TL319_r05_ARRM10to60E2r1.JRA-MOSART-Phys.mpaso.hist.am.eddyProductVariables.1991-01-01.nc'
# dat = xr.open_dataset(path + fname)
