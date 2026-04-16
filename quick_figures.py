import matplotlib.pyplot as plt

from atmo_analyses import ts_seasonal_avg
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
from climos_and_extrema import get_climatology
import cmasher as cmr
from helpers import *
from open_e3sm_files import *
from plot_unstructured import *
import seawater as sw


def plot_geospatial_MLD_climo_convexhull():
	runnum = 'avg'
	runtype = 'historical'
	varname = 'maxMLD'
	month = [1, 2, 3]
	ovr_write = False
	if isinstance(month, int):
		climo = get_climatology(varname, month, runtype, overwrite=ovr_write)
		mstr = f'm{month:02}'
	else:
		climo = get_climatology(varname, month[0], runtype, overwrite=ovr_write)
		for m in month[1:]:
			climo[varname].data += get_climatology(varname, m, runtype, overwrite=ovr_write)[varname].data

		climo[varname].data /= len(month)
		mstr = f'{MONTHS[month[0] - 1]}-{MONTHS[month[-1] - 1]}'

	if varname in ['slp']:
		lat = climo['lat'].data
		lon = climo['lon'].data
		lon[lon > 180] -= 360
	else:
		lat, lon, cellnum = mpaso_mesh_latlon()

	if runnum == 'avg':
		climo = climo.mean(dim='runname')
	else:
		climo = climo.sel(runname=runtype + runnum)

	fig, ax = unstructured_pcolor(lat, lon, climo[varname].values,
								  extent=[-70, -30, 50, 70],
								  extenttype='tight',
								  cmap='viridis',
								  # cmap=cmr.lavender.reversed(),
								  clim=[0, 2000],
								  clabel='MDL (m)',
								  gridlines=True,
								  landmask=True,
								  interp='grid')
	plt.title('Winter max MLD from E3SM ensemble mean')

	gdf = gpd.read_file('regional_masks/model_dczone.geojson')
	# 3. Create a ShapelyFeature
	geoms = gdf.geometry.values
	feat = ShapelyFeature(geoms, ccrs.PlateCarree(), edgecolor='red', facecolor='none', linewidth=2)

	# 4. Add to cartopy
	ax.add_feature(feat)

	plt.savefig(f'figs/pubs/{varname}_climo_{mstr.lower()}_{runtype + runnum}.png')
	plt.show()

def plot_E3SM_spatial_resolution():
	lat, lon, ncells = mpaso_mesh_latlon()
	lon[lon > 180] -= 360
	mesh = xr.open_dataset(MESHFILE_OCN)
	area = mesh.areaCell.values
	area = np.sqrt(area)/1000
	crs = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)

	fig = plt.figure(dpi=300)
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-10, 45))

	ax.add_feature(cfeature.LAND, zorder=11, edgecolor='black')

	ax.set_global()
	plt.scatter(lon, lat, c=area, s=0.01, cmap='viridis_r',
				zorder=10, transform=ccrs.PlateCarree())
	cbar = plt.colorbar()
	plt.clim(10, 50)
	cbar.set_label('Horizontal resolution (km)')

	plt.savefig('figs/pubs/MPASO_spatial_resolution.png')
	plt.show()

def plot_mld_ts(runnum):
	# runnum = '0151'
	tsroot = '/global/cfs/cdirs/m1199/romina/data/timeseries/'

	files = [
		'maxMLD_dcmean_ts_historical.nc',
	]
	yr_range = [1950, 2015]
	ensemble_mean = runnum == 'avg'
	title = f'run number: {runnum}'

	for f in files:
		dat = xr.open_dataset(tsroot + f)
		if ensemble_mean:
			dat = dat.mean(dim='runname')
			title = 'ensemble mean'
		else:
			type = 'historical'
			run = type + runnum

			if run in dat.runname:
				dat = dat.sel(runname=run)
			else:
				continue

		mld_annual = np.reshape(dat['maxMLD'].values, (-1, 12))
		mld_annual = np.max(mld_annual, axis=1)
		years, months, d = dt64_y_m_d(dat.time.values)
		years = np.reshape(years, (-1, 12))
		years = years[:, 0]

		# plt.plot(dat['time'], dat['maxMLD'])
		plt.plot(years, mld_annual)

	plt.xlabel('Year')
	plt.ylabel('Max MLD (m)')
	plt.title(f'Labrador Sea Annual Max MLD ({title})')

	plt.xlim(yr_range[0], yr_range[-1])
	plt.ylim([0,3500])

	ax = plt.gca()
	ax.invert_yaxis()
	plt.ylabel('Max MLD (m)')

	plt.savefig(f'figs/pubs/maxmld_annual_ts_{runnum}.png')
	plt.show()

def plot_mld_qnet_scatter(runnum):
	type = 'historical'
	monthrange = [1, 3]

	root = '/global/cfs/cdirs/m1199/romina/data/timeseries/'
	mldfile = f'maxMLD_dcmean_ts_{type}.nc'
	qnetfile = f'netHeatFlux_LabSeaDC_{type}.nc'

	mlddata = xr.open_dataset(root + mldfile)
	qnetdata = xr.open_dataset(root + qnetfile)

	if 'avg' in runnum:
		mlddata = mlddata.mean(dim='runname')
		qnetdata = qnetdata.mean(dim='runname')

	else:
		qnetdata = qnetdata.sel(runname=type + runnum)
		mlddata = mlddata.sel(runname=type + runnum)

	qnet = qnetdata['netHeatFlux'].values.reshape(-1,)
	mld = mlddata['maxMLD'].values  # .reshape(-1,)
	time = qnetdata.Time.values

	qnet, years = ts_seasonal_avg(qnet, time, monthrange)

	mld = mld.reshape(-1, 12)
	mld = np.max(mld, axis=1)
	if monthrange[1] < monthrange[0]:
		mld = mld[:-1]

	idx = qnet < -100

	plt.scatter(qnet[idx], mld[idx])
	plt.scatter(qnet[~idx], mld[~idx], c='tab:grey')
	plt.xlabel('Net Heat Flux ($W/m^2$)')
	plt.ylabel('Max MLD (m)')
	plt.title(f'Wintertime Net Heat Flux Effect on Max MLD (run number: {run})')

	plt.ylim(0, 3500)
	plt.xlim(-400, 0)

	# stats for only years with MLD >300m
	mld = mld[idx]
	qnet = qnet[idx]
	years = years[idx]
	corr = np.corrcoef(mld, qnet)[0, 1]
	r2 = round(corr ** 2, 4)
	print(corr)

	ax = plt.gca()
	plt.text(.05, 0.95, f'$R^2$ = {r2}', ha='left', va='top', transform=ax.transAxes)
	ax.invert_yaxis()
	plt.savefig(f'figs/pubs/qnet_vs_maxMLD_DCzone_{type + runnum}_m{monthrange[0]:02}-{monthrange[1]:02}.png')

	plt.show()

def plot_e3sm_density_profile(month):
	root = '/global/cfs/projectdirs/m1199/romina/data/'

	climo = []
	for runnum in ['0101', '0151', '0201', '0251', '0301']:
		fname = root + f'profiles/LabSea_ctd_dczone_historical{runnum}.nc'
		prof_ds = xr.open_dataset(fname)
		prof_ds = prof_ds.isel(nVertLevels=(prof_ds['z'] < 3500))

		time = np.array(
			[np.datetime64(x.isoformat()) for x in prof_ds.Time.values]
		)
		y, m, d = dt64_y_m_d(time)
		field = 'dens'
		varname = VARNAMES[field]
		sname = VARNAMES['sal']
		tname = VARNAMES['ocntemp']

		for i in range(len(time)):
			if m[i] == month:
				prof = prof_ds.isel(Time=i)
				dens = sw.dens0(prof[sname].values, prof[tname].values)
				# print(time[i].astype('datetime64[M]'), prof.Time.values)
				plt.plot(dens, prof['z'].values, color='tab:grey', linewidth=0.5, alpha=0.5)
				climo.append(dens)

	climo = np.array(climo)
	mu = np.nanmean(climo, axis=0)
	sigma = np.nanstd(climo, axis=0)

	plt.plot(mu, prof_ds['z'].values, color='tab:red', label='mean profile')
	plt.fill_betweenx(prof_ds['z'].values, mu - sigma, mu + sigma,
					  color='tab:red', alpha=0.2, zorder=100, label=r'$\pm\sigma$ range')

	plt.legend()
	ax = plt.gca()
	plt.ylim([0, 2000])
	plt.xlim([1026, 1028])
	plt.xlabel('Density (kg/m$^3$)')
	plt.ylabel('Depth (m)')
	plt.title(f'E3SM {MONTHS[month - 1]} Density Profile')
	ax.invert_yaxis()
	plt.savefig(f'figs/pubs/LabSea_density_upper1km_m{month:02}_allhistorical.png')
	plt.show()

def plot_argo_density_profile(month):
	pass

def plot_climo(runnum='avg', month=[1,2,3], varname='maxMLD', **kwargs):
	runtype = 'historical'
	# month = [1]
	ovr_write = False
	if isinstance(month, int):
		climo = get_climatology(varname, month, runtype, overwrite=ovr_write)
		mstr = f'm{month:02}'
	else:
		climo = get_climatology(varname, month[0], runtype, overwrite=ovr_write)
		for m in month[1:]:
			climo[varname].data += get_climatology(varname, m, runtype, overwrite=ovr_write)[varname].data

		climo[varname].data /= len(month)
		mstr = f'{MONTHS[month[0]-1]}-{MONTHS[month[-1]-1]}'


	if varname in ['slp']:
		lat = climo['lat'].data
		lon = climo['lon'].data
		lon[lon>180] -= 360
	else:
		lat, lon, cellnum = mpaso_mesh_latlon()


	if runnum == 'avg':
		climo = climo.mean(dim='runname')
	else:
		climo = climo.sel(runname=runtype + runnum)

	if varname == 'maxMLD':
		plotargs = dict(
			projname='Miller',
			extent=[-70, -30, 50, 70],
			clim=[0, 2000],
			cmap='viridis',
			clabel='Max MLD (m)',
			landmask=True,
			dotsize=1,
			gridlines=True,
			title='Max Mixed Layer Depth',
		)
	elif varname == 'sic':
		plotargs = dict(
			projname='Miller',
			extent=[-70, -30, 50, 70],
			clim=[0, 1],
			# cmap='Blues_r',
			cmap=cmr.ocean,
			clabel='Sea Ice Concentration (%)',
			landmask=True,
			dotsize=1,
			gridlines=True,
			title = 'Sea Ice Concentration',
		)
	elif varname == 'siv':
		plotargs = dict(
			projname='Miller',
			extent=[-70, -30, 50, 70],
			clim=[0, 1],
			# cmap='Blues_r',
			cmap=cmr.arctic,
			clabel='Sea Ice Thickness (m)',
			landmask=True,
			dotsize=1,
			gridlines=True,
			title = 'Sea Ice Thickness',
		)
	elif varname == 'sssal':
		plotargs = dict(
			projname='Miller',
			extent=[-70, -30, 50, 70],
			clim=[30, 34.5],
			# cmap='viridis',
			cmap=cmr.toxic.reversed(),
			clabel='Sea Surface Salinity (psu)',
			landmask=True,
			dotsize=1,
			gridlines=True,
			title='Sea Surface Salinity',
		)

	fig, ax = unstructured_pcolor(lat, lon, climo[varname].values, **plotargs)
	title = plotargs['title']
	if runnum == 'avg':
		plt.title(f'{title} Climatology {mstr}\nEnsemble average')
	else:
		plt.title(f'{title} Climatology {mstr}\n(run number: {runnum})')


	gdf = gpd.read_file('regional_masks/model_dczone.geojson')
	# ax.add_geometries([geo_polygon], crs=ccrs.PlateCarree(),
	# 				  facecolor='blue', alpha=0.5, edgecolor='black')
	# 3. Create a ShapelyFeature
	geoms = gdf.geometry.values
	feat = ShapelyFeature(geoms, ccrs.PlateCarree(), edgecolor='red', facecolor='none')

	# 4. Add to cartopy
	ax.add_feature(feat)

	if 'saveas' in kwargs:
		plt.savefig(kwargs['saveas'])
	else:
		plt.savefig(f'figs/pubs/{varname}_climo_{mstr.lower()}_{runtype+runnum}.png')
	plt.show()

if __name__ == '__main__':
	# plot_geospatial_MLD_climo_convexhull()
	# for run in ['0101', '0151', '0201', '0251', '0301']:
	# 	plot_mld_qnet_scatter(run)
		# plot_mld_ts(run)
	plot_E3SM_spatial_resolution()

	# v = 'maxMLD'
	# plot_climo(runnum='0201', varname=v, month=[1, 2, 3])
	# plot_climo(runnum='avg', varname=v, month=[1, 2, 3])
	# for month in [1, 2, 3]:
	# 	# plot_e3sm_density_profile(month)
	# 	# plot_argo_density_profile()
	# 	plot_climo(runnum='0201', varname='sic', month=month)