from helpers import *
from open_e3sm_files import *
from plot_unstructured import *

def plot_climo(runnum='avg'):
	runtype = 'historical'
	varname = 'maxMLD'
	month = [1,2,3]
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

	fig, ax = unstructured_pcolor(lat, lon, climo[varname].values,
								  extent=[-70, -30, 50, 70],
								  # extent=[-70, 10, 30, 70],
								  extenttype='tight',
								  cmap='viridis',
								  # clim=[99500, 102500],
								  clim=[100, 2500],
								  # dotsize=5,
								  gridlines=True,
								  landmask=True,
								  interp='grid')
	plt.title(f'{varname} Climatology {mstr} {runtype + runnum}')

	plt.savefig(f'figs/{varname}_climo_{mstr.lower()}_{runtype+runnum}.png')
	plt.show()

def get_extrema_ts(k):

	varname = 'maxMLD'
	f = f'/global/cfs/projectdirs/m1199/romina/data/timeseries/{varname}_ts_historical.nc'
	var_dat = xr.open_dataset(f)

	idx_sorted = np.argsort(var_dat[varname].values,axis=0)
	idx_min = idx_sorted[:k, :]
	idx_max = idx_sorted[-k:, :]

	yrs_min = var_dat.time.values[idx_min]
	datmin = var_dat[varname].values[idx_min]
	yrs_max = var_dat.time.values[idx_max]
	datmax = var_dat[varname].values[idx_max]

	# print(yrs_max.T)
	print(datmax.T)

if __name__ == '__main__':


	get_extrema_ts(3)

	# for run in ['0101', '0151', '0201', '0251', '0301', 'avg']:
	# 	plot_climo(run)
