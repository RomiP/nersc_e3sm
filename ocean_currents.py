import numpy as np

from bootstrap import *
from open_e3sm_files import *
from plot_unstructured import *


def plot_current_at_depth(data, depth_range, lat, lon, hres=100, mask=None):
	z = mpaso_depth()
	iz1 = int(np.argmin(abs(z - depth_range[0])))
	iz2 = int(np.argmin(abs(z - depth_range[1])))

	u = data[VARNAMES['vzonal']][mask, iz1:iz2].mean(dim='nVertLevels', skipna=True)
	v = data[VARNAMES['vmeridional']][mask, iz1:iz2].mean(dim='nVertLevels', skipna=True)

	fig, ax = unstructured_pcolor(lat, lon, np.sqrt(u**2 + v**2),
								  extent=lon_range + lat_range)

	plt.show()


if __name__ == '__main__':
	runnum = 'historical0101'
	startdate = dt.datetime(1950, 1, 1)
	enddate = dt.datetime(1951, 1, 1)

	lat_range = [50, 70]
	lon_range = [-81, 0]
	depth_range = [300, 500]

	data = zip_subset_by_time(
		make_monthly_date_list(startdate, enddate),
		get_mpaso_file_by_date,
		varnames=['vzonal', 'vmeridional'],
		runname=runnum
	)
	data = data.mean(dim='Time', skipna=True)

	lat, lon, ncell = mpaso_mesh_latlon()
	mask = (lat_range[0] < lat) & (lat < lat_range[1]) & (lon_range[0] < lon) & (lon < lon_range[1])

	plot_current_at_depth(data, depth_range, lat[mask], lon[mask], mask=mask, hres=100)


