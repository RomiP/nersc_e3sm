import matplotlib.pyplot as plt
import numpy as np

from bootstrap import *
from mesh_tools import djikstra_edges, sign_flip, edge_sign_for_direction
from open_e3sm_files import *
from plot_unstructured import *


def vector_plot(u, v, lat, lon, **kwargs):
	speed = np.sqrt(u ** 2 + v ** 2)

	fig, ax, sc = unstructured_pcolor(lat, lon, speed, dotsize=1, sc_handle=True, clim=kwargs['clim'])
	plt.show()


if __name__ == '__main__':

	mesh = xr.open_dataset(MESHFILE_OCN)
	lats, lons, _ = mpaso_mesh_latlon(mesh)
	z = mpaso_depth(mesh)
	depth = 500
	print(depth)
	zidx = np.argmin(np.abs(z-depth)).squeeze()

	mask = geopolygon_mask('regional_masks/LabSea_whole.geojson', lons, lats)
	cellnums = np.argwhere(mask).squeeze()

	data = get_mpaso_file_by_date(1950, 1, ENSEMBLE[0]).isel(Time=0, nCells=cellnums, nVertLevels=zidx)
	vector_plot(data[VARNAMES['vzonal']], data[VARNAMES['vmeridional']],
				lats[mask], lons[mask], clim=[0,0.7])

