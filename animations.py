import datetime as dt
from helpers import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from open_e3sm_files import *
from plot_unstructured import *
from tqdm import tqdm

import matplotlib as mpl

mpl.rcParams["animation.ffmpeg_path"] = ("/global/common/software/e3sm/anaconda_envs/e3smu_1_12_0/"
										 "pm-cpu/conda/envs/e3sm_unified_1.12.0_login/bin/ffmpeg")  # e.g., from conda


def copilot_example():
	# Create a moving 2D Gaussian
	ny, nx = 100, 100
	Y, X = np.mgrid[-1:1:complex(ny), -1:1:complex(nx)]

	def frame_img(t):
		cx = 0.6 * np.cos(t)
		cy = 0.6 * np.sin(t)
		return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 0.1)

	fig, ax = plt.subplots(figsize=(4, 4))
	img0 = frame_img(0.0)
	im = ax.imshow(img0, vmin=0, vmax=1, cmap='viridis', origin='lower',
				   extent=[-1, 1, -1, 1])  # vmin/vmax fix the color scale
	ax.set_title("Moving Gaussian (fixed color scale & axes)")
	ax.set_xlabel("x");
	ax.set_ylabel("y")

	def update(frame):
		print(f'frame: {frame}')
		im.set_data(frame_img(frame * 0.1))

		return (im,)

	ani = FuncAnimation(fig, update, frames=120, blit=True, interval=50)
	writer = FFMpegWriter(fps=30, codec="libvpx-vp9", bitrate=2500, extra_args=["-pix_fmt", "yuv420p"])
	# ani.save("figs/test.mp4", writer=writer)
	plt.show()


def save_with_progress(fig, update, nframes, out_path,
					   fps=24, dpi=110,
					   codec="libx264",
					   extra_args=("-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "22")):
	writer = FFMpegWriter(fps=fps, codec=codec, extra_args=list(extra_args))
	with writer.saving(fig, out_path, dpi):
		for i in tqdm(range(nframes), desc="Saving video", unit="frame"):
			update(i)  # your update function must modify the artists for frame i
			writer.grab_frame()


def animation(frames, lat, lon, saveas, titles=[], **kwargs):
	fig, ax, sc = unstructured_pcolor(lat, lon, frames[1], sc_handle=True, **kwargs)
	if titles:
		title = ax.set_title("Initial title")

	def update(i):
		sc.set_array(frames[i])
		if titles:
			title.set_text(titles[i])
		return (sc,)

	print('init animation')
	ani = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=50)
	# print('writing animation')
	# writer = FFMpegWriter(fps=4, codec="libx264", bitrate=2500,
	# 					  extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast"])
	# # writer = FFMpegWriter(fps=4, codec="libvpx-vp9", bitrate=2500, extra_args=["-pix_fmt", "yuv420p"])
	# ani.save("figs/testdata.mp4", writer=writer, dpi=110)
	save_with_progress(fig, update, nframes=len(frames), fps=4, out_path=saveas)


def animate_e3sm_2d(runnum):
	dates = make_monthly_date_list(dt.datetime(1950, 1, 1), dt.datetime(2014, 12, 31))
	index_var = 'maxMLD'
	runtype = 'historical'
	varname = VARNAMES[index_var]

	lat, lon, ncells = mpaso_mesh_latlon()

	mask = (lat >= 45) & (lon <= -20)
	lat = lat[mask]
	lon = lon[mask]

	frames = []
	titles = []
	for d in tqdm(dates):
		ds = get_mpaso_file_by_date(d.year, d.month, runtype + runnum, varname='timeSeriesStatsMonthlyMax')
		frames.append(ds[varname].values.squeeze()[mask])
		titles.append(d.strftime('%Y %B'))

	plotargs = dict(
		projname='Miller',
		extent=[-70, -30, 50, 70],
		clim=[0, 3000],
		cmap='turbo',
		clabel='Max MLD (m)',
		# extenttype='tight',
		dotsize=1,
		gridlines=True,
		title=' Test',
	)
	savename = f'figs/{runtype}{runnum}_{index_var}.mp4'
	animation(frames, lat, lon, savename, titles, **plotargs)


if __name__ == '__main__':
	animate_e3sm_2d('0101')
# copilot_example()
