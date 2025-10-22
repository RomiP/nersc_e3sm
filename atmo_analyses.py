import datetime as dt
from dateutil.relativedelta import relativedelta
from open_e3sm_files import *
from tqdm import tqdm

def produce_NAO_ts(runname, startdate, enddate):
	il_coord = []
	# Icelow (Akureyri)
	lat_ice = 65.7
	lon_ice = -18.1

	# Azohigh (Ponta Delgada)
	lat_azo = 37.7
	lon_azo = -25.7

	dates = []
	while startdate < enddate:
		dates.append(startdate)
		startdate += relativedelta(months=1)


	il = []
	ah = []
	for date in tqdm(dates):
		atmodata = get_atmo_file_by_date(date.year, date.month, runname)
		lat = atmodata.lat.values
		lon = atmodata.lon.values
		# todo: vectorize coords to find nearest
		# idx_il =


if __name__ == '__main__':

	runname = 'historical0101'
	startdate = dt.datetime(1950, 1, 1)
	enddate = dt.datetime(2015, 1 ,1)
	produce_NAO_ts(runname, startdate, enddate)
