
RE = 6.4e6 # Earth's radius (m)
g = 9.81 # gravity (m/s^2)

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

VARNAMES = {
	'maxMLD': 'timeMonthlyMax_max_dThreshMLD',
	'sal': 'timeMonthly_avg_activeTracers_salinity',
	'ocntemp': 'timeMonthly_avg_activeTracers_temperature',
	'dens': 'timeMonthly_avg_density',
	'pdens': 'timeMonthly_avg_potentialDensity',
	'bvfml': 'timeMonthly_avg_bruntVaisalaFreqML',
	'brn': 'timeMonthly_avg_bulkRichardsonNumber',
	'bld': 'timeMonthly_avg_boundaryLayerDepth',
	'qlat': 'timeMonthly_avg_latentHeatFlux',
	'lwhfd': 'timeMonthly_avg_longWaveHeatFluxDown',
	'lwhfu': 'timeMonthly_avg_longWaveHeatFluxUp',
	'swhf': 'timeMonthly_avg_shortWaveHeatFlux',
	'qsens': 'timeMonthly_avg_sensibleHeatFlux',
	'sic': 'timeMonthly_avg_iceAreaCell',
	'isice': 'timeMonthly_avg_icePresent',
	'sssal': 'timeMonthly_avg_seaSurfaceSalinity',
	'sst': 'timeMonthly_avg_seaSurfaceTemperature',
	'sia': 'timeMonthly_avg_iceAgeCell',
	'siv': 'timeMonthly_avg_iceVolumeCell',
	'fsaledge': 'timeMonthly_avg_activeTracerHorizontalAdvectionEdgeFlux_salinityHorizontalAdvectionEdgeFlux',
	'ftempedge': 'timeMonthly_avg_activeTracerHorizontalAdvectionEdgeFlux_temperatureHorizontalAdvectionEdgeFlux',
	'vzonal': 'timeMonthly_avg_velocityZonal',
	'vmeridional': 'timeMonthly_avg_velocityMeridional',
}

COMPONENTS = {'composites': ['qnet'],
			  'atm': [],
			  'ice': ['sic', 'isice', 'sic', 'siv'],
			  'ocn': ['maxMLD', 'bvfml', 'lwhfd', 'lwhfu', 'swhf', 'qsens', 'qlat', 'sal', 'ocntemp', 'pdens',
					  'bvfml', 'brn', 'bld', 'vzonal', 'vmeridional']}

MESHFILE_OCN = ('/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/'
				'mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc')

E3SM_SIM_PATH = '/global/cfs/cdirs/m1199/e3sm-arrm-simulations/'