
RE = 6.4e6 # Earth's radius (m)
g = 9.81 # gravity (m/s^2)

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ENSEMBLE = ['historical0101', 'historical0151', 'historical0201', 'historical0251', 'historical0301']

VARNAMES = {
	'bld': 'timeMonthly_avg_boundaryLayerDepth',
	'brn': 'timeMonthly_avg_bulkRichardsonNumber',
	'bvfml': 'timeMonthly_avg_bruntVaisalaFreqML',
	'dens': 'timeMonthly_avg_density',
	'fsaledge': 'timeMonthly_avg_activeTracerHorizontalAdvectionEdgeFlux_salinityHorizontalAdvectionEdgeFlux',
	'ftempedge': 'timeMonthly_avg_activeTracerHorizontalAdvectionEdgeFlux_temperatureHorizontalAdvectionEdgeFlux',
	'isice': 'timeMonthly_avg_icePresent',
	'lwhfd': 'timeMonthly_avg_longWaveHeatFluxDown',
	'lwhfu': 'timeMonthly_avg_longWaveHeatFluxUp',
	'maxMLD': 'timeMonthlyMax_max_dThreshMLD',
	'mlev': 'timeMonthly_avg_normalMLEvelocity',
	'ocntemp': 'timeMonthly_avg_activeTracers_temperature',
	'pdens': 'timeMonthly_avg_potentialDensity',
	'qlat': 'timeMonthly_avg_latentHeatFlux',
	'qsens': 'timeMonthly_avg_sensibleHeatFlux',
	'sal': 'timeMonthly_avg_activeTracers_salinity',
	'sia': 'timeMonthly_avg_iceAgeCell',
	'sic': 'timeMonthly_avg_iceAreaCell',
	'siv': 'timeMonthly_avg_iceVolumeCell',
	'ssh': 'timeMonthly_avg_ssh',
	'sssal': 'timeMonthly_avg_seaSurfaceSalinity',
	'sst': 'timeMonthly_avg_seaSurfaceTemperature',
	'swhf': 'timeMonthly_avg_shortWaveHeatFlux',
	'vmeridional': 'timeMonthly_avg_velocityMeridional',
	'vzonal': 'timeMonthly_avg_velocityZonal',
}

COMPONENTS = {'composites': ['qnet', 'eke'],
			  'atm': [],
			  'ice': ['sic', 'isice', 'sic', 'siv'],
			  'ocn': ['maxMLD', 'bvfml', 'lwhfd', 'lwhfu', 'swhf', 'qsens', 'qlat', 'sal', 'ocntemp', 'pdens',
					  'bvfml', 'brn', 'bld', 'vzonal', 'vmeridional', 'mlev', 'ssh']}

MESHFILE_OCN = ('/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/'
				'mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc')

E3SM_SIM_PATH = '/global/cfs/cdirs/m1199/e3sm-arrm-simulations/'