from bootstrap import *
from open_e3sm_files import *
from plot_unstructured import *

def djikstra_edges(start, stop, mesh, mask=None):
	import heapq

	lons = np.degrees(mesh.lonEdge.values)
	lats = np.degrees(mesh.latEdge.values)
	lons[lons > 180] -= 360
	nedges = mesh.nEdges.values

	gateline = geom.LineString((start, stop))


	if mask is None:
		mask = np.ones(len(nedges), dtype=bool)
	istart = arg_nearest_geo(start, lons, lats)
	istop = arg_nearest_geo(stop, lons, lats)

	'''
	Create a distance array dist[] of size V and initialize all values to infinity
	since no paths are known yet. Set the distance of the source vertex to 0 and 
	insert it into the priority queue.
	'''
	dist = {i:np.inf for i in nedges[mask]}
	prev = {istart:None}
	dist[istart] = 0.0
	pq = []
	heapq.heappush(pq, (0, istart))

	while pq:
		'''
		While the priority queue is not empty, remove the vertex 
		with the smallest distance value.
		'''
		d, edge = heapq.heappop(pq)

		if edge == istop:
			break

		'''
		Check if the popped distance is greater than the recorded distance for this 
		vertex, it means this vertex has already been processed with a smaller 
		distance, so skip it
		'''
		if d > dist[edge]:
			continue

		# get neighbours of current edge
		verts = mesh.verticesOnEdge.values[edge] - 1
		ineighbours = mesh.edgesOnVertex.values[verts].ravel()
		ineighbours = [i - 1 for i in ineighbours if i > 0 and mask[i-1] and (i-1) != edge]

		point1 = (lats[edge], lons[edge])  # current cell
		for n in ineighbours:
			point2 = (lats[n], lons[n])  # neighbour

			'''
			For each neighbor n of currcell, check if the path through currcell gives a smaller 
			distance than the current dist[n]. If it does, update dist[n] = dist[currcell] + edge weight(d) 
			and push (dist[n], n) into the priority queue.	
			'''

			# Calculate distance in kilometers
			dgate = shapely.distance(gateline, geom.Point([lons[n], lats[n]]))/360*6000
			x = geodesic(point1, point2).km + d + dgate

			if x < dist[n]:
				dist[n] = x
				prev[n] = edge
				heapq.heappush(pq, (x, n))


	path = [istop]
	while prev[path[-1]] is not None:
		path.append(prev[path[-1]])


	return path


def sign_flip(unitvec, edges, mesh=None):

	if mesh is None:
		mesh = xr.load_dataset(MESHFILE_OCN)

	lons = np.degrees(mesh.lonCell.values)
	lats = np.degrees(mesh.latCell.values)
	lons[lons > 180] -= 360

	cellneighbours = mesh.cellsOnEdge.values[edges] - 1
	cellneighbours.sort(axis=1)
	mask = np.any(cellneighbours < 0, axis=1) # these cells border land

	tail = np.array([lons[cellneighbours[:,0]], lats[cellneighbours[:,0]]]).T
	tip = np.array([lons[cellneighbours[:,1]], lats[cellneighbours[:,1]]]).T

	vec = tip - tail
	dirn = vec @ unitvec
	fluxdir = np.where(dirn >= 0, 1, -1)
	fluxdir[mask] = 0

	return fluxdir, dirn

def edge_sign_for_direction(edgeIDs=None, mesh=None, target=(0., 1.)):

	if mesh is None:
		mesh = xr.open_dataset(MESHFILE_OCN)

	if edgeIDs is not None:
		theta = mesh.angleEdge[edgeIDs].values
	else:
		theta = mesh.angleEdge.values

	normal_x = -np.sin(theta)
	normal_y = np.cos(theta)

	target = np.asarray(target)
	dots = normal_x * target[0] + normal_y * target[1]

	return np.where(dots >= 0, 1, -1), dots


if __name__ == '__main__':
	# root = '/Volumes/Marvin/E3SM/'
	# mesh = xr.open_dataset(root + 'mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc')
	mesh = xr.load_dataset(MESHFILE_OCN)
	data = ''

	unitvec = np.array([1, 1])

	start, stop = [-26,69],[-23.492204845737092,66.12600770401113]

	path = djikstra_edges(start, stop, mesh)
	print(path)

	sign_flip(unitvec, path, mesh)