import numpy as np
def rotate(l, k):
	'''
	Rotate list or array k steps to the left
	:param l: list or array
	:param k: number of rotational steps
	:return: rotated list or array
	>>> l = [1, 2, 3, 4, 5]
	>>> a = np.array(l)
	>>> rotate(l, 1)
	[2, 3, 4, 5, 1]
	>>> rotate(l, -1)
	[5, 1, 2, 3, 4]
	>>> rotate(a, 3)
	array([4, 5, 1, 2, 3])
	>>> rotate(a, -3)
	array([3, 4, 5, 1, 2])
	'''

	if isinstance(l, list):
		l_rot = l[k:] + l[:k]
	else:
		l_rot = np.concat((l[k:] , l[:k]))
	return l_rot
