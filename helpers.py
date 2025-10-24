import numpy as np

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

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

def dt64_y_m_d(date):
	years = date.astype('datetime64[Y]').astype(int) + 1970
	months = date.astype('datetime64[M]').astype(int) % 12 + 1
	days = date - date.astype('datetime64[M]') + 1
	return years, months, days

def normalize(x):

	norm = (x - np.mean(x)) / np.std(x)
	return norm