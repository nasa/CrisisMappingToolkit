import ee

THRESHOLD    = 1

def threshold(domain):
	vv = domain.image.select(['vv'], ['b1'])
	hh = domain.image.select(['hh'], ['b1'])
	hv = domain.image.select(['hv'], ['b1'])
	return vv.lte(3000).Or(hh.lte(3000))

__ALGORITHMS = {
	THRESHOLD : ('Threshold', threshold, '00FFFF')
}

def detect_flood(image, algorithm):
	try:
		approach = __ALGORITHMS[algorithm]
	except:
		return None
	return approach[1](image)

def get_algorithm_name(algorithm):
	try:
		return __ALGORITHMS[algorithm][0]
	except:
		return None

def get_algorithm_color(algorithm):
	try:
		return __ALGORITHMS[algorithm][2]
	except:
		return None

