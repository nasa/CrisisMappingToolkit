import matgen

# From Towards an automated SAR-based flood monitoring system:
# Lessons learned from two case studies by Matgen, Hostache et. al.
MATGEN    = 1

__ALGORITHMS = {
	MATGEN : ('Matgen Threshold', matgen.threshold, '00FFFF')
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

