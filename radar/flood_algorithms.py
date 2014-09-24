import matgen
import learning

# From Towards an automated SAR-based flood monitoring system:
# Lessons learned from two case studies by Matgen, Hostache et. al.
MATGEN         = 1
RANDOM_FORESTS = 2
DECISION_TREE  = 3
SVM            = 4

__ALGORITHMS = {
	MATGEN : ('Matgen Threshold', matgen.threshold, '00FFFF'),
	RANDOM_FORESTS : ('Random Forests', learning.random_forests, 'FFFF00'),
	DECISION_TREE  : ('Decision Tree', learning.decision_tree, 'FF00FF'),
	SVM : ('SVM', learning.svm, '00AAFF')
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

