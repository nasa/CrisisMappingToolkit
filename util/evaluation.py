import ee
import threading
import functools

class WaitForResult(threading.Thread):
	def __init__(self, function, finished_function = None):
		threading.Thread.__init__(self)
		self.function = function
		self.finished_function = finished_function
		self.setDaemon(True)
		self.start()
	def run(self):
		self.finished_function(self.function())

def __evaluate_approach(result, ground_truth, region, fractional=False):
	if fractional:
		ground_truth = ground_truth.convolve(ee.Kernel.square(250, 'meters', True))
	
	# wrong = ground_truth.multiply(-1).add(1).min(result);
	# missed = ground_truth.subtract(result).max(0.0);
	correct = ground_truth.min(result);
	correct_sum = ee.data.getValue({'image': correct.stats(  30000, region, 'EPSG:4326').serialize(), 'fields': 'b1'})['properties']['b1']['values']['sum']
	result_sum = ee.data.getValue({'image': result.stats(    30000, region, 'EPSG:4326').serialize(), 'fields': 'b1'})['properties']['b1']['values']['sum']
	truth_sum = ee.data.getValue({'image': ground_truth.stats(30000, region, 'EPSG:4326').serialize(), 'fields': 'b1'})['properties']['b1']['values']['sum']
	return ((correct_sum / result_sum), (correct_sum / truth_sum))

def evaluate_approach(evaluation_function, result, ground_truth, region, fractional=False):
	WaitForResult(functools.partial(__evaluate_approach, result=result, ground_truth=ground_truth,
		region=region, fractional=fractional), evaluation_function)

