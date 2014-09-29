import ee
from util.mapclient_qt import centerMap, addToMap

import domains
from histogram import RadarHistogram

def __learning_threshold(domain, algorithm):
    training_domain = domains.get_radar_image(domains.TRAINING_DATA[domain.id])
    classifier = ee.apply('TrainClassifier', {'image': training_domain.image,
                            'subsampling' : 0.5,
                            'training_image' : domains.get_ground_truth(training_domain),
                            'training_band': 'b1',
                            'training_region' : training_domain.bounds,
                            'max_classification': 2,
                            'classifier_name': algorithm})
    classified = ee.call('ClassifyImage', domain.image, classifier).select(['classification'], ['b1']);
    return classified;

def decision_tree(domain):
    return __learning_threshold(domain, 'Cart')
def random_forests(domain):
    return __learning_threshold(domain, 'RifleSerialClassifier')
def svm(domain):
    return __learning_threshold(domain, 'Pegasos')


