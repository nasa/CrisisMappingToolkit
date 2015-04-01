# -----------------------------------------------------------------------------
# Copyright * 2014, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration. All
# rights reserved.
#
# The Crisis Mapping Toolkit (CMT) v1 platform is licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# -----------------------------------------------------------------------------

import ee
from cmt.mapclient_qt import centerMap, addToMap

from histogram import RadarHistogram


'''
Use Earth Engine's classifier tool for water detection.
'''

def __learning_threshold(domain, algorithm):
    
    training_domain = None
    if domain.training_domain:
        training_domain = domain.training_domain
    elif domain.unflooded_domain:
        training_domain = domain.unflooded_domain
    if not training_domain:
        raise Exception('Cannot use learning algorithms without a training image defined by the domain!')
    classifier = ee.apply('TrainClassifier', {'image': training_domain.get_radar().image,
                            'subsampling'       : 0.07,
                            'training_image'    : training_domain.ground_truth,
                            'training_band'     : 'b1',
                            'training_region'   : training_domain.bounds,
                            'max_classification': 2,
                            'classifier_name'   : algorithm})
    classified = ee.call('ClassifyImage', domain.get_radar().image, classifier).select(['classification'], ['b1']);
    return classified;

def decision_tree(domain):
    '''Use "Cart" method: Classification and Regression Tree'''
    return __learning_threshold(domain, 'Cart')
def random_forests(domain):
    '''Use "RifleSerialClassifier" method: A Random Forest technique'''
    return __learning_threshold(domain, 'RifleSerialClassifier')
def svm(domain):
    '''Use "Pegasos" method: Primal Estimated sub-GrAdient SOlver for SVM'''
    return __learning_threshold(domain, 'Pegasos')


