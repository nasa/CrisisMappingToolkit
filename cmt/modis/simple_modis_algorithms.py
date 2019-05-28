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
import math

from cmt.mapclient_qt import addToMap
from cmt.util.miscUtilities import safe_get_info
import modis_utilities

'''
Contains implementations of several simple MODIS-based flood detection algorithms.
'''

#==============================================================


def dem_threshold(domain, b):
    '''Just use a height threshold on the DEM!'''

    heightLevel = float(domain.algorithm_params['dem_threshold'])
    dem         = domain.get_dem().image
    return dem.lt(heightLevel).select(['elevation'], ['b1'])


#==============================================================

def evi(domain, b):
    '''Simple EVI based classifier'''
    #no_clouds = b['b3'].lte(2100).select(['sur_refl_b03'], ['b1'])
    criteria1 = b['EVI'].lte(0.3).And(b['LSWI'].subtract(b['EVI']).gte(0.05)).select(['sur_refl_b02'], ['b1'])
    criteria2 = b['EVI'].lte(0.05).And(b['LSWI'].lte(0.0)).select(['sur_refl_b02'], ['b1'])
    #return no_clouds.And(criteria1.Or(criteria2))
    return criteria1.Or(criteria2)

def xiao(domain, b):
    '''Method from paper: Xiao, Boles, Frolking, et. al. Mapping paddy rice agriculture in South and Southeast Asia using
                          multi-temporal MODIS images, Remote Sensing of Environment, 2006.
                          
        This method implements a very simple decision tree from several standard MODIS data products.
        The default constants were tuned for (wet) rice paddy detection.
    '''
    return b['LSWI'].subtract(b['NDVI']).gte(0.05).Or(b['LSWI'].subtract(b['EVI']).gte(0.05)).select(['sur_refl_b02'], ['b1']);


#==============================================================

def get_diff(b):
    '''Just the internals of the difference method'''
    return b['b2'].subtract(b['b1']).select(['sur_refl_b02'], ['b1'])

def diff_learned(domain, b):
    '''modis_diff but with the threshold calculation included (training image required)'''
    if domain.unflooded_domain == None:
        print('No unflooded training domain provided.')
        return None
    unflooded_b = modis_utilities.compute_modis_indices(domain.unflooded_domain)
    water_mask  = modis_utilities.get_permanent_water_mask()
    
    threshold = modis_utilities.compute_binary_threshold(get_diff(unflooded_b), water_mask, domain.bounds)
    return modis_diff(domain, b, threshold)

def modis_diff(domain, b, threshold=None):
    '''Compute (b2-b1) < threshold, a simple water detection index.
    
       This method may be all that is needed in cases where the threshold can be hand tuned.
    '''
    if threshold == None: # If no threshold value passed in, load it based on the data set.
        threshold = float(domain.algorithm_params['modis_diff_threshold'])
    return get_diff(b).lte(threshold)

#==============================================================

def get_dartmouth(b):
    A = 500
    B = 2500
    return b['b2'].add(A).divide(b['b1'].add(B)).select(['sur_refl_b02'], ['b1'])

def dart_learned(domain, b):
    '''The dartmouth method but with threshold calculation included (training image required)'''
    if domain.unflooded_domain == None:
        print('No unflooded training domain provided.')
        return None
    unflooded_b = modis_utilities.compute_modis_indices(domain.unflooded_domain)
    water_mask  = modis_utilities.get_permanent_water_mask()
    threshold   = modis_utilities.compute_binary_threshold(get_dartmouth(unflooded_b), water_mask, domain.bounds)
    return dartmouth(domain, b, threshold)

def dartmouth(domain, b, threshold=None):
    '''A flood detection method from the Dartmouth Flood Observatory.
    
        This method is a refinement of the simple b2-b1 detection method.
    '''
    if threshold == None:
        threshold = float(domain.algorithm_params['dartmouth_threshold'])
    return get_dartmouth(b).lte(threshold)

#==============================================================

def get_mod_ndwi(b):
    return b['b6'].subtract(b['b4']).divide(b['b4'].add(b['b6'])).select(['sur_refl_b06'], ['b1'])

def mod_ndwi_learned(domain, b):
    if domain.unflooded_domain == None:
        print('No unflooded training domain provided.')
        return None
    unflooded_b = modis_utilities.compute_modis_indices(domain.unflooded_domain)
    water_mask  = modis_utilities.get_permanent_water_mask()
    threshold   = modis_utilities.compute_binary_threshold(get_mod_ndwi(unflooded_b), water_mask, domain.bounds)
    return mod_ndwi(domain, b, threshold)

def mod_ndwi(domain, b, threshold=None):
    if threshold == None:
        threshold = float(domain.algorithm_params['mod_ndwi_threshold'])
    return get_mod_ndwi(b).lte(threshold)

#==============================================================

def get_fai(b):
    '''Just the internals of the FAI method'''
    return b['b2'].subtract(b['b1'].add(b['b5'].subtract(b['b1']).multiply((859.0 - 645) / (1240 - 645)))).select(['sur_refl_b02'], ['b1'])

def fai_learned(domain, b):
    if domain.unflooded_domain == None:
        print('No unflooded training domain provided.')
        return None
    unflooded_b = modis_utilities.compute_modis_indices(domain.unflooded_domain)
    water_mask  = modis_utilities.get_permanent_water_mask()
    
    threshold = modis_utilities.compute_binary_threshold(get_fai(unflooded_b), water_mask, domain.bounds)
    return fai(domain, b, threshold)

def fai(domain, b, threshold=None):
    ''' Floating Algae Index. Method from paper: Feng, Hu, Chen, Cai, Tian, Gan,
    Assessment of inundation changes of Poyang Lake using MODIS observations
    between 2000 and 2010. Remote Sensing of Environment, 2012.
    '''
    if threshold == None:
        threshold = float(domain.algorithm_params['fai_threshold'])
    return get_fai(b).lte(threshold)
