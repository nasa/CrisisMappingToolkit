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

from cmt.domain import Domain
from cmt.modis.simple_modis_algorithms import *
from cmt.mapclient_qt import addToMap
from cmt.util.miscUtilities import safe_get_info
import cmt.modis.modis_utilities

"""
   Contains functions needed to implement an Adaboost algorithm using several of the
   simple MODIS classifiers.
"""


def _create_adaboost_learning_image(domain, b):
    '''Like _create_learning_image but using a lot of simple classifiers to feed into Adaboost'''
    
    #a = get_diff(b).select(['b1'], ['b1'])
    a = b['b1'].select(['sur_refl_b01'],                                                 ['b1'           ])
    a = a.addBands(b['b2'].select(['sur_refl_b02'],                                      ['b2'           ]))
    a = a.addBands(b['b2'].divide(b['b1']).select(['sur_refl_b02'],                      ['ratio'        ]))
    a = a.addBands(b['LSWI'].subtract(b['NDVI']).subtract(0.05).select(['sur_refl_b02'], ['LSWIminusNDVI']))
    a = a.addBands(b['LSWI'].subtract(b['EVI']).subtract(0.05).select(['sur_refl_b02'],  ['LSWIminusEVI' ]))
    a = a.addBands(b['EVI'].subtract(0.3).select(['sur_refl_b02'],                       ['EVI'          ]))
    a = a.addBands(b['LSWI'].select(['sur_refl_b02'],                                    ['LSWI'         ]))
    a = a.addBands(b['NDVI'].select(['sur_refl_b02'],                                    ['NDVI'         ]))
    a = a.addBands(b['NDWI'].select(['sur_refl_b01'],                                    ['NDWI'         ]))
    a = a.addBands(get_diff(b).select(['b1'],                                            ['diff'         ]))
    a = a.addBands(get_fai(b).select(['b1'],                                             ['fai'          ]))
    a = a.addBands(get_dartmouth(b).select(['b1'],                                       ['dartmouth'    ]))
    a = a.addBands(get_mod_ndwi(b).select(['b1'],                                        ['MNDWI'        ]))
    return a


def _find_adaboost_optimal_threshold(domains, images, truths, band_name, weights, splits):
    '''Binary search to find best threshold for this band'''
    
    EVAL_RESOLUTION = 250
    choices = []
    for i in range(len(splits) - 1):
        choices.append((splits[i] + splits[i+1]) / 2)
        
    domain_range = range(len(domains))
    best         = None
    best_value   = None
    for k in range(len(choices)):
        # Pick a threshold and count how many pixels fall under it across all the input images
        c = choices[k]
        errors = [safe_get_info(weights[i].multiply(images[i].select(band_name).lte(c).neq(truths[i])).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in range(len(images))]
        error  = sum(errors)
        #threshold_sums = [safe_get_info(weights[i].mask(images[i].select(band_name).lte(c)).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in domain_range]
        #flood_and_threshold_sum = sum(threshold_sums)
        #
        ##ts         = [truths[i].multiply(weights[i]).divide(flood_and_threshold_sum).mask(images[i].select(band_name).lte(c))              for i in domain_range]
        ##entropies1 = [-safe_get_info(ts[i].multiply(ts[i].log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['b1'] for i in domain_range]# H(Y | X <= c)
        ##ts         = [truths[i].multiply(weights[i]).divide(1 - flood_and_threshold_sum).mask(images[i].select(band_name).gt(c))           for i in domain_range]
        ##entropies2 = [-safe_get_info(ts[i].multiply(ts[i].log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['b1'] for i in domain_range]# H(Y | X > c)
        #
        ## Compute the sums of two entropy measures across all images
        #entropies1 = entropies2 = []
        #for i in domain_range:
        #    band_image     = images[i].select(band_name)
        #    weighted_truth = truths[i].multiply(weights[i])
        #    ts1            = weighted_truth.divide(    flood_and_threshold_sum).mask(band_image.lte(c)) # <= threshold
        #    ts2            = weighted_truth.divide(1 - flood_and_threshold_sum).mask(band_image.gt( c)) # >  threshold
        #    entropies1.append(-safe_get_info(ts1.multiply(ts1.log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['b1'])# H(Y | X <= c)
        #    entropies2.append(-safe_get_info(ts2.multiply(ts2.log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['b1'])# H(Y | X > c)
        #entropy1 = sum(entropies1)
        #entropy2 = sum(entropies2)
        #
        ## Compute the gain for this threshold choice
        #gain = (entropy1 * (    flood_and_threshold_sum)+
        #        entropy2 * (1 - flood_and_threshold_sum))
        print c, error
        if (best == None) or abs(0.5 - error) > abs(0.5 - best_value): # Record the maximum gain
            best       = k
            best_value = error
    
    # ??
    return (choices[best], best + 1, best_value)

def apply_classifier(image, band, threshold):
    '''Apply LTE threshold and convert to -1 / 1 (Adaboost requires this)'''
    return image.select(band).lte(threshold).multiply(2).subtract(1)

def get_adaboost_sum(domain, b, classifier = None):
    if classifier == None:
        # These are a set of known good computed values:  (Algorithm, Detection threshold, Weight)
        # learned from everything
        classifier = [(u'dartmouth', 0.31912828680782945, 1.395747066246276), (u'b2', 2259.24753825682, 0.7171153010990784), (u'MNDWI', 0.3466563830908956, 0.3030486480658588), (u'MNDWI', -0.6528042882273133, 0.19734829053102623), (u'b2', 465.7426147704591, -0.16212431026694754), (u'b2', 1592.0713073852298, 0.209564177038749), (u'diff', 15.215761821366016, -0.10014577994432751), (u'NDVI', -0.513638356572826, 0.14966005394881143), (u'diff', -144.3816112084063, -0.1456971131078552), (u'b2', 1258.4831919494345, 0.13265827613471332), (u'diff', -64.58292469352014, -0.0824191631492969), (u'NDVI', -0.6670832910016303, 0.11686769178883176), (u'diff', -224.18029772329245, -0.09589374745354495), (u'NDVI', -0.5903608237872282, 0.08685876311873326), (u'diff', -104.48226795096322, -0.07413904317173395), (u'b2', 1091.689134231537, 0.07696744378114175), (u'diff', -124.43193957968475, -0.05711643914226209), (u'b1', 1467.5741042345276, -0.07626024737632536), (u'b2', 2926.42376912841, 0.06950714577618346), (u'fai', 1103.6380636661445, -0.0644851346724657), (u'MNDWI', 0.0014298871363433718, 0.04942448753682089), (u'MNDWI', 0.17404313511361946, 0.06911618892100886), (u'diff', -84.53259632224169, -0.08692193122130303), (u'MNDWI', 0.26034975910225755, 0.0719781844974314), (u'NDWI', -0.375716597950533, 0.09276360724692449), (u'b2', 1925.659422821025, -0.05586529811930332), (u'MNDWI', 0.3035030710965766, 0.07283044201044488), (u'NDWI', -0.5220599892756219, 0.08136380018265663), (u'b1', 194.72231270358304, -0.068322599792114), (u'b2', 2092.4534805389226, -0.06345085588609038), (u'b2', 1425.2772496673322, 0.07990515239052665), (u'b2', 2175.850509397871, -0.062371519802571544), (u'b2', 2592.835653692615, 0.07510005616885976), (u'b2', 2217.5490238273455, -0.07136338189633898), (u'b2', 2759.6297114105128, 0.07859544955234492), (u'b2', 2238.398281042083, -0.06805487314746281), (u'b2', 1758.8653651031273, 0.08746154611595613), (u'b2', 2248.8229096494515, -0.06542687119482093), (u'b2', 2843.0267402694617, 0.06301067648753582), (u'b2', 236.16638389886896, 0.058695973047853865), (u'diff', -264.0796409807355, -0.06399538541469801), (u'b2', 121.37826846307387, 0.06579193323062736), (u'diff', -244.12996935201397, -0.05553729071254952), (u'MNDWI', 0.28192641509941707, 0.05072156990141125), (u'b2', 2254.0352239531358, -0.05461234230993423), (u'b2', 1842.2623939620762, 0.058931013996517435), (u'EVI', -0.006317971745977369, -0.052758468925946614), (u'b2', 178.77232618097142, 0.05534265992842654), (u'b2', 350.95449933466404, -0.04999103171155494), (u'b2', 207.46935503992017, 0.04655141306377409), (u'b2', 408.34855705256155, -0.06271139142354751), (u'b2', 150.07529732202264, 0.055797436150486296), (u'b2', 437.04558591151033, -0.070232209539299), (u'b2', 1008.2921053725883, 0.07035385158558453), (u'b2', 451.3941003409847, -0.04958911317829448), (u'b2', 164.42381175149703, 0.0640170306181281), (u'b2', 444.2198431262475, -0.05567053353074905), (u'b2', 171.59806896623422, 0.04892797113400034), (u'b2', 2801.328225839987, 0.04746893666184241), (u'b2', 2256.641381104978, -0.05051136402827347), (u'fai', 653.9033139866116, 0.05567763266111302), (u'b2', 440.6327145188789, -0.03530696027382092), (u'b2', 193.12084061044578, 0.04590443648040932), (u'MNDWI', 0.6918828790454478, 0.042430234062282565), (u'MNDWI', 0.5192696310681717, 0.05723985732531784), (u'b2', 157.24955453675983, 0.08655983712470855), (u'MNDWI', 0.6055762550568098, 0.058037513577376194), (u'NDWI', -0.44888829361307747, 0.06994763684621111), (u'MNDWI', 0.5624229430624907, 0.049256647449872494), (u'b2', 200.29509782518298, 0.06566289719639512), (u'MNDWI', 0.5408462870653312, 0.05700619823029491), (u'NDWI', -0.4854741414443497, 0.06101202049590758), (u'NDWI', -0.2293732066254442, -0.06384910607105221), (u'MNDWI', 0.551634615063911, 0.04116354794763669), (u'NDWI', -0.5037670653599858, 0.061715019463720135), (u'NDWI', -0.15620151096289978, -0.0455257763838119), (u'MNDWI', 0.546240451064621, 0.04895962846980223), (u'b2', 2257.944459680899, -0.048136126436163294), (u'b2', 2009.0564516799736, 0.03834574882351756), (u'NDWI', -0.3025449022879886, 0.04066687396682093), (u'NDWI', -0.11961566313162757, -0.03789216197859281), (u'b2', 442.4262788225632, -0.034998185059411754), (u'b2', 135.72678289254824, 0.051373275230902404), (u'b2', 441.5294966707211, -0.04441840333106898), (u'b2', 203.8822264325516, 0.038973367324459766), (u'MNDWI', 0.5489375330642661, 0.0413918293340524), (u'b2', 128.55252567781105, 0.03607680512297566), (u'b2', 441.08110559479996, -0.03516851671166877), (u'b2', 205.67579073623588, 0.03141466327547689), (u'b2', 2676.232682551564, 0.033786018201683955), (u'b2', 2258.5959989688595, -0.041040012776301364), (u'b2', 2780.4789686252498, 0.03064651061938805), (u'NDWI', -0.2659590544567164, 0.028764551204168116), (u'LSWIminusNDVI', -0.37354177362800933, -0.03578440992855484), (u'NDWI', -0.5129135273178038, 0.0325733920070771), (u'MNDWI', 0.4329630070795336, 0.030212235229533955), (u'b2', 206.572572888078, 0.0310925345029783), (u'MNDWI', 0.47611631907385266, 0.039035519462548476), (u'NDWI', -0.5174867582967129, 0.03569213599509797), (u'MNDWI', 0.4976929750710122, 0.03962475457128431), (u'b2', 124.96539707044246, 0.03867022069680135), (u'MNDWI', 0.4869046470724324, 0.029469011967394473), (u'MNDWI', 0.3250797270937361, -0.034096425013469045), (u'NDWI', -0.13790858704726366, -0.02444966428188256), (u'EVI', 0.4598391902116855, -0.026572751815372206), (u'diff', 1522.6552539404552, 0.02404140887707103), (u'diff', 928.5328809106829, -0.039880496519416754), (u'NDWI', -0.12876212508944562, -0.03210366544625396), (u'EVI', 0.22676060923285407, -0.023924611663183708), (u'b2', 2050.754966109448, 0.02461166045462737), (u'b2', 2258.2702293248794, -0.02908987661967497), (u'b2', 2071.604223324185, 0.025950513918970106), (u'b2', 2258.107344502889, -0.023032032486975468), (u'MNDWI', 0.4922988110717223, 0.026458034300998273), (u'fai', -1.7471535393819408, 0.03121701526540736), (u'EVI', 0.11022131874343835, -0.03129372953645484), (u'MNDWI', 0.2711380871008373, 0.027805632773075233), (u'NDWI', -0.5952316849381663, 0.02330356087019242), (u'b2', 440.8569100568394, -0.03309132592493886), (u'fai', 101.21070538384838, 0.0353271574429358), (u'b2', 440.7448122878592, -0.02634890500041707), (u'b1', 2017.7870521172638, -0.03283051121142451), (u'MNDWI', 0.48960172907207733, 0.02644326719520533), (u'NDWI', -0.6318175327694384, 0.031239945195141), (u'b2', 440.6887634033691, -0.028775285945758785), (u'b1', 1742.6805781758958, -0.02808047056876568), (u'b2', 2770.0543400178813, 0.03068011992420211), (u'b1', 1605.1273412052117, -0.022218212174410647), (u'MNDWI', 0.49095027007189984, 0.02432836950171423), (u'MNDWI', 0.31429139909515635, -0.029369703868232), (u'fai', 152.68963484546353, 0.02098259070819379), (u'EVI', 0.05195167349873049, -0.020904307140870906), (u'b1', 1536.3507227198697, -0.021669751108939296), (u'dartmouth', 1.5061250148057486, 0.02884211847370296), (u'fai', 126.95017011465595, 0.022140157847408665), (u'dartmouth', 1.3724218873381238, 0.019024795727913317), (u'NDWI', -0.6501104566850746, 0.023782834421931028), (u'dartmouth', 1.3055703236043112, 0.02230497078460576), (u'b1', 1501.9624134771987, -0.02361289905015126), (u'EVI', 0.08108649612108443, -0.020775409636097555), (u'MNDWI', 0.26574392310154743, 0.021340227300476654), (u'NDWI', -0.2842519783723525, 0.023090080272430938), (u'MNDWI', 0.2684410051011924, 0.01941178701297503), (u'NDWI', -0.27510551641453446, 0.01571455969994724), (u'NDWI', -0.46718121752871355, -0.018465755108615197), (u'NDWI', -0.5586458371068941, 0.022314314284390326), (u'NDWI', -0.4580347555708955, -0.01845154841999253), (u'fai', 1553.3728133456773, -0.01698363989763652), (u'dartmouth', 1.3389961054712174, 0.019744106506667823), (u'NDWI', -0.5769387610225302, 0.020590322433275743), (u'NDWI', -0.4534615245919865, -0.019205783920171575), (u'MNDWI', 0.31968556309444623, -0.01692289980149157), (u'b2', 2082.0288519315536, 0.01806848113773849), (u'b2', 2258.025902091894, -0.02086951030517275), (u'b2', 2087.241166235238, 0.019568132395932813), (u'b2', 2258.0666232973917, -0.01759581543782954), (u'b2', 2634.5341681220893, 0.018326066784634124), (u'b1', 1519.156568098534, -0.01979502323023555), (u'MNDWI', 0.4902759995719886, 0.018772733991657187), (u'fai', 878.7706888263781, -0.019558715071018792), (u'b2', 2196.699766612608, 0.01730677148647626), (u'EVI', -0.47247513370364025, 0.017987115905319706)]
        #classifier = [(u'b2', 1066.9529712504814, 1.5025586686710706), (u'NDWI', 0.14661938301755412, -0.21891567708553822), (u'dartmouth', 0.48798681823081197, -0.15726997982017618), (u'dartmouth', 0.6365457444743317, 0.18436960110357703), (u'LSWIminusNDVI', 0.3981981030948878, -0.10116535428832296), (u'fai', 355.7817695891917, -0.11241883192214887), (u'dartmouth', 0.7108252075960915, 0.16267637123701892), (u'diff', 528.9578763019633, -0.08056940174311174), (u'NDWI', -0.2608919987915783, -0.0662560864223818), (u'diff', 945.4263065720343, -0.06468547496541238), (u'LSWI', 0.10099215524728983, 0.06198258456041972), (u'LSWI', 0.4036574931704132, -0.13121098919819557), (u'NDVI', -0.11873600974959503, -0.06877321671018986), (u'dartmouth', 0.6736854760352116, 0.058740830970174365), (u'diff', 737.1920914369988, -0.07784405443757562), (u'fai', 637.5040900767088, 0.06383077739328656), (u'LSWI', 0.2523248242088515, -0.06159092845229366), (u'diff', 841.3091990045166, -0.03543296624866381), (u'NDWI', -0.0571363078870121, -0.033363758883119425), (u'NDWI', -0.1590141533392952, -0.04351253722452526), (u'LSWIminusNDVI', -0.021934005871228984, 0.05405714553564335), (u'LSWIminusNDVI', -0.23200006035428739, -0.05945459980438702), (u'diff', 893.3677527882754, -0.04401238934808345), (u'LSWIminusNDVI', -0.33703308759581657, -0.03270875405530488), (u'LSWIminusEVI', -0.06154292961108998, 0.03531804439403144), (u'LSWIminusEVI', -0.6658933123079813, 0.049495070534741545), (u'LSWIminusNDVI', -0.284516573975052, -0.0652646748372963), (u'LSWI', 0.17665848972807066, 0.035535330348720445), (u'LSWIminusEVI', -0.9680685036564269, 0.03385062752160848), (u'dartmouth', 0.6551156102547716, 0.0255425888326403), (u'diff', 867.338475896396, -0.029608603189018888), (u'dartmouth', 0.6644005431449915, 0.031453944391964694), (u'b2', 1597.1343803620828, -0.032483706321846446), (u'b2', 1862.2250849178836, 0.11634887737020584), (u'diff', 880.3531143423356, -0.0759983094592842), (u'LSWIminusNDVI', -0.2582583171646697, -0.03107758927177279), (u'LSWI', 0.13882532248768026, 0.028397075633745206), (u'fai', 496.64292983295024, -0.033912739368973946), (u'EVI', -0.050627832167768394, 0.01948538465509801), (u'MNDWI', -0.314992942152472, -0.028318432983029183), (u'LSWIminusEVI', -0.8169809079822041, 0.017475858734323002), (u'EVI', 0.393374399578999, 0.02621185146352496), (u'LSWIminusEVI', -0.7414371101450927, 0.024903163093461297), (u'LSWIminusNDVI', -0.27138744556986083, -0.026585292504625754), (u'MNDWI', 0.008416570589500877, -0.020089052435031476), (u'MNDWI', 0.17012132696048732, 0.09013852992730866), (u'MNDWI', 0.2509737051459805, 0.07860221464785291), (u'NDWI', -0.20995307606543676, 0.12420114244297033), (u'MNDWI', 0.2105475160532339, 0.058700913757496295), (u'diff', 873.8457951193658, -0.06536366152935989), (u'MNDWI', 0.2307606105996072, 0.05774578894612022), (u'LSWIminusNDVI', -0.3107748307854343, 0.06324847449109565), (u'MNDWI', 0.24086715787279384, 0.05222308145337588), (u'NDVI', 0.2321600980908248, -0.044636312533643016), (u'b1', 840.9896718836895, -0.04386969870931187), (u'b1', 431.64304441090013, 0.10652512181225056), (u'MNDWI', 0.24592043150938717, 0.04629035855680991)]
        #classifier = [(u'b2', 1066.9529712504814, 1.5025586686710706), (u'NDWI', 0.14661938301755412, -0.21891567708553822), (u'dartmouth', 0.48798681823081197, -0.15726997982017618), (u'dartmouth', 0.6365457444743317, 0.18436960110357703), (u'LSWIminusNDVI', 0.3981981030948878, -0.10116535428832296), (u'fai', 355.7817695891917, -0.11241883192214887), (u'dartmouth', 0.7108252075960915, 0.16267637123701892), (u'diff', 528.9578763019633, -0.08056940174311174), (u'NDWI', -0.2608919987915783, -0.0662560864223818), (u'diff', 945.4263065720343, -0.06468547496541238), (u'LSWI', 0.10099215524728983, 0.06198258456041972), (u'LSWI', 0.4036574931704132, -0.13121098919819557), (u'NDVI', -0.11873600974959503, -0.06877321671018986), (u'dartmouth', 0.6736854760352116, 0.058740830970174365), (u'diff', 737.1920914369988, -0.07784405443757562), (u'fai', 637.5040900767088, 0.06383077739328656), (u'LSWI', 0.2523248242088515, -0.06159092845229366), (u'diff', 841.3091990045166, -0.03543296624866381), (u'NDWI', -0.0571363078870121, -0.033363758883119425), (u'NDWI', -0.1590141533392952, -0.04351253722452526), (u'LSWIminusNDVI', -0.021934005871228984, 0.05405714553564335), (u'LSWIminusNDVI', -0.23200006035428739, -0.05945459980438702), (u'diff', 893.3677527882754, -0.04401238934808345), (u'LSWIminusNDVI', -0.33703308759581657, -0.03270875405530488), (u'LSWIminusEVI', -0.06154292961108998, 0.03531804439403144), (u'LSWIminusEVI', -0.6658933123079813, 0.049495070534741545), (u'LSWIminusNDVI', -0.284516573975052, -0.0652646748372963), (u'LSWI', 0.17665848972807066, 0.035535330348720445), (u'LSWIminusEVI', -0.9680685036564269, 0.03385062752160848), (u'dartmouth', 0.6551156102547716, 0.0255425888326403), (u'diff', 867.338475896396, -0.029608603189018888), (u'dartmouth', 0.6644005431449915, 0.031453944391964694), (u'b2', 1597.1343803620828, -0.032483706321846446), (u'b2', 1862.2250849178836, 0.11634887737020584), (u'diff', 880.3531143423356, -0.0759983094592842), (u'LSWIminusNDVI', -0.2582583171646697, -0.03107758927177279), (u'LSWI', 0.13882532248768026, 0.028397075633745206), (u'fai', 496.64292983295024, -0.033912739368973946), (u'EVI', -0.050627832167768394, 0.01948538465509801), (u'MNDWI', -0.314992942152472, -0.028318432983029183), (u'LSWIminusEVI', -0.8169809079822041, 0.017475858734323002), (u'EVI', 0.393374399578999, 0.02621185146352496), (u'LSWIminusEVI', -0.7414371101450927, 0.024903163093461297), (u'LSWIminusNDVI', -0.27138744556986083, -0.026585292504625754), (u'MNDWI', 0.008416570589500877, -0.020089052435031476), (u'MNDWI', 0.17012132696048732, 0.09013852992730866), (u'MNDWI', 0.2509737051459805, 0.07860221464785291), (u'NDWI', -0.20995307606543676, 0.12420114244297033), (u'MNDWI', 0.2105475160532339, 0.058700913757496295), (u'diff', 873.8457951193658, -0.06536366152935989), (u'MNDWI', 0.2307606105996072, 0.05774578894612022), (u'LSWIminusNDVI', -0.3107748307854343, 0.06324847449109565), (u'MNDWI', 0.24086715787279384, 0.05222308145337588), (u'NDVI', 0.2321600980908248, -0.044636312533643016), (u'b1', 840.9896718836895, -0.04386969870931187), (u'b1', 431.64304441090013, 0.10652512181225056), (u'MNDWI', 0.24592043150938717, 0.04629035855680991), (u'LSWI', 0.15774190610787547, 0.036843417853100205), (u'dartmouth', 0.6690430095901015, -0.03212513821387442), (u'NDVI', 0.40760815201103473, 0.02804021082285224), (u'NDWI', -0.23542253742850755, 0.03186450807020461), (u'MNDWI', 0.24339379469109051, 0.02862992753172424), (u'NDWI', -0.24815726811004293, 0.02759848052686244), (u'MNDWI', 0.24844706832768382, 0.028767258391409676), (u'NDWI', -0.2545246334508106, 0.023920554168868943), (u'MNDWI', 0.24971038673683216, 0.019530411666440373), (u'diff', 854.3238374504563, -0.017420419212339854), (u'fai', 567.0735099548295, 0.017319660531085516), (u'dartmouth', 0.6667217763675466, -0.01514266152502469), (u'EVI', 0.6153755154523827, 0.015811343455312793), (u'NDWI', -0.2577083161211945, 0.016367445780215994), (u'EVI', 0.5043749575156908, 0.021397992947186542), (u'b1', 636.3163581472949, 0.017679309609453388), (u'b1', 533.9797012790975, 0.052480869491753616), (u'LSWIminusNDVI', -0.2779520097724564, -0.025442696952484856), (u'b2', 1729.6797326399833, -0.02851944225149679), (u'b2', 1663.407056501033, -0.031126464288873154), (u'EVI', 0.4488746785473449, 0.02559032221555967), (u'b1', 585.1480297131961, 0.02206908624237135), (u'diff', 789.2506452207576, -0.016514173533494117), (u'fai', 531.8582198938899, 0.023477946621113858), (u'b1', 610.7321939302456, -0.012825485385474514), (u'diff', 815.2799221126371, -0.01149904087913293), (u'MNDWI', 0.249078727532258, 0.01515097935650566), (u'NDWI', -0.2593001574563864, 0.01083047626473297), (u'LSWIminusNDVI', -0.28123429187375415, -0.012898006552737352), (u'NDWI', -0.2600960781239824, 0.011987178213307102), (u'LSWIminusNDVI', -0.2828754329244031, -0.010878187527895457), (u'NDWI', -0.26049403845778035, 0.010250229825634711), (u'LSWIminusNDVI', -0.2820548623990786, -0.009697140264622033), (u'LSWIminusEVI', -0.703665211226537, 0.00944101306441367), (u'NDVI', 0.3198841250509298, 0.009414806179635152), (u'NDVI', 0.36374613853098225, 0.013660042816894051), (u'NDWI', -0.2606930186246793, 0.01871187931420404), (u'NDVI', 0.3856771452710085, 0.02480484723062163), (u'ratio', 1.9036367764380129, -0.025236576927683663), (u'ratio', 2.8194639395525054, 0.02624098822658558), (u'LSWIminusEVI', -0.6847792617672592, 0.0235627702128661), (u'ratio', 2.361550357995259, 0.017108797919490024), (u'diff', 828.2945605585769, -0.01756108404460174)]

        
    test_image = _create_adaboost_learning_image(domain, b)
    total = ee.Image(0).select(['constant'], ['b1'])
    for c in classifier:
      total = total.add(test_image.select(c[0]).lte(c[1]).multiply(2).subtract(1).multiply(c[2]))
    return total

def adaboost(domain, b, classifier = None):
    '''Run Adaboost classifier'''
    total = get_adaboost_sum(domain, b, classifier)
    return total.gte(0.0) # Just threshold the results at zero (equal chance of flood / not flood)

def adaboost_dem(domain, b, classifier = None):
    
    # Get raw adaboost output
    total = get_adaboost_sum(domain, b, classifier)
    addToMap(total, {'min': -10, 'max': 10}, 'raw ADA', False)
    
    # Convert this range of values into a zero to one probability scale
    # - These bounds represent where the probability plateaus.  Thes plateaus are
    #    usually not at 0% or 100% !!
    MIN_SUM = -6.0
    MAX_SUM =  2.0
    val_range = MAX_SUM - MIN_SUM
    
    fraction = total.subtract(ee.Image(MIN_SUM)).divide(ee.Image(val_range)).clamp(0.0, 1.0)
    addToMap(fraction, {'min': 0, 'max': 1}, 'fraction', False)
    return cmt.modis.modis_utilities.apply_dem(domain, fraction)

def __compute_threshold_ranges(training_domains, training_images, water_masks, bands):
    '''For each band, find lowest and highest fixed percentiles among the training domains.'''
    LOW_PERCENTILE  = 20
    HIGH_PERCENTILE = 100
    EVAL_RESOLUTION = 250
    
    band_splits = dict()
    for band_name in bands: # Loop through each band (weak classifier input)
        split = None
        print 'Computing threshold ranges for: ' + band_name
      
        mean = 0
        for i in range(len(training_domains)): # Loop through all input domains
            # Compute the low and high percentiles for the data in the training image
            masked_input_band = training_images[i].select(band_name).mask(water_masks[i])
            ret = safe_get_info(masked_input_band.reduceRegion(ee.Reducer.percentile([LOW_PERCENTILE, HIGH_PERCENTILE], ['s', 'b']), training_domains[i].bounds, EVAL_RESOLUTION))
            s   = [ret[band_name + '_s'], ret[band_name + '_b']] # Extract the two output values
            mean += compute_binary_threshold(training_images[i].select([band_name], ['b1']), water_masks[i], training_domains[i].bounds)
            
            if split == None: # True for the first training domain
                split = s
            else: # Track the minimum and maximum percentiles for this band
                split[0] = min(split[0], s[0])
                split[1] = max(split[1], s[1])
        mean = mean / len(training_domains)
            
        # For this band: bound by lowest percentile and maximum percentile, start by evaluating mean
        band_splits[band_name] = [split[0], split[0] + (mean - split[0]) / 2, mean + (split[1] - mean) / 2, split[1]]
    return band_splits

def adaboost_learn(domain, b):
    '''Train Adaboost classifier'''
    
    EVAL_RESOLUTION = 250

    # Load inputs for this domain and preprocess
    #all_problems      = ['kashmore_2010_8.xml', 'mississippi_2011_5.xml', 'mississippi_2011_6.xml', 'new_orleans_2005_9.xml', 'sf_bay_area_2011_4.xml']
    #all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
    #training_domains  = [domain.unflooded_domain for domain in all_domains[:-1]] + [all_domains[-1]] # SF is unflooded
    all_problems      = ['unflooded_mississippi_2010.xml', 'unflooded_new_orleans_2004.xml', 'sf_bay_area_2011_4.xml']
    all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
    training_domains  = all_domains
    water_masks       = [get_permanent_water_mask() for d in training_domains]
    training_images   = [_create_adaboost_learning_image(d, compute_modis_indices(d)) for d in training_domains]
    
    # add pixels in flood permanent water masks to training
    #training_domains.extend(all_domains)
    #water_masks.extend([get_permanent_water_mask() for d in all_domains])
    #training_images.append([_create_adaboost_learning_image(domain, compute_modis_indices(domain)).mask(get_permanent_water_mask()) for domain in all_domains])
    
    transformed_masks = [water_mask.multiply(2).subtract(1) for water_mask in water_masks]

    bands             = safe_get_info(training_images[0].bandNames())
    print 'Computing threshold ranges.'
    band_splits = __compute_threshold_ranges(training_domains, training_images, water_masks, bands)
    counts = [safe_get_info(training_images[i].select('diff').reduceRegion(ee.Reducer.count(), training_domains[i].bounds, 250))['diff'] for i in range(len(training_images))]
    count = sum(counts)
    weights = [ee.Image(1.0 / count) for i in training_images] # Each input pixel in the training images has an equal weight
    
    # Initialize for pre-existing partially trained classifier
    full_classifier = []
    for (c, t, alpha) in full_classifier:
        band_splits[c].append(t)
        band_splits[c] = sorted(band_splits[c])
        total = 0
        for i in range(len(training_images)):
            weights[i] = weights[i].multiply(apply_classifier(training_images[i], c, t).multiply(transformed_masks[i]).multiply(-alpha).exp())
            total += safe_get_info(weights[i].reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, EVAL_RESOLUTION))['constant']
        for i in range(len(training_images)):
            weights[i] = weights[i].divide(total)
    
    # Apply weak classifiers to the input test image
    test_image = _create_adaboost_learning_image(domain, b)
    
    # learn 100 weak classifiers
    while len(full_classifier) < 1000:
        best = None
        for band_name in bands: # For each weak classifier
            # Find the best threshold that we can choose
            (threshold, ind, error) = _find_adaboost_optimal_threshold(training_domains, training_images, water_masks, band_name, weights, band_splits[band_name])
            
            # Compute the sum of weighted classification errors across all of the training domains using this threshold
            #errors = [safe_get_info(weights[i].multiply(training_images[i].select(band_name).lte(threshold).neq(water_masks[i])).reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in range(len(training_images))]
            #error  = sum(errors)
            print '%s found threshold %g with error %g' % (band_name, threshold, error)
            
            # Record the band/threshold combination with the highest abs(error)
            if (best == None) or (abs(0.5 - error) > abs(0.5 - best[0])): # Classifiers that are always wrong are also good with negative alpha
                best = (error, band_name, threshold, ind)
        
        # add an additional split point to search between for thresholds
        band_splits[best[1]].insert(best[3], best[2])
      
        print 'Using %s < %g. Error %g.' % (best[1], best[2], best[0])
        alpha      = 0.5 * math.log((1 - best[0]) / best[0])
        classifier = (best[1], best[2], alpha)
        full_classifier.append(classifier)
        
        # update the weights
        weights = [weights[i].multiply(apply_classifier(training_images[i], classifier[0], classifier[1]).multiply(transformed_masks[i]).multiply(-alpha).exp()) for i in range(len(training_images))]
        totals  = [safe_get_info(weights[i].reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in range(len(training_images))]
        total   = sum(totals)
        weights = [w.divide(total) for w in weights]
        print full_classifier


# The results from this don't look great!
#
#import modis_utilities
#def adaboost_dem_learn(classifier = None):
#    '''Train Adaboost classifier'''
#    
#    EVAL_RESOLUTION = 250
#
#    # Load inputs for this domain and preprocess
#    all_problems      = ['kashmore_2010_8.xml', 'mississippi_2011_5.xml', 'mississippi_2011_6.xml', 'new_orleans_2005_9.xml', 'sf_bay_area_2011_4.xml']
#    all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
#    training_domains  = [domain.unflooded_domain for domain in all_domains[:-1]] + [all_domains[-1]] # SF is unflooded
#    water_masks       = [get_permanent_water_mask() for d in training_domains]
#    
#    THRESHOLD_INTERVAL = 1
#    TARGET_PERCENTAGE  = 0.95
#    
#    print 'Computing thresholds'
#    
#    # Loop through each of the raw result images
#    for (truth_image, train_domain, name) in zip(water_masks, training_domains, all_problems):
#
#        # Apply the Adaboost computation to each training image and get the raw results
#        b = modis_utilities.compute_modis_indices(train_domain)
#        sum_image = get_adaboost_sum(train_domain, b, classifier)
#        #addToMap(sum_image, {'min': -10, 'max': 10}, 'raw ADA', False)
#        print '================================'
#        print name
#
#        # For each threshold level above zero, how likely is the pixel to be actually flooded?
#        curr_threshold = -5.0
#        percentage = 0
#        #while percentage < TARGET_PERCENTAGE:
#        while curr_threshold < 5.0:
#            
#            curr_results = sum_image.gte(curr_threshold)
#            #addToMap(curr_results, {'min': 0, 'max': 1}, str(curr_threshold), False)
#            #addToMap(truth_image, {'min': 0, 'max': 1}, 'truth', False)
#            sum_correct  = safe_get_info(curr_results.multiply(truth_image).reduceRegion(ee.Reducer.sum(), train_domain.bounds, EVAL_RESOLUTION))['b1']
#            sum_total    = safe_get_info(curr_results.reduceRegion(ee.Reducer.sum(), train_domain.bounds, EVAL_RESOLUTION))['b1']
#            #print sum_correct
#            if sum_total > 0:
#                percentage   = sum_correct / sum_total
#                print str(curr_threshold) +': '+ str(sum_total) + ' --> '+ str(percentage)
#            else:
#                break
#            curr_threshold += THRESHOLD_INTERVAL
#        
#    raise Exception('DEBUG')
#
#        # For each threshold level below zero, how likely is the pixel to be actually dry?
        



