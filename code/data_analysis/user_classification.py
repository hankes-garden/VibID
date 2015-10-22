# -*- coding: utf-8 -*-
"""
This script provide various functions to help extraction of features and labels
from data. Also, it provides interface to train & test differfent models.

Note: this script can only handle the new version data, which contains multiple responses in single data, and only consists of data from one accelerometer

@author: jason
"""
import bp_filter
import multiple_data as md
import single_data as sd
import dataset as ds

import pandas as pd
import numpy as np
import sys
import operator

import sklearn.preprocessing as prepro
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# name of vibrations in a single response
SEGMENT = "segment"

# name of statistics
MEAN = "mean"
STD = "std"
ONE_QUARTER = "25p"
TWO_QUARTER = "50p"
THREE_QUARTER = "75p"
MIN = "min"
MAX = "max"


LABEL = 'label'
MODEL_ACCURACY = "accuracy"
MODEL_FEATURE_IMPORTANCE = "feature_importance"
MODEL_PRECISION = "precision"
MODEL_PRECISION_MEAN = "precision_mean"
MODEL_PRECISION_STD = "precision_std"
MODEL_RECALL = "recall"
MODEL_RECALL_MEAN = "recall_mean"
MODEL_RECALL_STD = "recall_std"
MODEL_F1 = "F1"
MODEL_CONFUSION_MATRIX = "confusion_matrix"


def extractSegmentFeatures(strCol, nSegID, arrSeg):
    """
        Extrac statistic for each segment
        
        Parameters:
        ----
        strCol: 
            the axis name of which this segment belongs to
        nSegID: 
            the id of current segment
        arrSeg: 
            data of segment
        
        Returns:
        ----
        a dict of statistics, each of which has a key like: "x0_s1_std"
    """
    dcRet = {}
    dfStat = pd.Series(arrSeg).describe()
    
    # in case there are something wrong
    dfStat.replace([np.inf, -np.inf], np.nan, inplace=True)
    dfStat.fillna(0.0, inplace=True)

    dcRet["%s_s%d_%s" % (strCol, nSegID, STD)] = dfStat['std']
    dcRet["%s_s%d_%s" % (strCol, nSegID, MIN)] = dfStat['min']
    dcRet["%s_s%d_%s" % (strCol, nSegID, ONE_QUARTER)] = dfStat['25%']
    dcRet["%s_s%d_%s" % (strCol, nSegID, TWO_QUARTER)] = dfStat['50%']
    dcRet["%s_s%d_%s" % (strCol, nSegID, THREE_QUARTER)] = dfStat['75%']
    dcRet["%s_s%d_%s" % (strCol, nSegID, MAX)] = dfStat['max']

    return dcRet

def extractAxisFeature(dfData, 
                       strDataName, 
                       strCol, 
                       dSamplingFreq, 
                       nReponses,
                       nSegmentPerRespsonse, 
                       dVibrationDuration,
                       dIntervalDuration,
                       dRestDuration,
                       nFreqLowCut=20, 
                       nFreqHighCut=120):
    """
        Extract feature for single-axis data as follows:
        1. divide this data into multiple responses;
        2. divide each response into several segments;
        3. extract features from all segment;
        4. combine all features together;
        
        Parameters:
        ----
        dfData: 
            the data frame
        strDataName:
            name of data
        strCol: 
            axis name
        dSamplingFreq: 
            sampling frequency of data
        nResponse: 
            number of responses in single-axis data
        nSegmentPerRespsonse: 
            number of excitation segment per response
        dVibrationDuration: 
            duration of vibration (seconds)
        dIntervalDuration: 
            interval duration btw segments (seconds)
        dRestDuration: 
            the static duration btw responses
        nFreqLowCut: 
            the low cutting frequency of bandpass filter
        nFreqHighCut: 
            the high cutting frequency of bandpass filter 
        
        Returns:
        ----
        dcResponseFeature: 
            a dict of key-value pairs like: 
                     {'r1': {
                             'x0_s1_std': 0.85, 
                             'y0_s5_min': 4.72,
                             ...
                             }
                     }
    """
    arrData = dfData[strCol].values
    
    # bandpass filter to select spectrum of interest
    arrFiltered = bp_filter.butter_bandpass_filter(arrData, 
                                                   lowcut=nFreqLowCut,
                                                   highcut=nFreqHighCut,
                                                   fs=dSamplingFreq,
                                                   order=7)
                         
    # find reponses & segements in each response
    lsResponses, arrResponseEndIndex, \
    arrLocalPeakRefVal = sd.splitData(arrFiltered, 
                                      dSamplingFreq, 
                                      nReponses,
                                      nSegmentPerRespsonse,
                                      dVibrationDuration,
                                      dIntervalDuration,
                                      dRestDuration)
                                     
    dcResponseFeature = {}
    for nRespID, lsSegments in enumerate(lsResponses):
        dcFeaturePerResponse = dcResponseFeature.get( \
            "%s_r%d"% (strDataName, nRespID), None)
        if(dcFeaturePerResponse == None):
            dcFeaturePerResponse = {}
            dcResponseFeature["%s_r%d"% (strDataName, nRespID)] = \
                dcFeaturePerResponse
    
        for nSegID, (nSegStart, nSegEnd) in enumerate(lsSegments):
            arrSeg = arrFiltered[nSegStart:nSegEnd]
            dcSegFeatures = extractSegmentFeatures(strCol, nSegID, arrSeg)
            dcFeaturePerResponse.update(dcSegFeatures)
            
    return dcResponseFeature

def crossValidate(arrX, 
                  arrY, 
                  strModelName, 
                  dcModelParams,
                  lsFeatureNames, 
                  nFold=10):
    '''
        Cross validate specific model on given data set

        Params:
        ----
                arrX
                    features
                    
                arry
                    labels
                    
                strModelName
                    model to usefdsclfds

                dcModelParams
                    model params
                nFold
                    # fold
        Returns:
        ----
                dcResults
                    a dict of evaluation result of each fold
    '''

    # cross validation
    dcCVResult = {}
    kf = cross_validation.KFold(len(arrY), nFold, shuffle=True)

    i = 0
    for arrTrainIndex, arrTestIndex in kf:
        # split data
        arrX_train, arrX_test = arrX[arrTrainIndex], arrX[arrTestIndex]
        arrY_train, arrY_test = arrY[arrTrainIndex], arrY[arrTestIndex]

        # train
        model = trainModel(strModelName, dcModelParams, 
                           arrX_train, arrY_train)

        # test
        dcFoldResult = testModel(model, arrX_test, arrY_test, lsFeatureNames)
        
        dcCVResult[i] = dcFoldResult
        i = i+1

    return dcCVResult
    
def extractFeatureLabel(strDataPath, 
                        lsFileNames, 
                        lsDataAxisName, 
                        dSamplingFreq, 
                        dcLabel, 
                        nResponsePerData, 
                        nSegmentPerRespsonse, 
                        dVibrationDuration, 
                        dIntervalDuration,
                        dRestDuration, 
                        lsAxis2Inspect = ['x0', 'y0', 'z0']):
    """
        Extract features and labels from give data set
        
        Parameters:
        ----
        strPath: 
            folder containing the data
        lsFileNames: 
            list of data file names
        dcLabel: 
            the diction of label mapping
        nResponsePerData: 
            number of responses in a data
        
        Returns:
        ----
        dfFeatureLabel: 
            a data frame of features & corresponding labels
    """
    # load data
    lsData = md.loadDataEx(strDataPath, lsFileNames, lsDataAxisName)
    
    # extract features & labels
    lsDataFeatures = []
    lsLabels = []
    for strDataName, dfData in zip(lsFileNames, lsData):
        nLabel = dcLabel[strDataName[:2] ]
        
        lsAxisFeatures = []
        # extract features for each axis
        for strCol in lsAxis2Inspect:
            dcAxisFeatures = extractAxisFeature(dfData, 
                                                strDataName,
                                                strCol,
                                                dSamplingFreq, 
                                                nResponsePerData,
                                                nSegmentPerRespsonse,
                                                dVibrationDuration, 
                                                dIntervalDuration,
                                                dRestDuration)
                                                
            if (len(dcAxisFeatures.keys() ) != nResponsePerData ):
                raise RuntimeError("the responses extracted does not "
                "equal to predefined")
                
            dfAxisFeatures = pd.DataFrame(dcAxisFeatures).T
            lsAxisFeatures.append(dfAxisFeatures)
                   
        # combine X, Y, Z horizontally, note the index is used!
        dfDataFeatures = pd.concat(lsAxisFeatures, axis=1, 
                                   ignore_index=False)
        
        # save for furthur combination
        lsDataFeatures.append(dfDataFeatures)
        lsLabels.extend([nLabel]*dfDataFeatures.shape[0])

    # concatenate feature frames of all data together (vertically)
    dfFeatureLabel = pd.concat(lsDataFeatures, axis=0, ignore_index=False)
    dfFeatureLabel[LABEL] = pd.Series(lsLabels, index=dfFeatureLabel.index)
    
    return dfFeatureLabel
    
def preprocessFeatureLabel(arrX, arrY):
    """
        preprocess features and labels in place.
    """
    # fill nan
    arrX[np.isnan(arrX)] = 0.0

    # normalize features
    min_max_scaler = prepro.MinMaxScaler(copy=False)
    arrX = min_max_scaler.fit_transform(arrX)
    
    return arrX, arrY
    
    
def trainModel(strModelName, dcModelParams, arrX_train, arrY_train):
    """
        Use the given model setting and data to train a model
    """
    # setup model
    model = None
    if (strModelName == 'GBRT'):
        model = GradientBoostingClassifier()
    elif (strModelName == 'decision_tree'):
        model = DecisionTreeClassifier()
    elif (strModelName == 'extra_trees'):
        model = ExtraTreesClassifier()
    elif (strModelName == 'random_forest'):
        model = RandomForestClassifier()
    elif (strModelName == 'SVM'):
        model = SVC()
    else:
        raise  KeyError("Unsupported model: %s" % strModelName)
        
    if(dcModelParams is not None):
        model.set_params(**dcModelParams)

    # train
    model.fit(arrX_train, arrY_train)
    
    return model
    
def testModel(model, arrX_test, arrY_test, lsFeatureNames=None):
    """
        Use given data set to test model performance.
        
        Returns
        ----
        dcResult:
            a diction of model performance metrics
    """
    # test
    dcResult = {}
    arrY_pred = model.predict(arrX_test)
    
    print "test", arrY_test
    print "pred", arrY_pred, "\n"

    # evluate
    dAccuracy = accuracy_score(arrY_test, arrY_pred)
    dPrecision = precision_score(arrY_test, arrY_pred,
                                 average='macro', pos_label=None)
    dRecall = recall_score(arrY_test, arrY_pred, 
                           average='macro', pos_label=None)
    dF1 = f1_score(arrY_test, arrY_pred, average='macro', pos_label=None)
    mtConfusionMatrix = confusion_matrix(arrY_test, arrY_pred)
#    mtNormalizedCM = mtConfusionMatrix.astype('float') \
#                    / mtConfusionMatrix.sum(axis=1)[:, np.newaxis]
    
    dcResult[MODEL_CONFUSION_MATRIX] = mtConfusionMatrix
    dcResult[MODEL_ACCURACY] = dAccuracy
    dcResult[MODEL_PRECISION] = dPrecision
    dcResult[MODEL_RECALL] = dRecall
    dcResult[MODEL_F1] = dF1
    
    if(lsFeatureNames is not None):
        dcFeatureImportance = {k:v for (k,v) in zip(lsFeatureNames,
                               model.feature_importances_.tolist()) }
        dcResult[MODEL_FEATURE_IMPORTANCE] = dcFeatureImportance
    
    return dcResult
    
    
def customizeSegmentFeatures(strCol, nSegID, arrSeg, nValidSegment):
    """
        Extrac customized statistics from each vibration segment
        
        Parameters:
        ----
        strCol:
            the axis name of which this segment belongs to
        nSegID: 
            the id of current segment
        arrSeg: 
            data of segment
        nValidSegment: 
            number of points which can be used to extract features
        
        Returns:
        ----
        a dict of statistics, each of which has a key like: "x0_s1_std"
    """
    dcRet = {}
    dfStat = pd.Series(arrSeg[0: nValidSegment]).describe()
    
    # in case there are something wrong
    dfStat.replace([np.inf, -np.inf], np.nan, inplace=True)
    dfStat.fillna(0.0, inplace=True)

    dcRet["%s_s%d_%s" % (strCol, nSegID, STD)] = dfStat['std']
    dcRet["%s_s%d_%s" % (strCol, nSegID, MIN)] = dfStat['min']
    dcRet["%s_s%d_%s" % (strCol, nSegID, ONE_QUARTER)] = dfStat['25%']
    dcRet["%s_s%d_%s" % (strCol, nSegID, TWO_QUARTER)] = dfStat['50%']
    dcRet["%s_s%d_%s" % (strCol, nSegID, THREE_QUARTER)] = dfStat['75%']
    dcRet["%s_s%d_%s" % (strCol, nSegID, MAX)] = dfStat['max']

    return dcRet

def customizeAxisFeature(dfData,
                         strDataName,
                         strCol, 
                         dSamplingFreq, 
                         nReponses,
                         nSegmentPerRespsonse, 
                         dVibrationDuration,
                         dIntervalDuration,
                         dRestDuration,
                         lsValidSegmentID,
                         dValidSegmentDuration,
                         nFreqLowCut=20, nFreqHighCut=120):
    """
        Extract feature for single-axis data as follows:
        1. divide this data into multiple responses;
        2. divide each response into several segments;
        3. extract customized feature from selected segment;
        4. combine corresponding features together;
        
        Parameters:
        ----
        dfData: 
            the data frame
        strDataName:
            name of data
        strCol: 
            axis name
        dSamplingFreq: 
            sampling frequency of data
        nResponse: 
            number of responses in single-axis data
        nSegmentPerRespsonse: 
            number of excitation segment per response
        dVibrationDuration: 
            duration of vibration (seconds)
        dIntervalDuration: 
            interval duration btw segments (seconds)
        dRestDuration: 
            the static duration btw responses
        lsValidSegmentID: 
            list of valid segment id to extract features
        dValidSegmentDuration: 
            valid segment duration for extracting features
        nFreqLowCut: 
            the low cutting frequency of bandpass filter
        nFreqHighCut: 
            the high cutting frequency of bandpass filter 
        
        Returns:
        ----
        dcResponseFeature: 
            a dict of key-value pairs like: 
                     {'r1': {
                             'x0_s1_std': 0.85, 
                             'y0_s5_min': 4.72,
                             ...
                             }
                     }
    """
    arrData = dfData[strCol].values
    
    # bandpass filter to select spectrum of interest
    arrFiltered = bp_filter.butter_bandpass_filter(arrData, 
                                                   lowcut=nFreqLowCut,
                                                   highcut=nFreqHighCut,
                                                   fs=dSamplingFreq,
                                                   order=7)
                         
    # find reponses & segements in each response
    lsResponses, arrResponseEndIndex, \
    arrLocalPeakRefVal = sd.splitData(arrFiltered, 
                                      dSamplingFreq, 
                                      nReponses,
                                      nSegmentPerRespsonse,
                                      dVibrationDuration,
                                      dIntervalDuration,
                                      dRestDuration)
                                     
    dcResponseFeature = {}
    nValidSegmentPoints = int(dValidSegmentDuration * dSamplingFreq)
    for nRespID, lsSegments in enumerate(lsResponses):
        dcFeaturePerResponse = \
            dcResponseFeature.get("%s_r%d"%(strDataName, nRespID), None)
        if(dcFeaturePerResponse == None):
            dcFeaturePerResponse = {}
            dcResponseFeature["%s_r%d"%(strDataName, nRespID)] = \
                dcFeaturePerResponse
    
        for nSegID, (nSegStart, nSegEnd) in enumerate(lsSegments):
            if (nSegID in lsValidSegmentID):
                arrSeg = arrFiltered[nSegStart:nSegEnd]
                dcSegFeatures = customizeSegmentFeatures(strCol, 
                                                         nSegID,
                                                         arrSeg,
                                                         nValidSegmentPoints)
                dcFeaturePerResponse.update(dcSegFeatures)
            
    return dcResponseFeature


def customizeFeatureLabel(strDataPath, 
                          lsFileNames,
                          lsDataAxisName,
                          dSamplingFreq,
                          dcLabel, 
                          nResponsePerData, 
                          nSegmentPerRespsonse, 
                          dVibrationDuration, 
                          dIntervalDuration,
                          dRestDuration, 
                          lsValidSegmentID,
                          dValidSegmentDuration,
                          lsAxis2Inspect = ['x0', 'y0', 'z0'] ):
    """
        Extract customized features and labels from give data set
        
        Parameters:
        ----
        strPath: 
                folder path of data
        lsFileNames: 
                    names of files to use
        dcLabel: 
            diction for mapping btw file name and label
        nResponsePerData: 
            number of responses in single data
        nSegmentPerRespsonse: 
            number of segments in single response
        dVibrationDuration: 
            duration of each vibration segment
        dIntervalDuration: 
            static duration btw segments
        dRestDuration: 
            static duration btw responses
        lsValidSegmentID: 
            list of segments to extract features
        dValidSegmentDuration: 
            duration of segment to extract features
        lsAxis2Inspect:
            axis to extract features, default is ['x0', 'y0', 'z0']
        
        Returns:
        ----
        dfFeatures: 
            a data frame of features
        lsLabels: 
            a list of corresponding labels
    """
    if (len(lsValidSegmentID) == 0 or 
        len(lsValidSegmentID) > nSegmentPerRespsonse or
        dValidSegmentDuration <= 0.0 or 
        dValidSegmentDuration > dVibrationDuration):
            raise ValueError()
        
    
    # load data
    lsData = md.loadDataEx(strDataPath, lsFileNames, lsDataAxisName)
    
    # extract features & labels
    lsDataFeatures = []
    lsLabels = []
    
    for strDataName, dfData in zip(lsFileNames, lsData):
        nLabel = dcLabel[strDataName[:2] ]
        
        lsAxisFeatures = []
        # extract features for each axis
        for strCol in lsAxis2Inspect:
            dcAxisFeatures = customizeAxisFeature(dfData,
                                                  strDataName,
                                                  strCol,
                                                  dSamplingFreq, 
                                                  nResponsePerData,
                                                  nSegmentPerRespsonse,
                                                  dVibrationDuration, 
                                                  dIntervalDuration,
                                                  dRestDuration,
                                                  lsValidSegmentID,
                                                  dValidSegmentDuration)
                                                
            if (len(dcAxisFeatures.keys() ) != nResponsePerData ):
                raise RuntimeError("the responses extracted "
                "does not equal to predefined")
                
            dfAxisFeatures = pd.DataFrame(dcAxisFeatures).T
            lsAxisFeatures.append(dfAxisFeatures)
                   
        # combine X, Y, Z horizontally
        dfDataFeatures = pd.concat(lsAxisFeatures, axis=1, 
                                   ignore_index=False)
        
        # save for furthur combination
        lsDataFeatures.append(dfDataFeatures)
        lsLabels.extend([nLabel]*dfDataFeatures.shape[0])

    # concatenate feature frames of all data together (vertically)
    dfFeatureLabel = pd.concat(lsDataFeatures, axis=0, ignore_index=False)
    dfFeatureLabel[LABEL] = pd.Series(lsLabels, index=dfFeatureLabel.index)
    
    return dfFeatureLabel
    

    
if __name__ == "__main__":
    # basic plot setting
    lsBasicColors = ['r', 'g', 'b', 'c', 'm', 'y']
    lsMarkers = ['o', 'v', 'd', 's', '+', 'x',  '1', '2', '3', '4']
    strBasicFontName = "Times new Roman"
    nBasicFontSize = 16
    
    # data setting
    lsDataAxisName = ['x0', 'y0','z0', 'gx0', 'gy0','gz0']
    dSamplingFreq = 320.0

    # classification setting
    dcLabel = {"yl":1, "cy":2, "hc":3, "ww":4,
               "qy":5, "hy":6, "zy":7, "ch":8}
               
    print("This script is designed for cells, please run cells manually.")
    sys.exit(0)
               
#%% load data set & extract features
                 
    # data set to use
    strDataDir = "../../data/experiment/wearing_location/"
    lsFileNames = ds.lsYL_t36_v0_p0_m0_d1_l0 + \
                  ds.lsYL_t36_v0_p0_m0_d2_l0 + \
                  ds.lsYL_t36_v0_p0_m0_d3_l0 + \
                  ds.lsWW_t11_v0_p0_m0_d1_l0 + \
                  ds.lsWW_t11_v0_p0_m0_d2_l0 + \
                  ds.lsWW_t11_v0_p0_m0_d3_l0 + \
                  ds.lsQY_t7_v0_p0_m0_d1_l0 + \
                  ds.lsQY_t7_v0_p0_m0_d2_l0 + \
                  ds.lsQY_t7_v0_p0_m0_d3_l0 + \
                  ds.lsHCY_t10_v0_p0_m0_d1_l0 + \
                  ds.lsHCY_t10_v0_p0_m0_d2_l0 + \
                  ds.lsHCY_t10_v0_p0_m0_d3_l0 + \
                  ds.lsHCY_t10_v0_p0_m0_d4_l0 + \
                  ds.lsCHX_t5_v0_p0_m0_d1_l0 + \
                  ds.lsCHX_t5_v0_p0_m0_d2_l0 + \
                  ds.lsCHX_t5_v0_p0_m0_d3_l0 + \
                  ds.lsCHX_t5_v0_p0_m0_d4_l0

    # extract features & labels
    print("Extracting features and labels..." )
    dfFeaturesLabel= extractFeatureLabel(strDataDir, 
                                         lsFileNames,
                                         lsDataAxisName,
                                         dSamplingFreq,
                                         dcLabel, 
                                         nResponsePerData=3,
                                         nSegmentPerRespsonse = 13,
                                         dVibrationDuration = 1.4, 
                                         dIntervalDuration=0.0,
                                         dRestDuration=1.0, 
                                         lsAxis2Inspect=['x0', 'y0', 'z0'])
                                   
    # preprocess features
    lsFeatureNames = [strCol for strCol in dfFeaturesLabel.columns \
                        if strCol != LABEL]
    arrX = dfFeaturesLabel[lsFeatureNames].as_matrix()
    arrY = dfFeaturesLabel[LABEL].values
    
    arrX, arrY = preprocessFeatureLabel(arrX, arrY)
             

#%%  select training data set
    
    lsTrainingMask = dfFeaturesLabel.index.str.contains("v0_p0_m0_d1_l0")
    arrX_train = arrX[lsTrainingMask]
    arrY_train = arrY[lsTrainingMask]
    
    lsTestingMask = dfFeaturesLabel.index.str.contains("v0_p0_m0_d2_l0")                       
    arrX_test = arrX[lsTestingMask]
    arrY_test = arrY[lsTestingMask]

                      
#%%  train model
#    strModelName = 'GBRT'
#    modelParams = {'n_estimators':500, 'max_features':"auto"}
#    


    strModelName = 'random_forest'
    modelParams = {'n_estimators':10, "criterion":"gini", 
                   "max_features": "auto", "oob_score":True, 
                   "n_jobs": -1, "warm_start": False}


#    strModelName = 'decision_tree'
#    modelParams = {"criterion": "gini"}

    print("training model...")
    model = trainModel(strModelName, modelParams, arrX_train, arrY_train)
    
    
#  test model
    print("Testing model..." )
    dcResult = testModel(model, arrX_test, arrY_test, lsFeatureNames)
    
    lsFeatureImp = sorted(dcResult[MODEL_FEATURE_IMPORTANCE].items(),
                          key=operator.itemgetter(1), reverse=True )
    print("Feature importance:")
    for strFName, dFImp in lsFeatureImp[:5]:
        print("\t %s=%.2f" % (strFName, dFImp))
    
    for k, v in dcResult.iteritems():
        if(k != MODEL_FEATURE_IMPORTANCE):
            print k,"=", v
            
            

#%% cross validate on selected data
    lsCondition = dfFeaturesLabel.index.str.contains("d1")
    arrX_selected = arrX[lsCondition]
    arrY_selected = arrY[lsCondition]
    
    dcCVResults = crossValidate(arrX_selected, arrY_selected, 
                                strModelName='decision_tree', 
                                dcModelParams = {"criterion": "gini"}, 
                                lsFeatureNames = lsFeatureNames, 
                                nFold=5)
                  
    # output details of each fold
    for nFold, dcFoldResult in dcCVResults.iteritems():
        print("--Fold %d--" % (nFold) )
        
        print("Feature importance:")
        lsFeatureImp = sorted(dcFoldResult[MODEL_FEATURE_IMPORTANCE].items(),
                              key=operator.itemgetter(1), reverse=True )               
        for strFeature, dImp in lsFeatureImp[:5]:
            print("\t%s=%.2f" % (strFeature, dImp) )

        for k, v in dcFoldResult.iteritems():
            if(k != MODEL_FEATURE_IMPORTANCE):
                print k, ":",  v
        
        print("\n")

    # output overall performance
    lsAccuracy = [ i[MODEL_ACCURACY] for i in dcCVResults.values()]
    dBestAccuracy = np.max(lsAccuracy)
    dWorstAccuracy = np.min(lsAccuracy)
    dMeanAccuracy = np.mean(lsAccuracy)
    dAccuracyStd = np.std(lsAccuracy)
    print("overall performance on %d responses: \n  best=%.2f, worst=%.2f, "
          "mean=%.2f, std=%.2f" % \
           (len(arrX_selected), dBestAccuracy, dWorstAccuracy,
            dMeanAccuracy, dAccuracyStd) )
            
