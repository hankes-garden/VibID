# -*- coding: utf-8 -*-
"""
Given a data set of user's responses to excitation, this script first 
extracts features and corresponing labels for each user, and then train
a model to classify user.

Note: this script can only handle the new version data, which contains multiple responses in single data, and only consists of data from one accelerometer

@author: jason
"""
import bp_filter
import multiple_data as md
import single_data as sd
import dataset as ds

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator

import sklearn.preprocessing as prepro
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# name of sub-signal
SEGMENT = "segment"

# name of statistics
MEAN = "mean"
STD = "std"
ONE_QUARTER = "25p"
TWO_QUARTER = "50p"
THREE_QUARTER = "75p"
MIN = "min"
MAX = "max"

CN_MODEL_ACCURACY = "accuracy"
CN_MODEL_FEATURE_IMP = "feature_importance"

def extractSegmentFeatures(strCol, nSegID, arrSeg):
    """
        extrac statistic for each segment
        
        Parameters:
        ----
        strCol: the axis name of which this segment belongs to
        nSegID: the id of current segment
        arrSeg: data of segment
        
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

def extractAxisFeature(dfData, strCol, 
                       dSamplingFreq, nReponses,
                       nSegmentPerRespsonse, 
                       dVibrationDuration,
                       dIntervalDuration,
                       dRestDuration,
                       nFreqLowCut=20, nFreqHighCut=120):
    """
        extract feature for single-axis data
        1. first divide this data into reponses, which consists
           of multiple segments
        2. extract feature for each segment
        
        Parameters:
        ----
        dfData: the data frame
        strCol: axis name
        dSamplingFreq: sampling frequency of data
        nResponse: number of responses in single-axis data
        nSegmentPerRespsonse: number of excitation segment per response
        dVibrationDuration: duration of vibration (seconds)
        dIntervalDuration: interval duration btw segments (seconds)
        dRestDuration: the static duration btw responses
        nFreqLowCut: the low cutting frequency of bandpass filter
        nFreqHighCut: the high cutting frequency of bandpass filter 
        
        Returns:
        ----
        dcResponseFeature: a dict of key-value pairs like: 
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
        dcFeaturePerResponse = dcResponseFeature.get("r%d"%nRespID, None)
        if(dcFeaturePerResponse == None):
            dcFeaturePerResponse = {}
            dcResponseFeature["r%d"%nRespID] = dcFeaturePerResponse
    
        for nSegID, (nSegStart, nSegEnd) in enumerate(lsSegments):
            arrSeg = arrFiltered[nSegStart:nSegEnd]
            dcSegFeatures = extractSegmentFeatures(strCol, nSegID, arrSeg)
            dcFeaturePerResponse.update(dcSegFeatures)
            
    return dcResponseFeature

def classify(arrX, arrY, strModelName, dcModelParams, lsFeatureNames, nFold=10):
    '''
        Use given model to classify data

        params:
                arrX - features
                arry - labels
                strModelName - model to usefdsclfds

                dcModelParams - model params
                nFold - # fold
        return:
                dcResults - a dict of evaluation result of each fold
    '''

    #===========================================================================
    # cross validation
    #===========================================================================
    dcResults = {}
    kf = cross_validation.KFold(len(arrY), nFold, shuffle=True)

    i = 0
    for arrTrainIndex, arrTestIndex in kf:
        dcCurrentFold = {}

        # setup model
        model = None
        if (strModelName == 'GBRT'):
            if dcModelParams is not None:
                model = GradientBoostingClassifier(**dcModelParams)
            else:
                model = GradientBoostingClassifier()

        elif (strModelName == 'decision_tree'):
            if dcModelParams is not None:
                model = DecisionTreeClassifier(**dcModelParams)
            else:
                model = DecisionTreeClassifier()

        elif (strModelName == 'extra_trees'):
            if dcModelParams is not None:
                model = ExtraTreesClassifier(**dcModelParams)
            else:
                model = ExtraTreesClassifier()
                
        elif (strModelName == 'random_forest'):
            if dcModelParams is not None:
                model = RandomForestClassifier(**dcModelParams)
            else:
                model = RandomForestClassifier()

        elif (strModelName == 'SVM'):
            if dcModelParams is not None:
                model = SVC(**dcModelParams)
            else:
                model = SVC()
        else:
            print 'unsupported baseline!'
            break

        # fill nan
        arrX[np.isnan(arrX)] = 0.0

        # normalize features
        min_max_scaler = prepro.MinMaxScaler(copy=False)
        arrX = min_max_scaler.fit_transform(arrX)

        # split data
        arrX_train, arrX_test = arrX[arrTrainIndex], arrX[arrTestIndex]
        arrY_train, arrY_test = arrY[arrTrainIndex], arrY[arrTestIndex]

        # train
        model.fit(arrX_train, arrY_train)

        # test
        arrY_pred = model.predict(arrX_test)

        # evluate
        dAccuracy = accuracy_score(arrY_test, arrY_pred)
        
        print "test=", arrY_test
        print "pred=", arrY_pred
        
        print("acc: %.2f\n" % dAccuracy)

        dcCurrentFold[CN_MODEL_ACCURACY] = dAccuracy
        
        if(strModelName != "SVM"):
            dcFeatureImportance = { k:v for (k,v) in zip(lsFeatureColumns,
                                     model.feature_importances_.tolist()) }
            dcCurrentFold[CN_MODEL_FEATURE_IMP] = dcFeatureImportance

        dcResults[i] = dcCurrentFold
        i = i+1

    return dcResults, model
    
def extractFeatureLabel(strPath, lsFileNames, dcLabel, 
                        nResponsePerData, nSegmentPerRespsonse, 
                        dVibrationDuration, 
                        dIntervalDuration,
                        dRestDuration, 
                        lsAxis2Inspect = ['x0', 'y0', 'z0']):
    """
        Extract features and labels from give data set
        
        Parameters:
        ----
        strPath: folder containing the data
        lsFileNames: list of data file names
        dcLabel: the diction of label mapping
        nResponsePerData: number of responses in a data
        
        Returns:
        ----
        dfFeatures: a data frame of features
        lsLabels: a list of corresponding labels
    """
    # load data
    lsData = md.loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)
    
    # extract features & labels
    lsDataFeatures = []
    lsLabels = []
    for strDataName, dfData in zip(lsFileNames, lsData):
        nLabel = dcLabel[strDataName[:2] ]
        
        lsAxisFeatures = []
        # extract features for each axis
        for strCol in lsAxis2Inspect:
            dcAxisFeatures = extractAxisFeature(dfData, strCol,
                                                dSamplingFreq, 
                                                nResponsePerData,
                                                nSegmentPerRespsonse,
                                                dVibrationDuration, 
                                                dIntervalDuration,
                                                dRestDuration)
                                                
            if (len(dcAxisFeatures.keys() ) != nResponsePerData ):
                raise RuntimeError("the responses extracted does not equal to predefined")
                
            dfAxisFeatures = pd.DataFrame(dcAxisFeatures).T
            lsAxisFeatures.append(dfAxisFeatures)
                   
        # combine X, Y, Z horizontally
        dfDataFeatures = pd.concat(lsAxisFeatures, axis=1, 
                                   ignore_index=False)
        
        # save for furthur combination
        lsDataFeatures.append(dfDataFeatures)
        lsLabels.extend([nLabel]*dfDataFeatures.shape[0])

    # concatenate feature frames of all data together (vertically)
    dfFeatures = pd.concat(lsDataFeatures, axis=0, ignore_index=False)
    
    return dfFeatures, lsLabels
    
if __name__ == "__main__":
    # basic plot setting
    lsBasicColors = ['r', 'g', 'b', 'c', 'm', 'y']
    lsMarkers = ['o', 'v', 'd', 's', '+', 'x',  '1', '2', '3', '4']
    strBasicFontName = "Times new Roman"
    nBasicFontSize = 16
    
    # data setting
    lsColumnNames = ['x0', 'y0','z0', 'gx0', 'gy0','gz0']
    dSamplingFreq = 320.0

    # classification setting
    dcLabel = {"yl":1, "cy":2, "hc":3, "ww":4,
               "qy":5, "hy":6, "zy":7}
               
    #%% extract features & labels
    strWorkingDir = "../../data/experiment/user_identification_v2/"
    lsFileNames = ds.lsYL_t32_l2_p0 + ds.lsYL_t33_l2_p0 + \
                  ds.lsCYJ_t17_l2_p0 + ds.lsCYJ_t18_l2_p0 + \
                  ds.lsQY_t5_l2_p0 + ds.lsQY_t6_l2_p0 + \
                  ds.lsHCY_t6_l2_p0 + ds.lsHCY_t7_l2_p0 + \
                  ds.lsWW_t9_l2_p0 + ds.lsWW_t10_l2_p0

    print("Extracting features and labels..." )
    dfFeatures, lsLabels = extractFeatureLabel(strWorkingDir, 
                                               lsFileNames, 
                                               dcLabel, 
                                               nResponsePerData=3,
                                               nSegmentPerRespsonse = 13,
                                               dVibrationDuration = 1.4, 
                                               dIntervalDuration=0.0,
                                               dRestDuration=1.0, 
                                               lsAxis2Inspect=['x0', 'y0', 'z0'])

    #%% classify

    # prepare train & testing set
    lsFeatureColumns = dfFeatures.columns
    mtX = dfFeatures.as_matrix()
    arrY = np.array(lsLabels)

    # model setup
#    strModelName = 'GBRT'
#    modelParams = {'n_estimators':500, 'max_features':"auto"}
    
#    strModelName = 'extra_trees'
#    modelParams = {'n_estimators':500, "criterion":"gini", 
#                   "max_features": "auto", "oob_score":True, 
#                   "n_jobs": -1, "warm_start": False, 
#                   "random_state": 7}

#    strModelName = 'random_forest'
#    modelParams = {'n_estimators':500, "criterion":"gini", 
#                   "max_features": "auto", "oob_score":True, 
#                   "n_jobs": -1, "warm_start": False, 
#                   "random_state": 7}

    strModelName = 'decision_tree'
    modelParams = {"criterion": "gini"}

#    strModelName = 'SVM'
#    modelParams = None

    print("training & testing...")
    dcResults, model = classify(mtX, arrY, strModelName, modelParams,
                         lsFeatureColumns, nFold=5)

    # output feature importance
    for nFold, dcFoldResult in dcResults.iteritems():
        print("Fold %d:" % (nFold) )
        lsFeatureImp = sorted(dcFoldResult[CN_MODEL_FEATURE_IMP].items(),
                              key=operator.itemgetter(1), reverse=True )
        for strFeature, dImp in lsFeatureImp[:10]:
            print("%s: %.2f" % (strFeature, dImp) )

        print("--")
        print("Accuracy: %.2f" % (dcFoldResult[CN_MODEL_ACCURACY]) )
        print ("Sum: %.2f" % \
               sum(dcFoldResult[CN_MODEL_FEATURE_IMP].values() ) )
        print("****\n")

    # output overall performance
    lsAccuracy = [ i[CN_MODEL_ACCURACY] for i in dcResults.values()]
    dBestAccuracy = np.max(lsAccuracy)
    dWorstAccuracy = np.min(lsAccuracy)
    dMeanAccuracy = np.mean(lsAccuracy)
    dAccuracyStd = np.std(lsAccuracy)
    print("overall performance: \n  best=%.2f, worst=%.2f, "
          "mean=%.2f, std=%.2f" % \
           (dBestAccuracy, dWorstAccuracy,
            dMeanAccuracy, dAccuracyStd) )
            
