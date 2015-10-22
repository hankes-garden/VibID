# -*- coding: utf-8 -*-
"""
Micro-benchmarking of Excitation
1. number of excitation frequency
2. duration of excitation frequency
3. Intensity*

Created on Sun Oct 04 17:25:38 2015

@author: jason
"""

import dataset as ds
import user_classification as uc
import single_data as sd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy import fftpack

ACC_MEAN = "accuracy_mean"
ACC_STD = "accuracy_std"
PRECISION_MEAN = "precision_mean"
PRECISION_STD = "precision_std"
RECALL_MEAN = "recall_mean"
RECALL_STD = "recall_std"

POSTURE_0_0 = "0-0"
POSTURE_0_30 = "0-30"
POSTURE_0_60 = "0-60"
POSTURE_0_270 = "0-270"
POSTURE_0_330 = "0-330"
POSTURE_0_X = "0-x"
POSTURE_X_X = "x-x"

MUSCLE_R_R = "r-r"
MUSCLE_R_T = "r-t"
MUSCLE_X_X = "x-x"

MOBILITY_S_S = "s-s"
MOBILITY_S_M = "s-m"
MOBILITY_X_X = "x-x"


EXCITATION_PARAM_FREQ = "excitation_param_freq"
EXCITATION_PARAM_SEGMENT_DURATION = "excitation_param_segment_duration"

def summarizeCVResult(dcCVResults, strMetric):
    """
        Given a diction of cross validation result, compute average
        value and std for specific metric
    """
    lsMetricValue = [dcFoldResult[strMetric] \
                     for nFold, dcFoldResult in dcCVResults.iteritems() ]
    dcSummary = {}
    dcSummary["%s_mean" % strMetric] = np.mean(lsMetricValue)
    dcSummary["%s_std" % strMetric] = np.std(lsMetricValue)
    
    return dcSummary

def evaluateExcitationFrequency(strDataDir,
                                lsFileNames,
                                lsDataAxisName,
                                dcLabel,
                                nFold=10):
    """
        Evaluate the impact of excitation frequencies
        on user classification.
    """
    # customize the extraction of features
    dValidSegDuration = 0.6
    lsFullSegment = range(0, 12)
        
    dcEvaluationResult = {}
    for nSegNum in xrange(1, 12, 1):
        print("\n----nSegNum:%d----" % nSegNum)
        print("extracting features & labels...")
        lsValidSegID = lsFullSegment[:nSegNum]
        dfFeatureLabel = uc.customizeFeatureLabel(strDataDir, 
                                                  lsFileNames, 
                                                  lsDataAxisName,
                                                  dSamplingFreq,
                                                  dcLabel, 
                                                  nResponsePerData=3,
                                                  nSegmentPerRespsonse = 13,
                                                  dVibrationDuration = 1.4, 
                                                  dIntervalDuration=0.0,
                                                  dRestDuration=1.0,
                                      lsValidSegmentID=lsValidSegID,
                                      dValidSegmentDuration=dValidSegDuration,
                                      lsAxis2Inspect=['x0', 'y0', 'z0'])
                                      
        # prepare train & testing set
        lsFeatureNames = [strCol for strCol in dfFeatureLabel.columns \
                            if strCol != uc.LABEL]
        arrX = dfFeatureLabel[lsFeatureNames].as_matrix()
        arrY = dfFeatureLabel[uc.LABEL].values
        
        # cross validate
        strModelName = 'random_forest'
        dcModelParams = {'n_estimators':10, "criterion":"gini",
                       "max_features": "auto", "oob_score":True,
                       "n_jobs": -1, "warm_start": False}
    
        print("training & testing...")
        dcCVResult = uc.crossValidate(arrX, arrY, strModelName, dcModelParams,
                                      lsFeatureNames, nFold)
                             
        # overall performance
        lsAccuracy = [i[uc.MODEL_ACCURACY] \
                      for i in dcCVResult.values()]
        lsPrecision = [i[uc.MODEL_PRECISION] \
                       for i in dcCVResult.values()]
        lsRecall = [i[uc.MODEL_RECALL] \
                    for i in dcCVResult.values()]
        dcResult = {ACC_MEAN: np.mean(lsAccuracy), 
                    ACC_STD: np.std(lsAccuracy),
                    PRECISION_MEAN: np.mean(lsPrecision), 
                    PRECISION_STD: np.std(lsPrecision),
                    RECALL_MEAN: np.mean(lsRecall),
                    RECALL_STD: np.std(lsRecall) }
            
        dcEvaluationResult[nSegNum] = dcResult
        
    # plot
    dfEvaluationResult = pd.DataFrame(dcEvaluationResult).T

    ax = plt.figure(figsize=(5, 4) ).add_subplot(111)

    dBarWidth = 0.34
    dBarInterval = 0.07
    arrInd = np.arange(dfEvaluationResult.shape[0])
    
    ax.bar(arrInd, dfEvaluationResult[PRECISION_MEAN], 
           dBarWidth, color=strColor_light_1, label=uc.MODEL_PRECISION,
           hatch=lsHatch[0])
    ax.bar(arrInd+dBarWidth+dBarInterval, dfEvaluationResult[RECALL_MEAN], 
           dBarWidth, color=strColor_dark_1, label=uc.MODEL_RECALL,
           hatch=lsHatch[1])
    
    # decorate figure
    ax.set_xlim(0, 11)
    ax.set_xticks(arrInd+dBarWidth+dBarInterval)
    ax.set_xticklabels(dfEvaluationResult.index.values* 10, 
                   fontname=strBasicFontName,
                   fontsize=nBasicFontSize)
    
    ax.set_xlabel("Frequency bandwidth of excitation (Hz)",
                  fontname=strBasicFontName,
                  fontsize=nBasicFontSize)
    xTickLabels = ax.xaxis.get_ticklabels()
    plt.setp(xTickLabels, fontname=strBasicFontName,
             size=nBasicFontSize)
             
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Performance", 
                  fontname=strBasicFontName,
                  fontsize=nBasicFontSize)
    arrYTicks = ax.yaxis.get_ticklabels()
    plt.setp(arrYTicks, fontname=strBasicFontName,
             size=nBasicFontSize)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
                                       
    return dcEvaluationResult

    
def evaluateExcitationDuration(strDataDir,
                               lsFileNames,
                               lsDataAxisName,
                               dcLabel,
                               nFold=10):
    """
        Evaluate the impact of duration of each segment
        on user classification.
    """
    # customize the extraction of features
    lsFullSegment = range(0, 12)
    
    dcEvaluationResult = {}
    for dSegDuration in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]:
        print("\n----dSegDuration:%.2f----" % dSegDuration)
        print("extracting features & labels...")
        lsValidSegID = lsFullSegment
        dfFeatureLabel = uc.customizeFeatureLabel(strDataDir, 
                                                  lsFileNames, 
                                                  lsDataAxisName,
                                                  dSamplingFreq,
                                                  dcLabel, 
                                                  nResponsePerData=3,
                                                  nSegmentPerRespsonse = 13,
                                                  dVibrationDuration = 1.4, 
                                                  dIntervalDuration=0.0,
                                                  dRestDuration=1.0,
                                          lsValidSegmentID=lsValidSegID,
                                          dValidSegmentDuration=dSegDuration,
                                          lsAxis2Inspect=['x0', 'y0', 'z0'])
                                      
        # prepare train & testing set
        lsFeatureNames = [strCol for strCol in dfFeatureLabel.columns \
                            if strCol != uc.LABEL]
        arrX = dfFeatureLabel[lsFeatureNames].as_matrix()
        arrY = dfFeatureLabel[uc.LABEL].values
    
        # cross validate
        strModelName = 'random_forest'
        dcModelParams = {'n_estimators':10, "criterion":"gini",
                       "max_features": "auto", "oob_score":True,
                       "n_jobs": -1, "warm_start": False}
    
        print("training & testing...")
        dcCVResult = uc.crossValidate(arrX, arrY, strModelName,
                                      dcModelParams, lsFeatureNames, nFold)
                             
        # overall performance
        lsAccuracy = [i[uc.MODEL_ACCURACY] \
                      for i in dcCVResult.values()]
        lsPrecision = [i[uc.MODEL_PRECISION] \
                       for i in dcCVResult.values()]
        lsRecall = [i[uc.MODEL_RECALL] \
                    for i in dcCVResult.values()]
        dcResult = {ACC_MEAN: np.mean(lsAccuracy), 
                    ACC_STD: np.std(lsAccuracy),
                    PRECISION_MEAN: np.mean(lsPrecision), 
                    PRECISION_STD: np.std(lsPrecision),
                    RECALL_MEAN: np.mean(lsRecall),
                    RECALL_STD: np.std(lsRecall) }
        dcEvaluationResult[dSegDuration] = dcResult
                    
    # plot
    dfEvaluationResult = pd.DataFrame(dcEvaluationResult).T
 
    ax = plt.figure(figsize=(5, 4) ).add_subplot(111)

    dBarWidth = 0.34
    dBarInterval = 0.07
    arrInd = np.arange(dfEvaluationResult.shape[0])
    
    ax.bar(arrInd, dfEvaluationResult[PRECISION_MEAN], 
           dBarWidth, color=strColor_light_1, label=uc.MODEL_PRECISION,
           hatch=lsHatch[0])
    ax.bar(arrInd+dBarWidth+dBarInterval, dfEvaluationResult[RECALL_MEAN], 
           dBarWidth, color=strColor_dark_1, label=uc.MODEL_RECALL,
           hatch=lsHatch[1])
    
    # decorate figure
    ax.set_xlim(0, dfEvaluationResult.shape[0])
    ax.set_xticks(arrInd+dBarWidth+dBarInterval)
    ax.set_xticklabels(dfEvaluationResult.index.values, 
                   fontname=strBasicFontName,
                   fontsize=nBasicFontSize)
    
    ax.set_xlabel("Vibration duration (seconds)",
                  fontname=strBasicFontName,
                  fontsize=nBasicFontSize)
    xTickLabels = ax.xaxis.get_ticklabels()
    plt.setp(xTickLabels, fontname=strBasicFontName,
             size=nBasicFontSize)
             
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Performance", 
                  fontname=strBasicFontName,
                  fontsize=nBasicFontSize)
    arrYTicks = ax.yaxis.get_ticklabels()
    plt.setp(arrYTicks, fontname=strBasicFontName,
             size=nBasicFontSize)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    return dfEvaluationResult
         
def evaluatePosture(strDataDir, lsFileNames, lsDataAxisName, 
                    dcLabel, nFold=10):
    """
        Evaluate impact of arm posture
        1. customize feature
        2. select training & test data
        3. train & test
    """
    dValidSegDuration = 1.4
    lsFullSegment = range(0, 13)
    lsValidSegID = lsFullSegment
        
    dcResult = {}
    
    
    print("-->extracting features & labels...")
    dfFeatureLabel = uc.customizeFeatureLabel(strDataDir, 
                                              lsFileNames, 
                                              lsDataAxisName,
                                              dSamplingFreq,
                                              dcLabel, 
                                              nResponsePerData=3,
                                              nSegmentPerRespsonse = 13,
                                  dVibrationDuration = dValidSegDuration, 
                                  dIntervalDuration=0.0,
                                  dRestDuration=1.0,
                                  lsValidSegmentID=lsValidSegID,
                                  dValidSegmentDuration=dValidSegDuration,
                                  lsAxis2Inspect=['x0', 'y0', 'z0'])
    # preprocess features
    lsFeatureNames = [strCol for strCol in dfFeatureLabel.columns \
                    if strCol != uc.LABEL]
    arrX = dfFeatureLabel[lsFeatureNames].as_matrix()
    arrY = dfFeatureLabel[uc.LABEL].values
    arrX, arrY = uc.preprocessFeatureLabel(arrX, arrY)
    
    # set up model 
#    strModelName = 'decision_tree'
#    dcModelParams = {"criterion": "gini"}
    strModelName = 'random_forest'
    dcModelParams = {'n_estimators':10, "criterion":"gini",
                   "max_features": "auto", "oob_score":True,
                   "n_jobs": -1, "warm_start": False}

    # 0 vs. 0 degree
    lsCriterion = dfFeatureLabel.index.str.contains("v0_p0_m0_d1_l0")
    arrX_selected = arrX[lsCriterion]
    arrY_selected = arrY[lsCriterion]
    
    dcTemp = uc.crossValidate(arrX_selected, arrY_selected, strModelName,
                              dcModelParams, None, nFold)
    dcResult_0_0 = {}
    dcResult_0_0[uc.MODEL_PRECISION_MEAN] = \
        (summarizeCVResult(dcTemp, uc.MODEL_PRECISION) ) \
        [uc.MODEL_PRECISION_MEAN]
    dcResult_0_0[uc.MODEL_PRECISION_STD] = \
        (summarizeCVResult(dcTemp, uc.MODEL_PRECISION) ) \
        [uc.MODEL_PRECISION_STD]
        
    dcResult_0_0[uc.MODEL_RECALL_MEAN] = \
        (summarizeCVResult(dcTemp, uc.MODEL_RECALL) ) \
        [uc.MODEL_RECALL_MEAN]
    dcResult_0_0[uc.MODEL_RECALL_STD] = \
        (summarizeCVResult(dcTemp, uc.MODEL_RECALL) ) \
        [uc.MODEL_RECALL_STD]
    
    dcResult[POSTURE_0_0] = dcResult_0_0
    
    
    # 0 vs. 30 degree
    arrTraningMask = dfFeatureLabel.index.str.contains("v0_p0_m0_d1_l0")
    arrX_train = arrX[arrTraningMask]
    arrY_train = arrY[arrTraningMask]
    arrTestingMask = dfFeatureLabel.index.str.contains("v0_p30_m0_d1_l0")
    arrX_test = arrX[arrTestingMask]
    arrY_test = arrY[arrTestingMask]
    
    lsFoldResult = []
    for i in xrange(nFold):
        model = uc.trainModel(strModelName, dcModelParams, 
                              arrX_train, arrY_train)
        dcTemp = uc.testModel(model, arrX_test, arrY_test, lsFeatureNames)
        lsFoldResult.append(dcTemp)
    dcResult_0_30 = {}
    dcResult_0_30[uc.MODEL_PRECISION_MEAN] = np.mean( [dc[uc.MODEL_PRECISION] \
                                                  for dc in lsFoldResult] )
    dcResult_0_30[uc.MODEL_PRECISION_STD] = np.std( [ dc[uc.MODEL_PRECISION] \
                                                  for dc in lsFoldResult] )
                                                      
    dcResult_0_30[uc.MODEL_RECALL_MEAN] = np.mean( [ dc[uc.MODEL_RECALL] \
                                                  for dc in lsFoldResult] )
    dcResult_0_30[uc.MODEL_RECALL_STD] = np.std( [ dc[uc.MODEL_RECALL] \
                                                  for dc in lsFoldResult] )
    dcResult[POSTURE_0_30] = dcResult_0_30
    
    # 0 vs. 60 degree
    arrTraningMask = dfFeatureLabel.index.str.contains("v0_p0_m0_d1_l0")
    arrX_train = arrX[arrTraningMask]
    arrY_train = arrY[arrTraningMask]
    arrTestingMask = dfFeatureLabel.index.str.contains("v0_p60_m0_d1_l0")
    arrX_test = arrX[arrTestingMask]
    arrY_test = arrY[arrTestingMask]
    
    lsFoldResult = []
    for i in xrange(nFold):
        model = uc.trainModel(strModelName, dcModelParams, 
                              arrX_train, arrY_train)
        dcTemp = uc.testModel(model, arrX_test, arrY_test, lsFeatureNames)
        lsFoldResult.append(dcTemp)
    dcResult_0_60 = {}
    dcResult_0_60[uc.MODEL_PRECISION_MEAN] = np.mean( [dc[uc.MODEL_PRECISION] \
                                                  for dc in lsFoldResult] )
    dcResult_0_60[uc.MODEL_PRECISION_STD] = np.std( [ dc[uc.MODEL_PRECISION] \
                                                  for dc in lsFoldResult] )
                                                      
    dcResult_0_60[uc.MODEL_RECALL_MEAN] = np.mean( [ dc[uc.MODEL_RECALL] \
                                                  for dc in lsFoldResult] )
    dcResult_0_60[uc.MODEL_RECALL_STD] = np.std( [ dc[uc.MODEL_RECALL] \
                                                  for dc in lsFoldResult] )
    dcResult[POSTURE_0_60] = dcResult_0_60
    
    # 0 vs. 330 degree
    arrTraningMask = dfFeatureLabel.index.str.contains("v0_p0_m0_d1_l0")
    arrX_train = arrX[arrTraningMask]
    arrY_train = arrY[arrTraningMask]
    arrTestingMask = dfFeatureLabel.index.str.contains("v0_p330_m0_d1_l0")
    arrX_test = arrX[arrTestingMask]
    arrY_test = arrY[arrTestingMask]
    
    lsFoldResult = []
    for i in xrange(nFold):
        model = uc.trainModel(strModelName, dcModelParams, 
                              arrX_train, arrY_train)
        dcTemp = uc.testModel(model, arrX_test, arrY_test, lsFeatureNames)
        lsFoldResult.append(dcTemp)
    dcResult_0_330 = {}
    dcResult_0_330[uc.MODEL_PRECISION_MEAN] = np.mean( [dc[uc.MODEL_PRECISION] \
                                                  for dc in lsFoldResult] )
    dcResult_0_330[uc.MODEL_PRECISION_STD] = np.std( [ dc[uc.MODEL_PRECISION] \
                                                  for dc in lsFoldResult] )
                                                      
    dcResult_0_330[uc.MODEL_RECALL_MEAN] = np.mean( [ dc[uc.MODEL_RECALL] \
                                                  for dc in lsFoldResult] )
    dcResult_0_330[uc.MODEL_RECALL_STD] = np.std( [ dc[uc.MODEL_RECALL] \
                                                  for dc in lsFoldResult] )
    dcResult[POSTURE_0_330] = dcResult_0_330
    
    # 0 vs. x degree
    arrTraningMask = dfFeatureLabel.index.str.contains("v0_p0_m0_d1_l0")
    arrX_train = arrX[arrTraningMask]
    arrY_train = arrY[arrTraningMask]
    arrTestingMask = \
        dfFeatureLabel.index.str.contains("v0_p30_m0_d1_l0") \
        | dfFeatureLabel.index.str.contains("v0_p60_m0_d1_l0") \
        | dfFeatureLabel.index.str.contains("v0_p330_m0_d1_l0")
    arrX_test = arrX[arrTestingMask]
    arrY_test = arrY[arrTestingMask]
    
    lsFoldResult = []
    for i in xrange(nFold):
        model = uc.trainModel(strModelName, dcModelParams, 
                              arrX_train, arrY_train)
        dcTemp = uc.testModel(model, arrX_test, arrY_test, lsFeatureNames)
        lsFoldResult.append(dcTemp)
    dcResult_0_x = {}
    dcResult_0_x[uc.MODEL_PRECISION_MEAN] = np.mean( [ dc[uc.MODEL_PRECISION] \
                                                  for dc in lsFoldResult] )
    dcResult_0_x[uc.MODEL_PRECISION_STD] = np.std( [ dc[uc.MODEL_PRECISION] \
                                                  for dc in lsFoldResult] )
                                                      
    dcResult_0_x[uc.MODEL_RECALL_MEAN] = np.mean( [ dc[uc.MODEL_RECALL] \
                                                  for dc in lsFoldResult] )
    dcResult_0_x[uc.MODEL_RECALL_STD] = np.std( [ dc[uc.MODEL_RECALL] \
                                                  for dc in lsFoldResult] )
    dcResult[POSTURE_0_X] = dcResult_0_x
    
    # x vs. x degree
    lsCriterion = dfFeatureLabel.index.str.contains("v0_p0_m0_d1_l0") \
        | dfFeatureLabel.index.str.contains("v0_p30_m0_d1_l0") \
        | dfFeatureLabel.index.str.contains("v0_p60_m0_d1_l0") \
        | dfFeatureLabel.index.str.contains("v0_p330_m0_d1_l0")
    arrX_selected = arrX[lsCriterion]
    arrY_selected = arrY[lsCriterion]
    
    dcTemp = uc.crossValidate(arrX_selected, arrY_selected, strModelName,
                              dcModelParams, None, nFold)
    dcResult_x_x = {}
    dcResult_x_x[uc.MODEL_PRECISION_MEAN] = \
        (summarizeCVResult(dcTemp, uc.MODEL_PRECISION) ) \
        [uc.MODEL_PRECISION_MEAN]
    dcResult_x_x[uc.MODEL_PRECISION_STD] = \
        (summarizeCVResult(dcTemp, uc.MODEL_PRECISION) ) \
        [uc.MODEL_PRECISION_STD]
        
    dcResult_x_x[uc.MODEL_RECALL_MEAN] = \
        (summarizeCVResult(dcTemp, uc.MODEL_RECALL) ) \
        [uc.MODEL_RECALL_MEAN]
    dcResult_x_x[uc.MODEL_RECALL_STD] = \
        (summarizeCVResult(dcTemp, uc.MODEL_RECALL) ) \
        [uc.MODEL_RECALL_STD]
    
    dcResult[POSTURE_X_X] = dcResult_x_x

    # plot
    dfResult = pd.DataFrame(dcResult).T
    dfResult = dfResult.reindex([POSTURE_0_0, POSTURE_0_30,
                                 POSTURE_0_60, POSTURE_0_330, 
                                 POSTURE_0_X, POSTURE_X_X])
   
    ax0 = plt.figure(figsize=(5,4) ).add_subplot(111 )
    dBarWidth = 0.34
    dBarInterval = 0.07  
    arrInd = np.arange(dfResult.shape[0])
    
    ax0.bar(arrInd, dfResult[uc.MODEL_PRECISION_MEAN].values, 
            hatch=lsHatch[0],
            width=dBarWidth, color=strColor_light_1, 
            label=uc.MODEL_PRECISION)
            
    ax0.bar(arrInd+dBarWidth+dBarInterval, 
            dfResult[uc.MODEL_RECALL_MEAN].values, 
            hatch=lsHatch[1],
            width=dBarWidth, color=strColor_dark_1, label=uc.MODEL_RECALL)
    
    
    # decorate figure
    ax0.set_ylim(0.0, 1.1)
    ax0.set_xlim(0, dfResult.shape[0])
    ax0.set_xticks(arrInd+dBarWidth+dBarInterval)
    ax0.set_xticklabels(dfResult.index.tolist(), 
                       fontname=strBasicFontName,
                       fontsize=nBasicFontSize)
                   
    ax0.set_xlabel("Training angle - testing angle", fontname=strBasicFontName,
                  fontsize=nBasicFontSize)
    ax0.set_ylabel("Peformance", fontname=strBasicFontName,
                  fontsize=nBasicFontSize)
    arrYTicks = ax0.yaxis.get_ticklabels()
    plt.setp(arrYTicks, fontname=strBasicFontName,
             size=nBasicFontSize)
    ax0.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    return dfResult
    

def evaluateMuscleState(strDataDir, lsFileNames, lsDataAxisName, 
                        dcLabel, nFold=10):
    """
        Evaluate impact of muscle state
        1. customize feature
        2. select training & test data
        3. train & test
    """
    dValidSegDuration = 1.4
    lsFullSegment = range(0, 13)
    lsValidSegID = lsFullSegment
        
    print("-->extracting features & labels...")
    dcResult = {}
    dfFeatureLabel = uc.customizeFeatureLabel(strDataDir, 
                                              lsFileNames, 
                                              lsDataAxisName,
                                              dSamplingFreq,
                                              dcLabel, 
                                              nResponsePerData=3,
                                              nSegmentPerRespsonse = 13,
                                  dVibrationDuration = dValidSegDuration, 
                                  dIntervalDuration=0.0,
                                  dRestDuration=1.0,
                                  lsValidSegmentID=lsValidSegID,
                                  dValidSegmentDuration=dValidSegDuration,
                                  lsAxis2Inspect=['x0', 'y0', 'z0'])
    # preprocess features
    lsFeatureNames = [strCol for strCol in dfFeatureLabel.columns \
                    if strCol != uc.LABEL]
    arrX = dfFeatureLabel[lsFeatureNames].as_matrix()
    arrY = dfFeatureLabel[uc.LABEL].values
    arrX, arrY = uc.preprocessFeatureLabel(arrX, arrY)
    
    # set up model 
    strModelName = 'random_forest'
    dcModelParams = {'n_estimators':100, "criterion":"gini",
                   "max_features": "auto", "oob_score":True,
                   "n_jobs": -1, "warm_start": False}

    # relaxed vs. relaxed state
    lsCriterion = dfFeatureLabel.index.str.contains("v0_p0_m0_d1_l0")
    arrX_selected = arrX[lsCriterion]
    arrY_selected = arrY[lsCriterion]
    
    dcTemp = uc.crossValidate(arrX_selected, arrY_selected, strModelName,
                              dcModelParams, None, nFold)
    dcResult_r_r = {}
    dcResult_r_r[uc.MODEL_PRECISION_MEAN] = \
        (summarizeCVResult(dcTemp, uc.MODEL_PRECISION) ) \
        [uc.MODEL_PRECISION_MEAN]
    dcResult_r_r[uc.MODEL_PRECISION_STD] = \
        (summarizeCVResult(dcTemp, uc.MODEL_PRECISION) ) \
        [uc.MODEL_PRECISION_STD]
        
    dcResult_r_r[uc.MODEL_RECALL_MEAN] = \
        (summarizeCVResult(dcTemp, uc.MODEL_RECALL) ) \
        [uc.MODEL_RECALL_MEAN]
    dcResult_r_r[uc.MODEL_RECALL_STD] = \
        (summarizeCVResult(dcTemp, uc.MODEL_RECALL) ) \
        [uc.MODEL_RECALL_STD]
    
    dcResult[MUSCLE_R_R] = dcResult_r_r
    
    
    # relaxed vs. tense state
    arrTraningMask = dfFeatureLabel.index.str.contains("v0_p0_m0_d1_l0")
    arrX_train = arrX[arrTraningMask]
    arrY_train = arrY[arrTraningMask]
    arrTestingMask = dfFeatureLabel.index.str.contains("v0_p0_m1_d1_l0")
    arrX_test = arrX[arrTestingMask]
    arrY_test = arrY[arrTestingMask]
    
    lsFoldResult = []
    for i in xrange(nFold):
        model = uc.trainModel(strModelName, dcModelParams, 
                              arrX_train, arrY_train)
        dcTemp = uc.testModel(model, arrX_test, arrY_test, lsFeatureNames)
        lsFoldResult.append(dcTemp)
        
    dcResult_r_t = {}
    dcResult_r_t[uc.MODEL_PRECISION_MEAN] = np.mean( [dc[uc.MODEL_PRECISION] \
                                                for dc in lsFoldResult] )
    dcResult_r_t[uc.MODEL_PRECISION_STD] = np.std( [dc[uc.MODEL_PRECISION] \
                                                for dc in lsFoldResult] )
                                                    
    dcResult_r_t[uc.MODEL_RECALL_MEAN] = np.mean( [ dc[uc.MODEL_RECALL] \
                                                  for dc in lsFoldResult] )
    dcResult_r_t[uc.MODEL_RECALL_STD] = np.std( [ dc[uc.MODEL_RECALL] \
                                                  for dc in lsFoldResult] )
    dcResult[MUSCLE_R_T] = dcResult_r_t
    
    
    # mixed vs. mixed state
    lsCriterion = dfFeatureLabel.index.str.contains("v0_p0_m0_d1_l0") \
                | dfFeatureLabel.index.str.contains("v0_p0_m1_d1_l0")
    arrX_selected = arrX[lsCriterion]
    arrY_selected = arrY[lsCriterion]
    
    dcTemp = uc.crossValidate(arrX_selected, arrY_selected, strModelName,
                              dcModelParams, None, nFold)
    dcResult_x_x = {}
    dcResult_x_x[uc.MODEL_PRECISION_MEAN] = \
        (summarizeCVResult(dcTemp, uc.MODEL_PRECISION) ) \
        [uc.MODEL_PRECISION_MEAN]
    dcResult_x_x[uc.MODEL_PRECISION_STD] = \
        (summarizeCVResult(dcTemp, uc.MODEL_PRECISION) ) \
        [uc.MODEL_PRECISION_STD]
        
    dcResult_x_x[uc.MODEL_RECALL_MEAN] = \
        (summarizeCVResult(dcTemp, uc.MODEL_RECALL) ) \
        [uc.MODEL_RECALL_MEAN]
    dcResult_x_x[uc.MODEL_RECALL_STD] = \
        (summarizeCVResult(dcTemp, uc.MODEL_RECALL) ) \
        [uc.MODEL_RECALL_STD]
    
    dcResult[MUSCLE_X_X] = dcResult_x_x

    # plot
    dfResult = pd.DataFrame(dcResult).T
    dfResult = dfResult.reindex([MUSCLE_R_R, MUSCLE_R_T, MUSCLE_X_X])
    
    ax0 = plt.figure(figsize=(5,4) ).add_subplot(111 )
    dBarWidth = 0.34
    dBarInterval = 0.07  
    arrInd = np.arange(dfResult.shape[0])
    
    ax0.bar(arrInd, dfResult[uc.MODEL_PRECISION_MEAN].values,
            hatch=lsHatch[0],
            width=dBarWidth, color=strColor_light_1, label=uc.MODEL_PRECISION)
            
    ax0.bar(arrInd+dBarWidth+dBarInterval, 
            dfResult[uc.MODEL_RECALL_MEAN].values,
            hatch=lsHatch[1],
            width=dBarWidth, color=strColor_dark_1, label=uc.MODEL_RECALL)
    
    
    # decorate figure
    ax0.set_xlim(0, dfResult.shape[0])
    ax0.set_ylim(0.0, 1.1)
    ax0.set_xticks(arrInd+dBarWidth+dBarInterval)
    ax0.set_xticklabels(dfResult.index.tolist(), 
                       fontname=strBasicFontName,
                       fontsize=nBasicFontSize)
                   
    ax0.set_xlabel("Training state - testing state",
                   fontname=strBasicFontName,
                   fontsize=nBasicFontSize)
    ax0.set_ylabel("Peformance", fontname=strBasicFontName,
                  fontsize=nBasicFontSize)
    arrYTicks = ax0.yaxis.get_ticklabels()
    plt.setp(arrYTicks, fontname=strBasicFontName,
             size=nBasicFontSize)
    ax0.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    return dfResult
    
def evaluateUserMobility(strDataDir, lsFileNames, lsDataAxisName, 
                         dcLabel, nFold=10):
    """
        Evaluate impact of user mobility
        1. customize feature
        2. select training & test data
        3. train & test
    """
    dValidSegDuration = 1.4
    lsFullSegment = range(0, 13)
    lsValidSegID = lsFullSegment
        
    print("-->extracting features & labels...")
    dcResult = {}
    dfFeatureLabel = uc.customizeFeatureLabel(strDataDir, 
                                              lsFileNames, 
                                              lsDataAxisName,
                                              dSamplingFreq,
                                              dcLabel, 
                                              nResponsePerData=3,
                                              nSegmentPerRespsonse = 13,
                                  dVibrationDuration = dValidSegDuration, 
                                  dIntervalDuration=0.0,
                                  dRestDuration=1.0,
                                  lsValidSegmentID=lsValidSegID,
                                  dValidSegmentDuration=dValidSegDuration,
                                  lsAxis2Inspect=['x0', 'y0', 'z0'])
    # preprocess features
    lsFeatureNames = [strCol for strCol in dfFeatureLabel.columns \
                    if strCol != uc.LABEL]
    arrX = dfFeatureLabel[lsFeatureNames].as_matrix()
    arrY = dfFeatureLabel[uc.LABEL].values
    arrX, arrY = uc.preprocessFeatureLabel(arrX, arrY)
    
    # set up model 
#    strModelName = 'decision_tree'
#    dcModelParams = {"criterion": "gini"}
    strModelName = 'random_forest'
    dcModelParams = {'n_estimators':10, "criterion":"gini",
                   "max_features": "auto", "oob_score":True,
                   "n_jobs": -1, "warm_start": False}

    # static vs. static state
    lsCriterion = dfFeatureLabel.index.str.contains("v0_p0_m0_d1_l0")
    arrX_selected = arrX[lsCriterion]
    arrY_selected = arrY[lsCriterion]
    
    dcTemp = uc.crossValidate(arrX_selected, arrY_selected, strModelName,
                              dcModelParams, None, nFold)
    dcResult_s_s = {}
    dcResult_s_s[uc.MODEL_PRECISION_MEAN] = \
        (summarizeCVResult(dcTemp, uc.MODEL_PRECISION) ) \
        [uc.MODEL_PRECISION_MEAN]
    dcResult_s_s[uc.MODEL_PRECISION_STD] = \
        (summarizeCVResult(dcTemp, uc.MODEL_PRECISION) ) \
        [uc.MODEL_PRECISION_STD]
        
    dcResult_s_s[uc.MODEL_RECALL_MEAN] = \
        (summarizeCVResult(dcTemp, uc.MODEL_RECALL) ) \
        [uc.MODEL_RECALL_MEAN]
    dcResult_s_s[uc.MODEL_RECALL_STD] = \
        (summarizeCVResult(dcTemp, uc.MODEL_RECALL) ) \
        [uc.MODEL_RECALL_STD]
    
    dcResult[MOBILITY_S_S] = dcResult_s_s
    
    
    # static vs. mobile
    arrTraningMask = dfFeatureLabel.index.str.contains("v0_p0_m0_d1_l0")
    arrX_train = arrX[arrTraningMask]
    arrY_train = arrY[arrTraningMask]
    arrTestingMask = dfFeatureLabel.index.str.contains("v1_p0_m0_d1_l0")
    arrX_test = arrX[arrTestingMask]
    arrY_test = arrY[arrTestingMask]
    
    lsFoldResult = []
    for i in xrange(nFold):
        model = uc.trainModel(strModelName, dcModelParams, 
                              arrX_train, arrY_train)
        dcTemp = uc.testModel(model, arrX_test, arrY_test, lsFeatureNames)
        lsFoldResult.append(dcTemp)
    dcResult_s_m = {}
    dcResult_s_m[uc.MODEL_PRECISION_MEAN] = np.mean( [ dc[uc.MODEL_PRECISION] \
                                                  for dc in lsFoldResult] )
    dcResult_s_m[uc.MODEL_PRECISION_STD] = np.std( [ dc[uc.MODEL_PRECISION] \
                                                  for dc in lsFoldResult] )
                                                    
    dcResult_s_m[uc.MODEL_RECALL_MEAN] = np.mean( [ dc[uc.MODEL_RECALL] \
                                                  for dc in lsFoldResult] )
    dcResult_s_m[uc.MODEL_RECALL_STD] = np.std( [ dc[uc.MODEL_RECALL] \
                                                  for dc in lsFoldResult] )
    dcResult[MOBILITY_S_M] = dcResult_s_m
    
    
    # mixed vs. mixed state
    lsCriterion = dfFeatureLabel.index.str.contains("v0_p0_m0_d1_l0") \
                | dfFeatureLabel.index.str.contains("v1_p0_m0_d1_l0")
    arrX_selected = arrX[lsCriterion]
    arrY_selected = arrY[lsCriterion]
    
    dcTemp = uc.crossValidate(arrX_selected, arrY_selected, strModelName,
                              dcModelParams, None, nFold)
    dcResult_x_x = {}
    dcResult_x_x[uc.MODEL_PRECISION_MEAN] = \
        (summarizeCVResult(dcTemp, uc.MODEL_PRECISION) ) \
        [uc.MODEL_PRECISION_MEAN]
    dcResult_x_x[uc.MODEL_PRECISION_STD] = \
        (summarizeCVResult(dcTemp, uc.MODEL_PRECISION) ) \
        [uc.MODEL_PRECISION_STD]
        
    dcResult_x_x[uc.MODEL_RECALL_MEAN] = \
        (summarizeCVResult(dcTemp, uc.MODEL_RECALL) ) \
        [uc.MODEL_RECALL_MEAN]
    dcResult_x_x[uc.MODEL_RECALL_STD] = \
        (summarizeCVResult(dcTemp, uc.MODEL_RECALL) ) \
        [uc.MODEL_RECALL_STD]
    
    dcResult[MOBILITY_X_X] = dcResult_x_x

    # plot
    dfResult = pd.DataFrame(dcResult).T
    dfResult = dfResult.reindex([MOBILITY_S_S, MOBILITY_S_M, MOBILITY_X_X])
    
    ax0 = plt.figure(figsize=(5,4) ).add_subplot(111 )
    dBarWidth = 0.34
    dBarInterval = 0.07 
    arrInd = np.arange(dfResult.shape[0])
    
    ax0.bar(arrInd, dfResult[uc.MODEL_PRECISION_MEAN].values, 
            hatch=lsHatch[0],
            width=dBarWidth, color=strColor_light_1, label=uc.MODEL_PRECISION)
            
    ax0.bar(arrInd+dBarWidth+dBarInterval, 
            dfResult[uc.MODEL_RECALL_MEAN].values, 
            hatch=lsHatch[1],
            width=dBarWidth, color=strColor_dark_1, label=uc.MODEL_RECALL)
    
    
    # decorate figure
    ax0.set_xlim(0, dfResult.shape[0])
    ax0.set_ylim(0.0, 1.1)
    ax0.set_xticks(arrInd+dBarWidth+dBarInterval)
    ax0.set_xticklabels(dfResult.index.tolist(), 
                       fontname=strBasicFontName,
                       fontsize=nBasicFontSize)
                   
    ax0.set_xlabel("Training scenario - testing scenario",
                   fontname=strBasicFontName,
                   fontsize=nBasicFontSize)
    ax0.set_ylabel("Peformance", fontname=strBasicFontName,
                  fontsize=nBasicFontSize)
    arrYTicks = ax0.yaxis.get_ticklabels()
    plt.setp(arrYTicks, fontname=strBasicFontName,
             size=nBasicFontSize)
    ax0.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    return dfResult
    

def evaluateWearingLocation(strDataDir, lsFileNames,
                            lsDataAxisName, 
                            dcLabel, nFold=10):
    """
        Evaluate impact of install location
        1. customize feature
        2. select training & test data
        3. train & test
    """
    dValidSegDuration = 1.4
    lsFullSegment = range(0, 13)
    lsValidSegID = lsFullSegment
        
    print("-->extracting features & labels...")
    dfFeatureLabel = uc.customizeFeatureLabel(strDataDir, 
                                              lsFileNames, 
                                              lsDataAxisName,
                                              dSamplingFreq,
                                              dcLabel, 
                                              nResponsePerData=3,
                                              nSegmentPerRespsonse = 13,
                                  dVibrationDuration = dValidSegDuration, 
                                  dIntervalDuration=0.0,
                                  dRestDuration=1.0,
                                  lsValidSegmentID=lsValidSegID,
                                  dValidSegmentDuration=dValidSegDuration,
                                  lsAxis2Inspect=['x0', 'y0', 'z0'])
    # preprocess features
    lsFeatureNames = [strCol for strCol in dfFeatureLabel.columns \
                    if strCol != uc.LABEL]
    arrX = dfFeatureLabel[lsFeatureNames].as_matrix()
    arrY = dfFeatureLabel[uc.LABEL].values
    arrX, arrY = uc.preprocessFeatureLabel(arrX, arrY)
    
    # set up model 
#    strModelName = 'decision_tree'
#    dcModelParams = {"criterion": "gini"}
    strModelName = 'random_forest'
    dcModelParams = {'n_estimators':100, "criterion":"gini",
                   "max_features": "auto", "oob_score":True,
                   "n_jobs": -1, "warm_start": True}
                   
    

    # training: 0 cm, test: multiple distance
    dcResult = {}
    lsCandidateDistances = [1, 3, 2, 4]
    for nTrainingSpan in xrange(1, len(lsCandidateDistances)+1, 1):
        # train model with distance >= 1 and < span (only use files
        # with specific ids)
        lsTrainingFileIDs = np.random.choice([0,1,2], 2, replace=False)
        lsTrainingCriterion = ["v0_p0_m0_d%d_l0_%d"%(nDis, nFileID) \
            for nDis in lsCandidateDistances[:nTrainingSpan] \
            for nFileID in lsTrainingFileIDs]
                
        print lsTrainingCriterion
        
        arrTraningMask = np.array([False,]*dfFeatureLabel.shape[0] )
        for strTrainingCriterion in lsTrainingCriterion:
            arrTraningMask = arrTraningMask | \
                dfFeatureLabel.index.str.contains(strTrainingCriterion)
        arrX_train = arrX[arrTraningMask]
        arrY_train = arrY[arrTraningMask]
        
        # test with different distance
        for nTestDistance in lsCandidateDistances:
            lsTestingFileIDs = list(set(range(5)) - set(lsTrainingFileIDs))
            lsTestingCriterion = ["v0_p0_m0_d%d_l0_%d"%\
                (nTestDistance, nFileID) \
                for nFileID in lsTestingFileIDs ]
                    
            arrTestingMask = np.array([False,]*dfFeatureLabel.shape[0] )
            for strTestingCriterion in lsTestingCriterion:
                arrTestingMask = arrTestingMask | \
                    dfFeatureLabel.index.str.contains(strTestingCriterion)
                            
            arrX_test = arrX[arrTestingMask]
            arrY_test = arrY[arrTestingMask]
            
            lsFoldResult = []
            for i in xrange(nFold):
                model = uc.trainModel(strModelName, dcModelParams, 
                                  arrX_train, arrY_train)
                dcTemp = uc.testModel(model, arrX_test, arrY_test, 
                                      lsFeatureNames)
                lsFoldResult.append(dcTemp)
            dPrecision_mean = np.mean([dc[uc.MODEL_PRECISION] \
                                for dc in lsFoldResult] )    
            dPrecision_std = np.std([dc[uc.MODEL_PRECISION] \
                                for dc in lsFoldResult] )    
            dRecall_mean = np.mean([dc[uc.MODEL_RECALL] \
                                for dc in lsFoldResult] )    
            dRecall_std = np.std([dc[uc.MODEL_RECALL] \
                                for dc in lsFoldResult] )    
            dcResult[(nTrainingSpan, nTestDistance)] = \
                {uc.MODEL_PRECISION_MEAN: dPrecision_mean,
                 uc.MODEL_PRECISION_STD: dPrecision_std,
                 uc.MODEL_RECALL_MEAN: dRecall_mean,
                 uc.MODEL_RECALL_STD: dRecall_std}

    dfResult = pd.DataFrame.from_dict(dcResult, orient='index')
    
   
    # plot
    dBarWidth = 0.15
    dBarInterval = 0.05 
    # precision
    ax = plt.figure(figsize=(5,4)).add_subplot(111)
    for i, nTrainingSpan in enumerate(dfResult.index.levels[0] ):
        dfResultSlice = dfResult.loc[nTrainingSpan]
        arrInd = dfResultSlice.index
        ax.bar(arrInd+i*(dBarWidth+dBarInterval),
               dfResultSlice[uc.MODEL_PRECISION_MEAN].values, 
               color=lsColorPalette[i], width=dBarWidth,
               hatch=lsHatch[i],
               label=r"#loc.$\leq$%d"%nTrainingSpan )
                    
        # decorate figure
        ax.set_xticks(arrInd+ 2*(dBarWidth+dBarInterval))
        ax.set_xticklabels(arrInd-1, 
                           fontname=strBasicFontName,
                           fontsize=nBasicFontSize)
        arrXTicks = ax.xaxis.get_ticklabels()
        plt.setp(arrXTicks, fontname=strBasicFontName,
                 size=nBasicFontSize)
        ax.set_xlabel("Wearing location displacement (cm)",
                      fontname=strBasicFontName,
                      fontsize=nBasicFontSize)
        
        ax.set_ylim(0.0, 1.1)               
        arrYTicks = ax.yaxis.get_ticklabels()
        plt.setp(arrYTicks, fontname=strBasicFontName,
                 size=nBasicFontSize)
        ax.set_ylabel(uc.MODEL_PRECISION, fontname=strBasicFontName,
                      fontsize=nBasicFontSize)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    plt.show()   
    
    
    # recall
    ax = plt.figure(figsize=(5,4)).add_subplot(111)
    for i, nTrainingSpan in enumerate(dfResult.index.levels[0] ):
        dfResultSlice = dfResult.loc[nTrainingSpan]
        arrInd = dfResultSlice.index
        ax.bar(arrInd+i*(dBarWidth+dBarInterval),
               dfResultSlice[uc.MODEL_RECALL_MEAN],
               color=lsColorPalette[i],
               width=dBarWidth,
               hatch=lsHatch[i],
               label=r"#loc.$\leq$%d"%nTrainingSpan,
               align='center')
                    
        # decorate figure
        ax.set_xticks(arrInd+ 2*(dBarWidth+dBarInterval))
        ax.set_xticklabels(arrInd-1, 
                           fontname=strBasicFontName,
                           fontsize=nBasicFontSize)
        arrXTicks = ax.xaxis.get_ticklabels()
        plt.setp(arrXTicks, fontname=strBasicFontName,
                 size=nBasicFontSize)
        ax.set_xlabel("Wearing location displacement (cm)", 
                      fontname=strBasicFontName,
                      fontsize=nBasicFontSize)
        
        ax.set_ylim(0.0, 1.1)               
        arrYTicks = ax.yaxis.get_ticklabels()
        plt.setp(arrYTicks, fontname=strBasicFontName,
                 size=nBasicFontSize)
        ax.set_ylabel(uc.MODEL_RECALL, fontname=strBasicFontName,
                      fontsize=nBasicFontSize)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    plt.show()  
    
    return dfResult
    
def overall_test(strDataDir, lsFileNames, lsDataAxisName,
                 dcLabel, 
                 strModelName, 
                 dcModelParams, 
                 nFold=10):
    """
        Evaluate impact of muscle state
        1. customize feature
        2. select training & test data
        3. train & test
    """
    dValidSegDuration = 1.4
    lsFullSegment = range(0, 13)
    lsValidSegID = lsFullSegment
        
    print("-->extracting features & labels...")
    dfFeatureLabel = uc.customizeFeatureLabel(strDataDir, 
                                              lsFileNames, 
                                              lsDataAxisName,
                                              dSamplingFreq,
                                              dcLabel, 
                                              nResponsePerData=3,
                                              nSegmentPerRespsonse = 13,
                                  dVibrationDuration = dValidSegDuration, 
                                  dIntervalDuration=0.0,
                                  dRestDuration=1.0,
                                  lsValidSegmentID=lsValidSegID,
                                  dValidSegmentDuration=dValidSegDuration,
                                  lsAxis2Inspect=['x0', 'y0', 'z0'])
    # preprocess features
    lsFeatureNames = [strCol for strCol in dfFeatureLabel.columns \
                    if strCol != uc.LABEL]
    arrX = dfFeatureLabel[lsFeatureNames].as_matrix()
    arrY = dfFeatureLabel[uc.LABEL].values
    arrX, arrY = uc.preprocessFeatureLabel(arrX, arrY)
    
    # set up model 

    # cross validate on all data
    arrX_selected = arrX
    arrY_selected = arrY
    
    dcResult = uc.crossValidate(arrX_selected, arrY_selected, strModelName,
                              dcModelParams, lsFeatureNames, nFold)

    dfResult = pd.DataFrame.from_dict(dcResult, orient='index')
    print dfResult
    
    
    arrAvgNormalizedCM = None          
    for nFoldID, dcFoldResult in dcResult.iteritems():
        arrCM = dcFoldResult[uc.MODEL_CONFUSION_MATRIX]
        arrNormalizedCM = arrCM.astype('float') \
            / arrCM.sum(axis=1)[:, np.newaxis]
        arrAvgNormalizedCM = arrAvgNormalizedCM + arrNormalizedCM \
            if arrAvgNormalizedCM is not None else arrNormalizedCM
    
    arrAvgNormalizedCM = arrAvgNormalizedCM / len(dcResult)

    # plot
    fig = plt.figure(figsize=(5,4) )
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(arrAvgNormalizedCM), cmap=plt.cm.rainbow, 
                    interpolation='nearest')
    
    width = len(arrAvgNormalizedCM)
    height = len(arrAvgNormalizedCM[0])
    
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate("%.2f"% arrAvgNormalizedCM[x][y], xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center', 
                        fontname=strBasicFontName,
                        fontsize=nBasicFontSize)
    
    cb = fig.colorbar(res)
    
    # decorate figure
    arrXTicks = ax.xaxis.get_ticklabels()
    plt.setp(arrXTicks, fontname=strBasicFontName,
             size=nBasicFontSize)
    arrYTicks = ax.yaxis.get_ticklabels()
    plt.setp(arrYTicks, fontname=strBasicFontName,
             size=nBasicFontSize)
             
    arrBarXTicks = cb.ax.xaxis.get_ticklabels()
    plt.setp(arrBarXTicks, fontname=strBasicFontName,
             size=nBasicFontSize)
    arrBarYTicks = cb.ax.yaxis.get_ticklabels()
    plt.setp(arrBarYTicks, fontname=strBasicFontName,
             size=nBasicFontSize)
             
     
    plt.grid(True, which='minor')
    plt.tight_layout()
    plt.show()   
    
    return dcResult, arrAvgNormalizedCM

def simulateExcitation():
    dSamplingFreq = 2000.0
    dDuration = 0.5
    arrTime = np.linspace(0.0, 14*dDuration, 14*dDuration*dSamplingFreq) # time
    
     # input voltage increases with time
    nPointPerSecond = int(dSamplingFreq)
    
    arrInputVoltage = np.array( \
                      [0]*int(dDuration*nPointPerSecond) + \
                      [30]*int(dDuration*nPointPerSecond) + \
                      [40]*int(dDuration*nPointPerSecond) + \
                      [50]*int(dDuration*nPointPerSecond) + \
                      [60]*int(dDuration*nPointPerSecond) + \
                      [70]*int(dDuration*nPointPerSecond) + \
                      [80]*int(dDuration*nPointPerSecond) + \
                      [90]*int(dDuration*nPointPerSecond) + \
                      [100]*int(dDuration*nPointPerSecond) + \
                      [110]*int(dDuration*nPointPerSecond) + \
                      [120]*int(dDuration*nPointPerSecond) + \
                      [130]*int(dDuration*nPointPerSecond) + \
                      [140]*int(dDuration*nPointPerSecond) + \
                      [0]*int(dDuration*nPointPerSecond) )
                      
    
    # the relationship btw vibration frequency and input voltage
    # is estimated via experiments
    arrVibFrequency = 0.9125* arrInputVoltage - 3.875
    print arrVibFrequency
    
    # F = m*r*w^2, where m in KG, r in meter, and w in rad/sec
    arrVibForce = 1000* 0.01 * 0.02 * \
                  (arrVibFrequency*2.0*np.pi)**2.0  # excitation amplititude
                                                    # increases with time
                  
                  
    # plot input force
    ax = plt.figure(figsize=(5,4 ) ).add_subplot(111)
    plt.plot(arrTime, arrVibForce*np.sin(2*np.pi*arrVibFrequency*arrTime))
    # decorate figure
    ax.set_xlim(0.3, 6.8)
    ax.set_xlabel("Time (seconds)",
                  fontname=strBasicFontName,
                  fontsize=nBasicFontSize)
    xTickLabels = ax.xaxis.get_ticklabels()
    plt.setp(xTickLabels, fontname=strBasicFontName,
             size=nBasicFontSize)
    ax.set_ylabel("Excitation", 
                  fontname=strBasicFontName,
                  fontsize=nBasicFontSize)
    arrYTicks = ax.yaxis.get_ticklabels()
    plt.setp(arrYTicks, fontname=strBasicFontName,
             size=nBasicFontSize)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0) )
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0) )
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
def fftUserMobility():
    strWorkingDir = "../../data/experiment/user_motion/"
    strFileName = "arm_motion"
    dfData = sd.loadData(strWorkingDir, strFileName, lsDataAxisName)
    
    lsAxis2Inspect = ['x0', 'y0', 'z0']
    
    dfData2FFT = dfData[lsAxis2Inspect]
    
    nDCEnd = 5
    nFFTStart = 0
    nFFTEnd = dfData2FFT.shape[0]
    
    nFFTBatchStart = nFFTStart
    nFFTBatchEnd = nFFTBatchStart
    nFFTBatchSize = int(dSamplingFreq*200)
    
    while(nFFTBatchStart<nFFTEnd ):
        nFFTBatchEnd = min( (nFFTBatchStart+nFFTBatchSize), nFFTEnd)
        
        print nFFTBatchStart, nFFTBatchEnd
        fig, axes = plt.subplots(nrows=len(lsAxis2Inspect), 
                                 ncols=1, squeeze=False, figsize=(5,4),
                                 sharex=True )
        for i, strCol in enumerate(dfData2FFT.columns):
            srAxis = dfData2FFT.ix[nFFTBatchStart:nFFTBatchEnd, strCol]
    
            # fft
            arrTimeData = srAxis.values
            nSamples = len(arrTimeData)
            arrFreqData = fftpack.fft(arrTimeData)
            arrNormalizedPower = abs(arrFreqData)/(nSamples*1.0)
    
            dResolution = dSamplingFreq*1.0/nSamples
            arrFreqIndex = np.linspace(nDCEnd*dResolution, 
                                       dSamplingFreq/2.0, nSamples/2-nDCEnd)
            axes[i, 0].plot(arrFreqIndex, 
                            arrNormalizedPower[nDCEnd:nSamples/2],
                            color=lsBasicColors[i])
            # decorate figure
            axes[i, 0].set_xlim(0,50)
            axes[i, 0].xaxis.set_ticks(range(0, 55, 10) )
            xTickLabels = axes[i, 0].xaxis.get_ticklabels()
            plt.setp(xTickLabels, fontname=strBasicFontName,
                     size=nBasicFontSize)
                     
            axes[i, 0].yaxis.set_ticks(range(0, 1000, 300) )
            axes[i, 0].set_ylabel(strCol, 
                          fontname=strBasicFontName,
                          fontsize=nBasicFontSize)
            arrYTicks = axes[i, 0].yaxis.get_ticklabels()
            plt.setp(arrYTicks, fontname=strBasicFontName,
                     size=nBasicFontSize)
        axes[2, 0].set_xlabel("Frequency (Hz)",
                          fontname=strBasicFontName,
                          fontsize=nBasicFontSize)    
        fig.tight_layout()
        plt.show()
        nFFTBatchStart = nFFTBatchEnd
    
def simulateResonance():
    
    w = np.linspace(0.0, 60.0, 60)
    
    wn = 30.0
    
    arrInd = w/wn
    
    ax = plt.figure(figsize=(5, 4) ).add_subplot(111)
    for i, xi in enumerate(np.linspace(0.1, 0.6, 5) ):
        A = 1.0 / np.sqrt( (1-(w/wn)**2)**2 + (2*xi*w/wn)**2 )
        ax.plot(arrInd, A, label=r"$ \xi $ = %.1f" % xi, marker=lsMarkers[i])
    # decorate figure
    ax.set_ylim(0.0, 5.5)
    ax.set_xlabel(r"$\beta$",
                  fontname=strBasicFontName,
                  fontsize=nBasicFontSize)
    xTickLabels = ax.xaxis.get_ticklabels()
    plt.setp(xTickLabels, fontname=strBasicFontName,
             size=nBasicFontSize)
    ax.set_ylabel("Amplification Ratio", 
                  fontname=strBasicFontName,
                  fontsize=nBasicFontSize)
    arrYTicks = ax.yaxis.get_ticklabels()
    plt.setp(arrYTicks, fontname=strBasicFontName,
             size=nBasicFontSize)
#    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0) )
#    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0) )
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # basic plot setting
    lsBasicColors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    strColor_light_1 = '#c0c5ce'
    strColor_light_2 = '#747982'
    
    strColor_dark_1 = '#65737e'
    strColor_dark_2 = '#4f5b66'
    
    # space-gray like Color Palette 
    # (http://www.color-hex.com/color-palette/2280)
    strColor_1_1 = "#c0c5ce"
    strColor_2_1 = "#a7adba"
    strColor_3_1 = "#74818a"
    strColor_4_1 = "#4f5b66"
    
    lsColorPalette = [strColor_1_1, strColor_2_1,
                      strColor_3_1, strColor_4_1]
                           
    lsHatch = ['/', '-', 'x', '\\']
    
    lsMarkers = ['+', 'o', 'x', '*', 'd', 'x', '1', '2', '3', '4']
    strBasicFontName = "Times new Roman"
    nBasicFontSize = 16
    
    # data setting
    lsDataAxisName = ['x0', 'y0','z0', 'gx0', 'gy0','gz0']
    dSamplingFreq = 320.0

    # classification setting
    dcLabel = {"yl":1, "cy":2, "hc":3, "ww":4,
               "qy":5, "hy":6, "zy":7, "ch":8,
               "xh":9, "lq":10}
               
    nFold = 5   

    dBarWidth = 0.34
    dBarInterval = 0.07  


#%% overall test
    strDataDir = "../../data/experiment/overall_test/"
    lsFileNames = ds.lsLQ_t1_v0_p0_m0_d1_l0 + \
                  ds.lsLQ_t1_v0_p0_m0_d2_l0 + \
                  ds.lsLQ_t1_v0_p0_m0_d3_l0 + \
                  ds.lsLQ_t1_v0_p0_m0_d4_l0 + \
                  ds.lsLQ_t1_v0_p30_m0_d1_l0 + \
                  ds.lsLQ_t1_v0_p330_m0_d1_l0 + \
                  ds.lsLQ_t1_v0_p0_m1_d1_l0 + \
                  ds.lsLQ_t1_v1_p0_m0_d1_l0 + \
                  ds.lsYL_t36_v0_p0_m0_d1_l0 + \
                  ds.lsYL_t36_v0_p0_m0_d2_l0 + \
                  ds.lsYL_t36_v0_p0_m0_d3_l0 + \
                  ds.lsYL_t36_v0_p30_m0_d1_l0 + \
                  ds.lsYL_t36_v0_p60_m0_d1_l0 + \
                  ds.lsYL_t36_v0_p330_m0_d1_l0 + \
                  ds.lsYL_t36_v0_p270_m0_d1_l0 + \
                  ds.lsWW_t11_v0_p0_m0_d1_l0 + \
                  ds.lsWW_t11_v0_p0_m0_d2_l0 + \
                  ds.lsWW_t11_v0_p0_m0_d2_l1 + \
                  ds.lsWW_t11_v0_p0_m0_d3_l0 + \
                  ds.lsWW_t11_v0_p0_m0_d3_l1 + \
                  ds.lsWW_t11_v0_p0_m1_d1_l0 + \
                  ds.lsWW_t11_v0_p30_m0_d1_l0 + \
                  ds.lsWW_t11_v0_p330_m0_d1_l0 + \
                  ds.lsWW_t11_v1_p0_m0_d1_l0 + \
                  ds.lsQY_t7_v0_p0_m0_d1_l0 + \
                  ds.lsQY_t7_v0_p0_m0_d2_l0 + \
                  ds.lsQY_t7_v0_p0_m0_d2_l1 + \
                  ds.lsQY_t7_v0_p0_m0_d3_l0 + \
                  ds.lsQY_t7_v0_p0_m0_d3_l1 + \
                  ds.lsQY_t7_v0_p0_m1_d1_l0 + \
                  ds.lsQY_t7_v0_p30_m0_d1_l0 + \
                  ds.lsQY_t7_v0_p330_m0_d1_l0 + \
                  ds.lsQY_t7_v1_p0_m0_d1_l0 + \
                  ds.lsHCY_t10_v0_p0_m0_d1_l0 + \
                  ds.lsHCY_t10_v0_p0_m0_d2_l0 + \
                  ds.lsHCY_t10_v0_p0_m0_d3_l0 + \
                  ds.lsHCY_t10_v0_p0_m0_d3_l0 + \
                  ds.lsHCY_t10_v0_p0_m0_d4_l0 + \
                  ds.lsHCY_t10_v0_p0_m1_d1_l0 + \
                  ds.lsHCY_t10_v0_p30_m0_d1_l0 + \
                  ds.lsHCY_t10_v0_p330_m0_d1_l0 + \
                  ds.lsHCY_t10_v0_p60_m0_d1_l0 + \
                  ds.lsHCY_t10_v1_p0_m0_d1_l0 + \
                  ds.lsCHX_t5_v0_p0_m0_d1_l0 + \
                  ds.lsCHX_t5_v0_p0_m0_d2_l0 + \
                  ds.lsCHX_t5_v0_p0_m0_d3_l0 + \
                  ds.lsCHX_t5_v0_p0_m0_d3_l0 + \
                  ds.lsCHX_t5_v0_p0_m0_d4_l0 + \
                  ds.lsCHX_t5_v0_p0_m1_d1_l0 + \
                  ds.lsCHX_t5_v0_p30_m0_d1_l0 + \
                  ds.lsCHX_t5_v0_p330_m0_d1_l0 + \
                  ds.lsCHX_t5_v0_p60_m0_d1_l0 + \
                  ds.lsCHX_t5_v1_p0_m0_d1_l0 + \
                  ds.lsCYJ_t20_v0_p0_m0_d1_l0 + \
                  ds.lsCYJ_t20_v0_p0_m0_d2_l0 + \
                  ds.lsCYJ_t20_v0_p0_m0_d3_l0 + \
                  ds.lsCYJ_t20_v0_p0_m0_d4_l0 + \
                  ds.lsCYJ_t20_v0_p0_m1_d1_l0 + \
                  ds.lsCYJ_t20_v0_p30_m0_d1_l0 + \
                  ds.lsCYJ_t20_v0_p330_m0_d1_l0 + \
                  ds.lsCYJ_t20_v1_p0_m0_d1_l0 + \
                  ds.lsCYJ_t20_v0_p0_m0_d1_l0 + \
                  ds.lsXH_t1_v0_p0_m0_d1_l0 + \
                  ds.lsXH_t1_v0_p0_m0_d2_l0 + \
                  ds.lsXH_t1_v0_p0_m0_d3_l0 + \
                  ds.lsXH_t1_v0_p0_m0_d4_l0 + \
                  ds.lsXH_t1_v0_p0_m1_d1_l0 + \
                  ds.lsXH_t1_v0_p30_m0_d1_l0 + \
                  ds.lsXH_t1_v0_p330_m0_d1_l0 + \
                  ds.lsXH_t1_v1_p0_m0_d1_l0
    
    strModelName = 'random_forest'
    dcModelParams = {'n_estimators':5, "criterion":"entropy",
                     "max_features": "auto", "oob_score":True,
                     "n_jobs": -1, "warm_start": False}   
#    strModelName = 'decision_tree'
#    dcModelParams = {"criterion": "gini"}

    lsOverallResult = []
    for i in xrange(5):
        dcResult, mtNormalizedCM = overall_test(strDataDir, 
                                            lsFileNames, 
                                            lsDataAxisName,
                                            dcLabel, 
                                            strModelName,
                                            dcModelParams,
                                            nFold=3)
        lsOverallResult.append(dcResult)
                
##%% span of excitation frequency
#    strDataDir = "../../data/experiment/overall_test/"
#    lsFileNames = ds.lsLQ_t1_v0_p0_m0_d1_l0 + \
#                  ds.lsLQ_t1_v0_p0_m0_d2_l0 + \
#                  ds.lsLQ_t1_v0_p0_m0_d3_l0 + \
#                  ds.lsLQ_t1_v0_p0_m0_d4_l0 + \
#                  ds.lsLQ_t1_v0_p30_m0_d1_l0 + \
#                  ds.lsLQ_t1_v0_p330_m0_d1_l0 + \
#                  ds.lsLQ_t1_v0_p0_m1_d1_l0 + \
#                  ds.lsLQ_t1_v1_p0_m0_d1_l0 + \
#                  ds.lsYL_t36_v0_p0_m0_d1_l0 + \
#                  ds.lsYL_t36_v0_p0_m0_d2_l0 + \
#                  ds.lsYL_t36_v0_p0_m0_d3_l0 + \
#                  ds.lsYL_t36_v0_p30_m0_d1_l0 + \
#                  ds.lsYL_t36_v0_p60_m0_d1_l0 + \
#                  ds.lsYL_t36_v0_p330_m0_d1_l0 + \
#                  ds.lsYL_t36_v0_p270_m0_d1_l0 + \
#                  ds.lsWW_t11_v0_p0_m0_d1_l0 + \
#                  ds.lsWW_t11_v0_p0_m0_d2_l0 + \
#                  ds.lsWW_t11_v0_p0_m0_d2_l1 + \
#                  ds.lsWW_t11_v0_p0_m0_d3_l0 + \
#                  ds.lsWW_t11_v0_p0_m0_d3_l1 + \
#                  ds.lsWW_t11_v0_p0_m1_d1_l0 + \
#                  ds.lsWW_t11_v0_p30_m0_d1_l0 + \
#                  ds.lsWW_t11_v0_p330_m0_d1_l0 + \
#                  ds.lsWW_t11_v1_p0_m0_d1_l0 + \
#                  ds.lsQY_t7_v0_p0_m0_d1_l0 + \
#                  ds.lsQY_t7_v0_p0_m0_d2_l0 + \
#                  ds.lsQY_t7_v0_p0_m0_d2_l1 + \
#                  ds.lsQY_t7_v0_p0_m0_d3_l0 + \
#                  ds.lsQY_t7_v0_p0_m0_d3_l1 + \
#                  ds.lsQY_t7_v0_p0_m1_d1_l0 + \
#                  ds.lsQY_t7_v0_p30_m0_d1_l0 + \
#                  ds.lsQY_t7_v0_p330_m0_d1_l0 + \
#                  ds.lsQY_t7_v1_p0_m0_d1_l0 + \
#                  ds.lsHCY_t10_v0_p0_m0_d1_l0 + \
#                  ds.lsHCY_t10_v0_p0_m0_d2_l0 + \
#                  ds.lsHCY_t10_v0_p0_m0_d3_l0 + \
#                  ds.lsHCY_t10_v0_p0_m0_d3_l0 + \
#                  ds.lsHCY_t10_v0_p0_m0_d4_l0 + \
#                  ds.lsHCY_t10_v0_p0_m1_d1_l0 + \
#                  ds.lsHCY_t10_v0_p30_m0_d1_l0 + \
#                  ds.lsHCY_t10_v0_p330_m0_d1_l0 + \
#                  ds.lsHCY_t10_v0_p60_m0_d1_l0 + \
#                  ds.lsHCY_t10_v1_p0_m0_d1_l0 + \
#                  ds.lsCHX_t5_v0_p0_m0_d1_l0 + \
#                  ds.lsCHX_t5_v0_p0_m0_d2_l0 + \
#                  ds.lsCHX_t5_v0_p0_m0_d3_l0 + \
#                  ds.lsCHX_t5_v0_p0_m0_d3_l0 + \
#                  ds.lsCHX_t5_v0_p0_m0_d4_l0 + \
#                  ds.lsCHX_t5_v0_p0_m1_d1_l0 + \
#                  ds.lsCHX_t5_v0_p30_m0_d1_l0 + \
#                  ds.lsCHX_t5_v0_p330_m0_d1_l0 + \
#                  ds.lsCHX_t5_v0_p60_m0_d1_l0 + \
#                  ds.lsCHX_t5_v1_p0_m0_d1_l0 + \
#                  ds.lsCYJ_t20_v0_p0_m0_d1_l0 + \
#                  ds.lsCYJ_t20_v0_p0_m0_d2_l0 + \
#                  ds.lsCYJ_t20_v0_p0_m0_d3_l0 + \
#                  ds.lsCYJ_t20_v0_p0_m0_d4_l0 + \
#                  ds.lsCYJ_t20_v0_p0_m1_d1_l0 + \
#                  ds.lsCYJ_t20_v0_p30_m0_d1_l0 + \
#                  ds.lsCYJ_t20_v0_p330_m0_d1_l0 + \
#                  ds.lsCYJ_t20_v1_p0_m0_d1_l0 + \
#                  ds.lsCYJ_t20_v0_p0_m0_d1_l0 + \
#                  ds.lsXH_t1_v0_p0_m0_d1_l0 + \
#                  ds.lsXH_t1_v0_p0_m0_d2_l0 + \
#                  ds.lsXH_t1_v0_p0_m0_d3_l0 + \
#                  ds.lsXH_t1_v0_p0_m0_d4_l0 + \
#                  ds.lsXH_t1_v0_p0_m1_d1_l0 + \
#                  ds.lsXH_t1_v0_p30_m0_d1_l0 + \
#                  ds.lsXH_t1_v0_p330_m0_d1_l0 + \
#                  ds.lsXH_t1_v1_p0_m0_d1_l0
#
#
#    lsFreqResults = []
#    for i in xrange(5):
#        dcEvaluationResult = evaluateExcitationFrequency(strDataDir, 
#                                                     lsFileNames,
#                                                     lsDataAxisName,
#                                                     dcLabel, nFold=5)
#        lsFreqResults.append(dcEvaluationResult)
#
##%% duration of excitation duration
#    strDataDir = "../../data/experiment/overall_test/"
#    lsFileNames = ds.lsLQ_t1_v0_p0_m0_d1_l0 + \
#                  ds.lsLQ_t1_v0_p0_m0_d2_l0 + \
#                  ds.lsLQ_t1_v0_p0_m0_d3_l0 + \
#                  ds.lsLQ_t1_v0_p0_m0_d4_l0 + \
#                  ds.lsLQ_t1_v0_p30_m0_d1_l0 + \
#                  ds.lsLQ_t1_v0_p330_m0_d1_l0 + \
#                  ds.lsLQ_t1_v0_p0_m1_d1_l0 + \
#                  ds.lsLQ_t1_v1_p0_m0_d1_l0 + \
#                  ds.lsYL_t36_v0_p0_m0_d1_l0 + \
#                  ds.lsYL_t36_v0_p0_m0_d2_l0 + \
#                  ds.lsYL_t36_v0_p0_m0_d3_l0 + \
#                  ds.lsYL_t36_v0_p30_m0_d1_l0 + \
#                  ds.lsYL_t36_v0_p60_m0_d1_l0 + \
#                  ds.lsYL_t36_v0_p330_m0_d1_l0 + \
#                  ds.lsYL_t36_v0_p270_m0_d1_l0 + \
#                  ds.lsWW_t11_v0_p0_m0_d1_l0 + \
#                  ds.lsWW_t11_v0_p0_m0_d2_l0 + \
#                  ds.lsWW_t11_v0_p0_m0_d2_l1 + \
#                  ds.lsWW_t11_v0_p0_m0_d3_l0 + \
#                  ds.lsWW_t11_v0_p0_m0_d3_l1 + \
#                  ds.lsWW_t11_v0_p0_m1_d1_l0 + \
#                  ds.lsWW_t11_v0_p30_m0_d1_l0 + \
#                  ds.lsWW_t11_v0_p330_m0_d1_l0 + \
#                  ds.lsWW_t11_v1_p0_m0_d1_l0 + \
#                  ds.lsQY_t7_v0_p0_m0_d1_l0 + \
#                  ds.lsQY_t7_v0_p0_m0_d2_l0 + \
#                  ds.lsQY_t7_v0_p0_m0_d2_l1 + \
#                  ds.lsQY_t7_v0_p0_m0_d3_l0 + \
#                  ds.lsQY_t7_v0_p0_m0_d3_l1 + \
#                  ds.lsQY_t7_v0_p0_m1_d1_l0 + \
#                  ds.lsQY_t7_v0_p30_m0_d1_l0 + \
#                  ds.lsQY_t7_v0_p330_m0_d1_l0 + \
#                  ds.lsQY_t7_v1_p0_m0_d1_l0 + \
#                  ds.lsHCY_t10_v0_p0_m0_d1_l0 + \
#                  ds.lsHCY_t10_v0_p0_m0_d2_l0 + \
#                  ds.lsHCY_t10_v0_p0_m0_d3_l0 + \
#                  ds.lsHCY_t10_v0_p0_m0_d3_l0 + \
#                  ds.lsHCY_t10_v0_p0_m0_d4_l0 + \
#                  ds.lsHCY_t10_v0_p0_m1_d1_l0 + \
#                  ds.lsHCY_t10_v0_p30_m0_d1_l0 + \
#                  ds.lsHCY_t10_v0_p330_m0_d1_l0 + \
#                  ds.lsHCY_t10_v0_p60_m0_d1_l0 + \
#                  ds.lsHCY_t10_v1_p0_m0_d1_l0 + \
#                  ds.lsCHX_t5_v0_p0_m0_d1_l0 + \
#                  ds.lsCHX_t5_v0_p0_m0_d2_l0 + \
#                  ds.lsCHX_t5_v0_p0_m0_d3_l0 + \
#                  ds.lsCHX_t5_v0_p0_m0_d3_l0 + \
#                  ds.lsCHX_t5_v0_p0_m0_d4_l0 + \
#                  ds.lsCHX_t5_v0_p0_m1_d1_l0 + \
#                  ds.lsCHX_t5_v0_p30_m0_d1_l0 + \
#                  ds.lsCHX_t5_v0_p330_m0_d1_l0 + \
#                  ds.lsCHX_t5_v0_p60_m0_d1_l0 + \
#                  ds.lsCHX_t5_v1_p0_m0_d1_l0 + \
#                  ds.lsCYJ_t20_v0_p0_m0_d1_l0 + \
#                  ds.lsCYJ_t20_v0_p0_m0_d2_l0 + \
#                  ds.lsCYJ_t20_v0_p0_m0_d3_l0 + \
#                  ds.lsCYJ_t20_v0_p0_m0_d4_l0 + \
#                  ds.lsCYJ_t20_v0_p0_m1_d1_l0 + \
#                  ds.lsCYJ_t20_v0_p30_m0_d1_l0 + \
#                  ds.lsCYJ_t20_v0_p330_m0_d1_l0 + \
#                  ds.lsCYJ_t20_v1_p0_m0_d1_l0 + \
#                  ds.lsCYJ_t20_v0_p0_m0_d1_l0 + \
#                  ds.lsXH_t1_v0_p0_m0_d1_l0 + \
#                  ds.lsXH_t1_v0_p0_m0_d2_l0 + \
#                  ds.lsXH_t1_v0_p0_m0_d3_l0 + \
#                  ds.lsXH_t1_v0_p0_m0_d4_l0 + \
#                  ds.lsXH_t1_v0_p0_m1_d1_l0 + \
#                  ds.lsXH_t1_v0_p30_m0_d1_l0 + \
#                  ds.lsXH_t1_v0_p330_m0_d1_l0 + \
#                  ds.lsXH_t1_v1_p0_m0_d1_l0
#
#    lsDurationResult = []
#    for i in xrange(5):
#        dfEvaluationResult = evaluateExcitationDuration(strDataDir, 
#                                                     lsFileNames,
#                                                     lsDataAxisName,
#                                                     dcLabel, nFold=5)
#        lsDurationResult.append(dfEvaluationResult)

#%% arm position
    strDataDir = "../../data/experiment/arm_position/"
    lsFileNames = ds.lsYL_t36_v0_p0_m0_d1_l0 + \
                  ds.lsYL_t36_v0_p30_m0_d1_l0 + \
                  ds.lsYL_t36_v0_p60_m0_d1_l0 + \
                  ds.lsYL_t36_v0_p330_m0_d1_l0 + \
                  ds.lsWW_t11_v0_p0_m0_d1_l0 + \
                  ds.lsWW_t11_v0_p30_m0_d1_l0 + \
                  ds.lsWW_t11_v0_p330_m0_d1_l0 + \
                  ds.lsQY_t7_v0_p0_m0_d1_l0 + \
                  ds.lsQY_t7_v0_p30_m0_d1_l0 + \
                  ds.lsQY_t7_v0_p330_m0_d1_l0 + \
                  ds.lsCHX_t5_v0_p0_m0_d1_l0 + \
                  ds.lsCHX_t5_v0_p30_m0_d1_l0 + \
                  ds.lsCHX_t5_v0_p60_m0_d1_l0 + \
                  ds.lsCHX_t5_v0_p330_m0_d1_l0 + \
                  ds.lsHCY_t10_v0_p0_m0_d1_l0 + \
                  ds.lsHCY_t10_v0_p30_m0_d1_l0 + \
                  ds.lsHCY_t10_v0_p60_m0_d1_l0 + \
                  ds.lsHCY_t10_v0_p330_m0_d1_l0 + \
                  ds.lsCYJ_t20_v0_p0_m0_d1_l0 + \
                  ds.lsCYJ_t20_v0_p30_m0_d1_l0 + \
                  ds.lsCYJ_t20_v0_p330_m0_d1_l0 + \
                  ds.lsXH_t1_v0_p0_m0_d1_l0 + \
                  ds.lsXH_t1_v0_p30_m0_d1_l0 + \
                  ds.lsXH_t1_v0_p330_m0_d1_l0 + \
                  ds.lsLQ_t1_v0_p0_m0_d1_l0 + \
                  ds.lsLQ_t1_v0_p30_m0_d1_l0 + \
                  ds.lsLQ_t1_v0_p330_m0_d1_l0
    
    lsPositionResult = []
    for i in xrange(5):
        dfResult = evaluatePosture(strDataDir, lsFileNames, 
                               lsDataAxisName, dcLabel, nFold=5)
        lsPositionResult.append(dfResult)

#%% muscle state      
    strDataDir = "../../data/experiment/muscle_state/"
    lsFileNames = ds.lsYL_t35_v0_p0_m0_d1_l0 + \
                  ds.lsYL_t35_v0_p0_m1_d1_l0 + \
                  ds.lsWW_t11_v0_p0_m0_d1_l0 + \
                  ds.lsWW_t11_v0_p0_m1_d1_l0 + \
                  ds.lsQY_t7_v0_p0_m0_d1_l0 + \
                  ds.lsQY_t7_v0_p0_m1_d1_l0 + \
                  ds.lsHCY_t10_v0_p0_m0_d1_l0 + \
                  ds.lsHCY_t10_v0_p0_m1_d1_l0 + \
                  ds.lsCHX_t5_v0_p0_m0_d1_l0 + \
                  ds.lsCHX_t5_v0_p0_m1_d1_l0 + \
                  ds.lsCYJ_t20_v0_p0_m0_d1_l0 + \
                  ds.lsCYJ_t20_v0_p0_m1_d1_l0 + \
                  ds.lsXH_t1_v0_p0_m0_d1_l0 + \
                  ds.lsXH_t1_v0_p0_m1_d1_l0 + \
                  ds.lsLQ_t1_v0_p0_m0_d1_l0 + \
                  ds.lsLQ_t1_v0_p0_m1_d1_l0
                  
                  
    lsMuscleResult = []
    for i in xrange(5):
        dfResult = evaluateMuscleState(strDataDir, lsFileNames, 
                                       lsDataAxisName,
                                       dcLabel, nFold=5)
        lsMuscleResult.append(dfResult)
#%% user mobility
    strDataDir = "../../data/experiment/user_mobility/"
    lsFileNames = ds.lsYL_t35_v0_p0_m0_d1_l0 + \
                  ds.lsYL_t35_v1_p0_m0_d1_l0 + \
                  ds.lsWW_t11_v0_p0_m0_d1_l0 + \
                  ds.lsWW_t11_v1_p0_m0_d1_l0 + \
                  ds.lsQY_t7_v0_p0_m0_d1_l0 + \
                  ds.lsQY_t7_v1_p0_m0_d1_l0 + \
                  ds.lsHCY_t10_v0_p0_m0_d1_l0 + \
                  ds.lsHCY_t10_v1_p0_m0_d1_l0 + \
                  ds.lsCHX_t5_v0_p0_m0_d1_l0 + \
                  ds.lsCHX_t5_v1_p0_m0_d1_l0 + \
                  ds.lsCYJ_t20_v0_p0_m0_d1_l0 + \
                  ds.lsCYJ_t20_v1_p0_m0_d1_l0 + \
                  ds.lsXH_t1_v0_p0_m0_d1_l0 + \
                  ds.lsXH_t1_v1_p0_m0_d1_l0 + \
                  ds.lsLQ_t1_v0_p0_m0_d1_l0 + \
                  ds.lsLQ_t1_v1_p0_m0_d1_l0
    
    lsMobilityResult = []
    for i in xrange(5):
        dfResult = evaluateUserMobility(strDataDir, lsFileNames, lsDataAxisName,
                         dcLabel, nFold=5)
        lsMobilityResult.append(dfResult)


#%% install location
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
                  ds.lsCHX_t5_v0_p0_m0_d4_l0 + \
                  ds.lsCYJ_t20_v0_p0_m0_d1_l0 + \
                  ds.lsCYJ_t20_v0_p0_m0_d2_l0 + \
                  ds.lsCYJ_t20_v0_p0_m0_d3_l0 + \
                  ds.lsCYJ_t20_v0_p0_m0_d4_l0 + \
                  ds.lsXH_t1_v0_p0_m0_d1_l0 + \
                  ds.lsXH_t1_v0_p0_m0_d2_l0 + \
                  ds.lsXH_t1_v0_p0_m0_d3_l0 + \
                  ds.lsXH_t1_v0_p0_m0_d4_l0 + \
                  ds.lsLQ_t1_v0_p0_m0_d1_l0 + \
                  ds.lsLQ_t1_v0_p0_m0_d2_l0 + \
                  ds.lsLQ_t1_v0_p0_m0_d3_l0 + \
                  ds.lsLQ_t1_v0_p0_m0_d4_l0 
                  
    lsLocationResult = []
    for i in xrange(5):
        dfResult = evaluateWearingLocation(strDataDir, lsFileNames, lsDataAxisName,
                                dcLabel, nFold=5)
        lsLocationResult.append(dfResult)
                      