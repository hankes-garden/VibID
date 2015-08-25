# -*- coding: utf-8 -*-
"""
Given a data set of user's acc data, this script extracts some features and classify user

@author: jason
"""
import fastdtw
import bp_filter
import multiple_data as md
import single_data as sd
import dataset as ds

import pandas as pd
from sklearn import manifold
import matplotlib.pyplot as plt
import gc
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D # unused,but need for 3D projection
import operator

import sklearn.preprocessing as prepro
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import FastICA

CN_MODULUS = "modulus"
CN_DISTANCE = "dist"
CN_PATH_LIST = "path_list"
CN_AVG_DISTANCE = "avg_dist"

# type of signal
MODULUS = "modulus"
MOVING_MEAN = "movingAvg"
UPPER_ENVELOPE = "upper_envelope"
LOWER_ENVELOPE = "lower_envelope"

# type of statistics
BAND = "band"
MEAN = "mean"
STD = "std"
ONE_QUARTER = "25p"
TWO_QUARTER = "50p"
THREE_QUARTER = "75p"
MIN = "min"
MAX = "max"

CN_LABEL = "label"

CN_MODEL_ACCURACY = "accuracy"
CN_MODEL_FEATURE_IMP = "feature_importance"

def extractSignalStatistics(arrData, strDataName):
    """extrac statistic info. from a time series"""
    dcRet = {}
    dfStat = pd.Series(arrData).describe()
    dcRet["%s_%s" % (strDataName, MEAN)] = dfStat['mean']
    dcRet["%s_%s" % (strDataName, STD)] = dfStat['std']
    dcRet["%s_%s" % (strDataName, MIN)] = dfStat['min']
    dcRet["%s_%s" % (strDataName, ONE_QUARTER)] = dfStat['25%']
    dcRet["%s_%s" % (strDataName, TWO_QUARTER)] = dfStat['50%']
    dcRet["%s_%s" % (strDataName, THREE_QUARTER)] = dfStat['75%']
    dcRet["%s_%s" % (strDataName, MAX)] = dfStat['max']
    
    return dcRet

def extractTemporalFeatures(dfModulus, strDataName, dSamplingFreq):
    """
        Extract temporal features from data
        
        return
        -------
        dcRecord: a dict of features and label for one data
    """
    dcRecord = {}
    
    # label
    dcRecord[CN_LABEL] = dcUserID[strDataName[:2] ]

    # feature from modulus
    dcRecord.update(extractSignalStatistics(dfModulus[CN_MODULUS].values, MODULUS) )
    
    arrUpperEnv, arrLowerEnv = sd.computeEnvelope(dfModulus[CN_MODULUS].values,
                                                  nWindow=dSamplingFreq/3)
    dcRecord.update(extractSignalStatistics(arrUpperEnv, 
                                            "%s_%s"%(MODULUS, UPPER_ENVELOPE) ) )
    dcRecord.update(extractSignalStatistics(arrLowerEnv, 
                                            "%s_%s"%(MODULUS, LOWER_ENVELOPE)) )
    
    arrMean = pd.rolling_mean(dfModulus[CN_MODULUS].values, window=dSamplingFreq)
    dcRecord.update(extractSignalStatistics(arrMean, "%s_%s"%(MODULUS, MOVING_MEAN)) )

    # statistics of each freq band
    nBandWidth = 10
    for nLow in xrange(5, int(dSamplingFreq/2.0)-5, 5):
        nHigh = min(nLow + nBandWidth, int(dSamplingFreq/2.0)-1)
        arrFiltered = bp_filter.butter_bandpass_filter(dfModulus[CN_MODULUS].values,
                                                       nLow, nHigh,
                                                       dSamplingFreq, order=9)
        
        dcRecord.update(extractSignalStatistics(arrFiltered, 
                                                "%s%d%d"%(BAND, nLow, nHigh) ) )
        # envelope
        arrUpperEnv, arrLowerEnv = sd.computeEnvelope(arrFiltered,
                                                      nWindow=30)
        dcRecord.update(extractSignalStatistics(arrUpperEnv, 
                        "%s%d%d_%s"%(BAND, nLow, nHigh, UPPER_ENVELOPE) ) )
        dcRecord.update(extractSignalStatistics(arrLowerEnv,
                        "%s%d%d_%s"%(BAND, nLow, nHigh, LOWER_ENVELOPE) ) )
                        
        arrMean = pd.rolling_mean(arrFiltered, window=dSamplingFreq)
        dcRecord.update(extractSignalStatistics(arrMean,
                        "%s%d%d_%s"%(BAND, nLow, nHigh, MOVING_MEAN) ) )
                        
    return dcRecord
    
    


def computeFastICA(arrModulus, dSamplingFreq, dMinFreq, dcICAParams):
    # construct embedding matrix
    l = int(math.ceil(dSamplingFreq/dMinFreq) )
    k = len(arrModulus) - l + 1
    matEmbedding = np.array(arrModulus[0:k])
    for i in xrange(1, len(arrModulus)-k+1):
        matEmbedding = np.vstack((matEmbedding, arrModulus[i:i+k]))

    # fastICA
    ica = FastICA() if dcICAParams is None else FastICA(**dcICAParams)
    return ica.fit_transform(matEmbedding)

def computeFastICAEx(lsModulus, dSamplingFreq, dMinFreq, dcICAParams):
    lsComponents = []
    for dfModulus in lsModulus:
        arrModulus = dfModulus[CN_MODULUS].values
        lsComponents.append(computeFastICA(arrModulus, dSamplingFreq,
                                           dMinFreq, dcICAParams) )

    return lsComponents


def pairwiseFastDTW(lsModulus, lsDataNames):
    """Given a list of data, compute pairwise distance via fastDTW"""
    nLen = len(lsModulus)

    dcDTWDetails = {}
    dcPairwiseDistance = {}
    for i in xrange(0, nLen):
        for j in xrange(0, nLen):

            dDistance = 0.0
            dAvgDistance = 0.0
            lsPath = []
            strDetailKey = "%d-%d" % (min(i, j), max(i, j))
            if(i == j):
                dDistance = 0.0
                dAvgDistance = 0.0
                lsPath = zip(xrange(nLen), xrange(nLen))

                # update detail result
                dcDTWDetails[strDetailKey] = {CN_DISTANCE: dDistance,
                                              CN_PATH_LIST: lsPath,
                                              CN_AVG_DISTANCE: dAvgDistance }

                # update pairwise distance
                strModulus1 = lsDataNames[i]
                strModulus2 = lsDataNames[j]
                dcDistanceBtw = dcPairwiseDistance.get(strModulus1, None)
                if(dcDistanceBtw is None):
                    dcDistanceBtw = {}
                    dcPairwiseDistance[strModulus1] = dcDistanceBtw
                dcDistanceBtw[strModulus2] = dAvgDistance
            else:
                dcDetail = dcDTWDetails.get(strDetailKey, None)
                if (dcDetail is None):
                    # compute fastDTW
                    arrModolus1 = lsModulus[i][CN_MODULUS].values
                    arrModolus2 = lsModulus[j][CN_MODULUS].values
                    print("computing fastDTW for (%d, %d)..." % (i, j))
                    dDistance, lsPath = fastdtw.fastdtw(arrModolus1, arrModolus2)
                    dAvgDistance = dDistance / (len(lsPath) * 1.0)

                    # update detail result
                    dcDTWDetails[strDetailKey] = {CN_DISTANCE: dDistance,
                                                  CN_PATH_LIST: lsPath,
                                                  CN_AVG_DISTANCE: dAvgDistance }

                    # update pairwise distance
                    strModulus1 = lsDataNames[i]
                    strModulus2 = lsDataNames[j]
                    dcDistanceBtw = dcPairwiseDistance.get(strModulus1, None)
                    if(dcDistanceBtw is None):
                        dcDistanceBtw = {}
                        dcPairwiseDistance[strModulus1] = dcDistanceBtw
                    dcDistanceBtw[strModulus2] = dAvgDistance
                    # update symetricaly
                    dcDistanceBtw = dcPairwiseDistance.get(strModulus2, None)
                    if(dcDistanceBtw is None):
                        dcDistanceBtw = {}
                        dcPairwiseDistance[strModulus2] = dcDistanceBtw
                    dcDistanceBtw[strModulus1] = dAvgDistance
        gc.collect()

    return dcDTWDetails, dcPairwiseDistance

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
        print dAccuracy
        print arrY_pred
        print arrY_test

        dcCurrentFold[CN_MODEL_ACCURACY] = dAccuracy
        dcFeatureImportance = { k:v for (k,v) in zip(lsFeatureColumns,
                                 model.feature_importances_.tolist()) }
        dcCurrentFold[CN_MODEL_FEATURE_IMP] = dcFeatureImportance

        dcResults[i] = dcCurrentFold
        i = i+1

    return dcResults


# data sets & setup
lsColumnNames = ['x0', 'y0','z0', 'gx0', 'gy0','gz0',
                 'x1', 'y1','z1', 'gx1', 'gy1','gz1']

lsBasicColors = ['r', 'g', 'b', 'c', 'm', 'y']
lsRGB = ['r', 'g', 'b']

lsMarkers = ['o', 'v', 'd', 's', '+', 'x',  '1', '2', '3', '4']

strBasicFontName = "Times new Roman"

nBasicFontSize = 16

dSamplingFreq = 160.0
strWorkingDir = "../../data/feasibility_v7/"

# %%
if __name__ == "__main__":
    import sys
    sys.exit(0)



#==============================================================================
#     Time domain
#==============================================================================
     #%% X, Y, Z axes
    lsFileNames = ds.lsYL_t19 + ds.lsYL_t20 + ds.lsYL_t21 + \
                  ds.lsCYJ_t11 + ds.lsCYJ_t11 + ds.lsCYJ_t11
    lsData = md.loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)
    nAxesPerFig = len(lsFileNames)
    lsColors = [c for c in lsBasicColors for _ in xrange(nAxesPerFig)]
    lsColumn2Plot = ['x0', 'y0', 'z0']
    md.plotByDataAxis(lsData, lsFileNames, lsColumn2Plot,
                   nStartPoint=0, nEndPoint=-1,
                   nMaxRows=3, lsColors=lsColors)

#%% compute modulus
    lsFileNames = ds.lsYL_t19[2:5] + ds.lsCYJ_t11[2:5] + ds.lsHCY_t1[2:5] + ds.lsWW_t3[2:5]
    lsData = md.loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)

    lsXYZColumns = ['x0', 'y0', 'z0']
    lsModulus = md.computeModulusEx(lsData, lsXYZColumns)

#%% plot modulus
    lsColumn2Plot = ['x0', 'y0', 'z0']
    md.plotModolusEx(lsData, lsFileNames, lsColumn2Plot, 
                   dSamplingFreq, bPlotShapeLine=True)

#%% bandpass on modulus
    lsColors = md.getColorList(20, "gist_earth")

    for strDataName, dfModulus in zip(lsFileNames, lsModulus)[:2]:
        nBandWidth = 10
        nMaxRows = 4
        lsLowCuts = range(5, int(dSamplingFreq/2.0)-5, 5)
        nSubplots = len(lsLowCuts)

        fig, axes = plt.subplots(nrows=nMaxRows if nSubplots>nMaxRows else nSubplots,
                                 ncols=int(math.ceil(nSubplots*1.0/nMaxRows) ) )
        for i, nLowCut in enumerate(lsLowCuts):
            nHighCut = min((nLowCut+nBandWidth), ( (dSamplingFreq/2.0)-1) )
            # bp filter
            arrFiltered= bp_filter.butter_bandpass_filter(dfModulus[CN_MODULUS].values,
                                                          nLowCut, nHighCut,
                                                          dSamplingFreq, order=9)

            #visualize
            nRow2plot = i % nMaxRows
            nCol2Plot = i / nMaxRows
            axes[nRow2plot, nCol2Plot].plot(arrFiltered, color = lsColors[i])
            arrUpperEnv, arrLowerEnv = sd.computeEnvelope(arrFiltered, nWindow=30)
            axes[nRow2plot, nCol2Plot].plot(arrUpperEnv, 'r-', lw=2, alpha=0.6)
            axes[nRow2plot, nCol2Plot].plot(arrLowerEnv, 'r-', lw=2, alpha=0.6)
            axes[nRow2plot, nCol2Plot].plot(pd.rolling_mean(arrFiltered, window=100),
                                            'r-', lw=2, alpha=0.6)
            axes[nRow2plot, nCol2Plot].set_xlabel( "%d ~ %d Hz" % (nLowCut, nHighCut) )

        fig.suptitle(strDataName, fontname=strBasicFontName,
                     fontsize=nBasicFontSize)
        fig.tight_layout()
    plt.show()


##%%  DTW
#    lsFileNames = ds.lsZLW_l1[:3]
#    lsData = md.loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)
#
#    lsXYZColumns = ['x0', 'y0', 'z0']
#    lsModulus = md.computeModulusEx(lsData, lsXYZColumns)
#    # compute pairwise DTW
#    dcDTWDetails, dcDistance = pairwiseFastDTW(lsModulus, lsFileNames)
#    dfDistanceMatrix = pd.DataFrame(dcDistance)
#
##%% MDS
#    mds = manifold.MDS(n_components=3, dissimilarity="precomputed",
#                       random_state=7)
#    results = mds.fit(dfDistanceMatrix.as_matrix() )
#
#    coords = results.embedding_
#
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    lsUserNames = ["ZLW", "YL"]
#    lsUserDataCount = [18, 15]
#    for i, nUserDataCount in enumerate(lsUserDataCount):
#        nUserIndexBegin = sum(lsUserDataCount[:i])
#        nUserIndexEnd = sum(lsUserDataCount[:i+1])
#        ax.scatter(coords[nUserIndexBegin:nUserIndexEnd, 0],
#                   coords[nUserIndexBegin:nUserIndexEnd, 1],
#                   coords[nUserIndexBegin:nUserIndexEnd, 2],
#                   s=150,
#                   c=lsBasicColors[i],
#                   marker=lsMarkers[i],
#                   label=lsUserNames[i])
#    plt.legend()
#    plt.tight_layout()
#    plt.show()

#%%  feature extraction
    lsFileNames = ds.lsYL_t19 + ds.lsYL_t20 + ds.lsYL_t21 + \
                  ds.lsCYJ_t11 + ds.lsCYJ_t11 + ds.lsCYJ_t11 + \
                  ds.lsHCY_t1 + ds.lsHCY_t2 + ds.lsHCY_t3 + \
                  ds.lsWW_t3

    dcUserID = {"yl":1, "cy":2, "hc":3, "ww":4}
    lsData = md.loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)

    lsXYZColumns = ['x0', 'y0', 'z0']
    lsModulus = md.computeModulusEx(lsData, lsXYZColumns)

    lsFeatureLabel = []
    for strDataName, dfModulus in zip(lsFileNames, lsModulus):
        dcRecord = extractTemporalFeatures(dfModulus, strDataName, dSamplingFreq)
        lsFeatureLabel.append(dcRecord)

    dfFeatureLabel = pd.DataFrame(lsFeatureLabel)

#%% classification
    # prepare train & testing set
    lsFeatureColumns = [ col for col in dfFeatureLabel.columns if col != CN_LABEL]
    strLabelColumn = CN_LABEL
    mtX = dfFeatureLabel[lsFeatureColumns].as_matrix()
    arrY = dfFeatureLabel[strLabelColumn].values

    # model setup
    strModelName = 'GBRT'
    modelParams = {'n_estimators':100, 'max_features':0.5}

#    strModelName = 'random_forest'
#    modelParams = {'n_estimators':50}

#    strModelName = 'decision_tree'
#    modelParams = {"max_features": 0.8, "criterion": "entropy"}

#    strModelName = 'SVM'
#    modelParams = None

    dcResults = classify(mtX, arrY, strModelName, modelParams, lsFeatureColumns, nFold=5)

    # feature importance
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

    # overall performance
    lsAccuracy = [ i[CN_MODEL_ACCURACY] for i in dcResults.values()]
    dBestAccuracy = np.max(lsAccuracy)
    dWorstAccuracy = np.min(lsAccuracy)
    dMeanAccuracy = np.mean(lsAccuracy)
    dAccuracyStd = np.std(lsAccuracy)
    print("overall performance: \n  best=%.2f, worst=%.2f, "
          "mean=%.2f, std=%.2f" % \
           (dBestAccuracy, dWorstAccuracy,
            dMeanAccuracy, dAccuracyStd) )
