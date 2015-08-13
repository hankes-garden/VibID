# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 22:21:11 2015

@author: jason
"""
import fastdtw
import bp_filter
import single_data as sd
import multiple_data as md

import pandas as pd
from sklearn import manifold
import matplotlib.pyplot as plt
import gc
import numpy as np
import math

CN_MODULUS = "modulus"
CN_DISTANCE = "dist"
CN_PATH_LIST = "path_list"
CN_AVG_DISTANCE = "avg_dist"

def pairwiseFastDTW(lsModulus, lsDataNames):
    """Given a list of data, compute pairwise distance"""
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


# %% 
if __name__ == "__main__":
   #%% data sets & setup
    lsColumnNames = ['x0', 'y0','z0',
                     'gx0', 'gy0','gz0',
                     'x1', 'y1','z1',
                     'gx1', 'gy1','gz1']
    
    lsBasicColors = ['r', 'g', 'b', 'c', 'm', 'y']
    lsRGB = ['r', 'g', 'b']

    lsMarkers = ['o', 'v', 'd', 's', '+', 'x',  '1', '2', '3', '4']    
    
    strBasicFontName = "Times new Roman"
    
    nBasicFontSize = 16
    
    # CYJ
    lsCYJ_t6_l1 = ['cyj_t6_l1_35', 'cyj_t6_l1_35_1', 'cyj_t6_l1_45']
    
    lsCYJ_t7_l1 = ['cyj_t7_l1_35', 'cyj_t7_l1_40', 'cyj_t7_l1_40_1']
    
    lsCYJ_t8_l1 = ['cyj_t8_l1_30', 'cyj_t8_l1_40', 'cyj_t8_l1_45']
    
    lsCYJ_t9_l1 = ['cyj_t9_l1_40', 'cyj_t9_l1_40_1', 'cyj_t9_l1_40_2']
    
    lsCYJ_t10_l1 = ['cyj_t10_l1_40', 'cyj_t10_l1_40_1', 'cyj_t10_l1_40_2']
    
    # YL
    lsYL_t14_l1 = ["yl_t14_l1_35", "yl_t14_l1_35_1", "yl_t14_l1_40"]
    
    lsYL_t15_l1 = ["yl_t15_l1_35", "yl_t15_l1_40", "yl_t15_l1_50"]
    
    lsYL_t16_l1 = ["yl_t16_l1_35", "yl_t16_l1_40", "yl_t16_l1_40_1"]
    
    lsYL_t17_l1 = ["yl_t17_l1_35", "yl_t17_l1_35_1", "yl_t17_l1_35_2"]
    
    lsYL_t18_l1 = ["yl_t18_l1_45", "yl_t18_l1_35", "yl_t18_l1_35_1"]
                
    # LZY
    lsLZY_t1_l1 = ["LZY_t1_l1_30", "LZY_t1_l1_30_1", "LZY_t1_l1_35"]
    lsLZY_t1_l10 = ["LZY_t1_l10_30", "LZY_t1_l10_35", "LZY_t1_l10_35_1"]
    
    lsLZY_t2_l1 = ["LZY_t2_l1_45", "LZY_t2_l1_40", "LZY_t2_l1_40_1"]
    lsLZY_t2_l10 = ["LZY_t2_l10_30", "LZY_t2_l10_35", "LZY_t2_l10_50"]
    
    lsLZY_t3_l1 = ["LZY_t3_l1_35", "LZY_t3_l1_40", "LZY_t3_l1_40_1"]
    lsLZY_t3_l10 = ["LZY_t3_l10_35", "LZY_t3_l10_45", "LZY_t3_l10_50"]
    
    # ZLW
    lsZLW_t1_l1 = ["ZLW_t1_l1_40", "ZLW_t1_l1_40_1", "ZLW_t1_l1_45"]
    lsZLW_t1_l10 = ["ZLW_t1_l10_35", "ZLW_t1_l10_50", "ZLW_t1_l10_60"]
    
    lsZLW_t2_l1 = ["ZLW_t2_l1_35", "ZLW_t2_l1_40", "ZLW_t2_l1_45"]
    lsZLW_t2_l10 = ["ZLW_t2_l10_40", "ZLW_t2_l10_45", "ZLW_t2_l10_55"]
    
    lsZLW_t3_l1 = ["ZLW_t3_l1_35", "ZLW_t3_l1_35_1", "ZLW_t3_l1_45"]
    lsZLW_t3_l10 = ["ZLW_t3_l10_60", "ZLW_t3_l10_60_1", "ZLW_t3_l10_65"]
                  
    # %% load data
    dSamplingFreq = 160.0
    strWorkingDir = ("D:\\yanglin\\baidu_cloud\\research\\my_research\\resonance_lab\\"
                     "data\\feasibility_v6\\")
    
    lsFileNames = lsZLW_t1_l1 + lsZLW_t2_l1 + lsZLW_t3_l1 + \
                  lsLZY_t1_l1 + lsLZY_t2_l1 + lsLZY_t3_l1 + \
                  lsCYJ_t6_l1 + lsCYJ_t7_l1 + lsCYJ_t8_l1 + lsCYJ_t9_l1 + lsCYJ_t10_l1 + \
                  lsYL_t14_l1 + lsYL_t15_l1 + lsYL_t16_l1 + lsYL_t17_l1 + lsYL_t18_l1
    
    
#==============================================================================
#     Time domain
#==============================================================================
     #%% X, Y, Z axes
    lsData = md.loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)
    nAxesPerFig = len(lsFileNames)
    lsColors = [c for c in lsBasicColors for _ in xrange(nAxesPerFig)]
    lsColumn2Plot = ['x0', 'y0', 'z0', 'x1', 'y1', 'z1']
    md.plotByDataAxis(lsData, lsFileNames, lsColumn2Plot,
                   nStartPoint=0, nEndPoint=-1, 
                   nMaxRows=3, lsColors=lsColors)
                   
    #%% modulus
    lsData = md.loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)
    
    lsModulus = []
    lsXYZColumns = ['x0', 'y0', 'z0']
    for dfData in lsData:
        dfXYZ = dfData[lsXYZColumns]
        dfXYZ_noG = sd.removeGravity(dfXYZ, nEnd=int(dSamplingFreq * 3) )
        arrModulus = sd.computeModulus(dfXYZ_noG)
        dfModulus = pd.DataFrame(arrModulus, columns=[CN_MODULUS])
        lsModulus.append(dfModulus)
        
    md.plotByDataAxis(lsModulus, lsFileNames, ["modulus"],
                      lsColors=['b']*len(lsModulus) )
    
    #%% bandpass on modulus
    for strDataName, dfModulus in zip(lsFileNames, lsModulus)[:3]:
        nBandWidth = 10
        nMaxRows = 5
        nSubplots = dSamplingFreq/2.0/nBandWidth
        
        fig, axes = plt.subplots(nrows=nMaxRows if nSubplots>nMaxRows else nSubplots, 
                                 ncols=int(math.ceil(nSubplots/nMaxRows) ) )
        for i, nLowCut in enumerate(np.arange(2.0, (dSamplingFreq/2.0)-1, nBandWidth) ):
            nHighCut = min((nLowCut+nBandWidth), ( (dSamplingFreq/2.0)-1) )
            # bp filter
            arrFiltered= bp_filter.butter_bandpass_filter(dfModulus[CN_MODULUS].values,
                                                          nLowCut, nHighCut, 
                                                          dSamplingFreq, order=9)
            
            #visualize
            nRow2plot = i % nMaxRows        
            nCol2Plot = i / nMaxRows
            axes[nRow2plot, nCol2Plot].plot(arrFiltered, lsColors[i])
            axes[nRow2plot, nCol2Plot].set_xlabel( "%d ~ %d Hz" % (nLowCut, nHighCut) )
        
        fig.suptitle(strDataName, fontname=strBasicFontName,
                     fontsize=nBasicFontSize)
        fig.tight_layout()
    plt.show()
    
#==============================================================================
#     Classification
#==============================================================================
    # %% compute pairwise DTW
    dcDTWDetails, dcDistance = pairwiseFastDTW(lsModulus, lsFileNames)
    dfDistanceMatrix = pd.DataFrame(dcDistance)
    
    # %% MDS
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed")
    results = mds.fit(dfDistanceMatrix.as_matrix() )
    
    coords = results.embedding_

    ax = plt.figure().add_subplot(111)
    lsUserNames = ["ZLW", "LZY", "CYJ", "YL"]
    lsUserDataCount = [9, 9, 15, 15]
    for i, nUserDataCount in enumerate(lsUserDataCount):
        nUserIndexBegin = sum(lsUserDataCount[:i])
        nUserIndexEnd = sum(lsUserDataCount[:i+1])
        ax.scatter(coords[nUserIndexBegin:nUserIndexEnd, 0], 
                   coords[nUserIndexBegin:nUserIndexEnd, 1],
                   s=100,
                   c=lsBasicColors[i], 
                   marker=lsMarkers[i],
                   label=lsUserNames[i])
    plt.legend()
    plt.tight_layout()
    plt.show()
