# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 21:47:45 2015

@author: jason
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from matplotlib import colors, cm

def getColorList(nColors, strColorMap = 'gist_rainbow'):
    colorMap = plt.get_cmap(strColorMap)
    cNorm  = colors.Normalize(vmin= 0, vmax=nColors- 1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colorMap)
    
    lsColors=[scalarMap.to_rgba(i) for i in xrange(nColors)]
    return lsColors

def plotEx(lsData, nStartAxis, nEndAxis, strFontName, \
           nFontSize, nMaxRows = 3, lsColors=['r', 'g', 'b', 'c', 'y', 'm']):
    nData2Plot = len(lsData)
    nSubplotRows = nMaxRows if nData2Plot>=nMaxRows else nData2Plot
    nSubplotCols = int(math.ceil(nData2Plot*1.0/nMaxRows))
    
    i = 0
    for nAxisIndex in xrange(nStartAxis, nEndAxis, 1):
        fig, axes = plt.subplots(nrows=nSubplotRows, ncols=nSubplotCols, squeeze=False)
        for nDataIndex in xrange(len(lsData) ):
            dfAcc = lsData[nDataIndex]
            srAxis = dfAcc.iloc[:,nAxisIndex]
            nRow2plot = nDataIndex % nMaxRows        
            nCol2Plot = nDataIndex / nMaxRows
            axes[nRow2plot, nCol2Plot].plot(srAxis, \
                                    color=lsColors[i])
            axes[nRow2plot, nCol2Plot].set_xlabel(lsFileNames[nDataIndex], \
                                        fontname=strFontName, fontsize=nFontSize+2)
            plt.setp(axes[nRow2plot, nCol2Plot].get_xticklabels(), \
                     fontname=strFontName, fontsize=nFontSize, rotation=90)
            plt.setp(axes[nRow2plot, nCol2Plot].get_yticklabels(), \
                     fontname=strFontName, fontsize=nFontSize, rotation=0)
            i=i+1
    plt.tight_layout()
    plt.show()
    
def loadData(strWorkingDir, lsFileNames, lsColumnNames, strFileExt = '.txt'):
    # load data
    lsData = []
    for strName in lsFileNames:
        dfAcc = pd.read_csv(strWorkingDir+strName+strFileExt, dtype=np.float32)
        dfAcc.columns = lsColumnNames[:len(dfAcc.columns)]
        lsData.append(dfAcc)
        
    # clean data
    for i in xrange(len(lsData) ):
        lsMask = [True ]* len(lsData[i])
        for col in lsData[i].columns:
            lsMask = lsMask & (lsData[i][col] != -1) & (~lsData[i][col].isnull() )
        lsData[i] = lsData[i][lsMask]
        
    return lsData
    

if __name__ == '__main__':
    #%% setup
    dSamplingFreq = 190.0
    
    strWorkingDir = "D:\\yanglin\\baidu_cloud\\research\\my_research\\resonance_lab\\data\\"
    
    lsColumnNames = ['x0', 'y0','z0','x1', 'y1','z1']
    
        
    #%% ex1. motor
    lsFileNames = ['yl_3_50','yl_3_60','yl_3_65','yl_4_50','yl_4_75','yl_4_80']
    lsData = loadData(strWorkingDir, lsFileNames, lsColumnNames)
    plotEx(lsData[:3], nStartAxis=3, nEndAxis=6, strFontName='Times new Roman', nFontSize=14)
    
    #%% ex2. single user
    lsFileNames = ['yl_3_50','yl_3_60','yl_3_65']
    lsData = loadData(strWorkingDir, lsFileNames, lsColumnNames)
    plotEx(lsData, nStartAxis=0, nEndAxis=3, strFontName='Times new Roman', nFontSize=14)
    
    #%% ex3. different users
    lsFileNames = ['fan_3_45','ww_3_45','yl_3_50']
    lsData = loadData(strWorkingDir, lsFileNames, lsColumnNames)
    plotEx(lsData, nStartAxis=0, nEndAxis=3, strFontName='Times new Roman', nFontSize=14)
    
    #%% ex4. different measurement points
    lsFileNames = ['location_v2\\yl_4_45','location_v2\\yl_4_60','location_v2\\yl_4_85',
    'experiment_on_locations\\yl_4_65_v2','experiment_on_locations\\yl_4_50_v2','experiment_on_locations\\yl_4_55_v2']
    lsData = loadData(strWorkingDir, lsFileNames, lsColumnNames)
    lsColors = ['r']*6 + ['g']*6 + ['b']*6
    plotEx(lsData, nStartAxis=0, nEndAxis=3, strFontName='Times new Roman', nFontSize=14, nMaxRows=3, lsColors=lsColors)

