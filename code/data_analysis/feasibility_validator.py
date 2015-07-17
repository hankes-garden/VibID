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

def plotByDataAxis(lsData, lsDataNames, nStartAxis, nEndAxis, strFontName='Times new Roman', \
           nFontSize=14, nMaxRows = 3, lsColors=['r', 'g', 'b', 'c', 'y', 'm']):
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
            axes[nRow2plot, nCol2Plot].set_xlabel(lsDataNames[nDataIndex], \
                                        fontname=strFontName, fontsize=nFontSize+2)
            plt.setp(axes[nRow2plot, nCol2Plot].get_xticklabels(), \
                     fontname=strFontName, fontsize=nFontSize, rotation=45)
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
    
#%%
if __name__ == '__main__':
    #%% setup
    dSamplingFreq = 380.0
    
    strWorkingDir = "D:\\yanglin\\baidu_cloud\\research\\my_research\\resonance_lab\\data\\feasibility_v2\\"
    
    lsColumnNames = ['x0', 'y0','z0','x1', 'y1','z1']
    
    lsBasicColors = ['r', 'g', 'b', 'c', 'y', 'm']
    
    lsQY_t1 = ['qy_t1_60', 'qy_t1_45', 'qy_t1_45_1', 'qy_t1_45_2', 'qy_t1_60_1']
    
    lsYL_t1 = ['yl_t1_50','yl_t1_55','yl_t1_45','yl_t1_50_1','yl_t1_45_1']
    lsYL_t2 = ['yl_t2_50','yl_t2_70','yl_t2_65','yl_t2_60','yl_t2_55']
    lsYL_t3 = ['yl_t3_45','yl_t3_60','yl_t3_50','yl_t3_50_1','yl_t3_55']
    
    #%% compare axis
    lsFileNames = lsYL_t1+lsYL_t2+lsYL_t3
    lsData = loadData(strWorkingDir, lsFileNames, lsColumnNames)
    nAxesPerFig = len(lsFileNames)
    lsColors = [c for c in lsBasicColors for _ in xrange(nAxesPerFig)]
    plotByDataAxis(lsData, lsFileNames, nStartAxis=0, nEndAxis=3, \
        nMaxRows=5, lsColors=lsColors)

