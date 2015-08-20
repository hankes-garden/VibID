# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 21:47:45 2015

@author: jason
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.fftpack as fftpack
from matplotlib import colors, cm
import single_data as sd
import bp_filter
import dataset as ds

CN_MODULUS = "modulus"

def plotSpecgramEx(lsData, lsDataNames, strColumn2inspect, nMaxRows=3):
    nData2Plot = len(lsData)
    nSubplotRows = nMaxRows if nData2Plot>=nMaxRows else nData2Plot
    nSubplotCols = int(math.ceil(nData2Plot*1.0/nMaxRows))
    
    fig, axes = plt.subplots(nrows=nSubplotRows, ncols=nSubplotCols, squeeze=False)
    for i, (dfData, strDataName) in enumerate(zip(lsData, lsDataNames) ):
                            
        nRow2plot = i % nMaxRows
        nCol2Plot = i / nMaxRows
        
        Pxx, freqs, bins, im = axes[nRow2plot, nCol2Plot].specgram(dfAcc['y1'].values, 
                                       NFFT=int(dSamplingFreq), 
                                       Fs=dSamplingFreq,
                                       noverlap=int(dSamplingFreq/2.0))
        axes[nRow2plot, nCol2Plot].set_xlabel("Time", 
                                              fontname=strBasicFontName, 
                                              fontsize=18)
        axes[nRow2plot, nCol2Plot].set_ylabel("Frequency", 
                                              fontname=strBasicFontName, 
                                              fontsize=18)
    plt.tight_layout()
    plt.show()
    
    
    
    plt.tight_layout()

def computeModulusEx(lsData, lsXYZColumns, dSamplingFreq=160.0):
    """compute modulus for a list of data"""
    lsModulus = []
    for dfData in lsData:
        dfXYZ = dfData[lsXYZColumns]
        dfXYZ_noG = sd.removeGravity(dfXYZ, nEnd=int(dSamplingFreq * 3) )
        arrModulus = sd.computeModulus(dfXYZ_noG)
        dfModulus = pd.DataFrame(arrModulus, columns=[CN_MODULUS])
        lsModulus.append(dfModulus)
    return lsModulus

def getColorList(nColors, strColorMap = 'gist_rainbow'):
    colorMap = plt.get_cmap(strColorMap)
    cNorm  = colors.Normalize(vmin= 0, vmax=nColors- 1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=colorMap)
    
    lsColors=[scalarMap.to_rgba(i) for i in xrange(nColors)]
    return lsColors

def plotByDataAxis(lsData, lsDataNames, lsAxis2Plot,
                   nStartPoint = 0, nEndPoint = -1,
                   arrXTicks=None, strFontName='Times new Roman',
                   nFontSize=14, nMaxRows = 3,
                   lsColors=['r', 'g', 'b', 'c', 'y', 'm']):
    """
        This function plots the same axis of each data in the list on a same figure
    """
    nData2Plot = len(lsData)
    nSubplotRows = nMaxRows if nData2Plot>=nMaxRows else nData2Plot
    nSubplotCols = int(math.ceil(nData2Plot*1.0/nMaxRows))
    
    lsColors = lsColors * int(math.ceil(len(lsData)*1.0/len(lsColors)) ) \
                    if len(lsColors) < len(lsData) else lsColors
    
    i = 0
    for strCol in lsAxis2Plot:
        fig, axes = plt.subplots(nrows=nSubplotRows, ncols=nSubplotCols, squeeze=False)
        for nDataIndex in xrange(len(lsData) ):
            dfAcc = lsData[nDataIndex]
            srAxis = dfAcc[strCol].iloc[nStartPoint:nEndPoint]
            nRow2plot = nDataIndex % nMaxRows        
            nCol2Plot = nDataIndex / nMaxRows
            axes[nRow2plot, nCol2Plot].plot(srAxis.index, 
                                            srAxis, color=lsColors[i])
            axes[nRow2plot, nCol2Plot].set_xlabel(lsDataNames[nDataIndex],
                                                  fontname=strFontName, 
                                                  fontsize=nFontSize+2)
            plt.setp(axes[nRow2plot, nCol2Plot].get_xticklabels(), \
                     fontname=strFontName, fontsize=nFontSize, rotation=0)
            plt.setp(axes[nRow2plot, nCol2Plot].get_yticklabels(), \
                     fontname=strFontName, fontsize=nFontSize, rotation=0)
            i=i+1
            plt.tight_layout()
    plt.show()
    
def plotModolus(lsData, lsDataNames, lsXYZ,
                arrXTicks=None, strFontName='Times new Roman',
                nFontSize=14, nMaxRows = 3,
                lsColors=None):
    """plot modulus for a list of data"""
    nData2Plot = len(lsData)
    nSubplotRows = nMaxRows if nData2Plot>=nMaxRows else nData2Plot
    nSubplotCols = int(math.ceil(nData2Plot*1.0/nMaxRows))
    lsColors = ['b', ] * len(lsData) if lsColors is None else lsColors
    
    fig, axes = plt.subplots(nrows=nSubplotRows, ncols=nSubplotCols, squeeze=False)
    for i, (dfData, strDataName) in enumerate(zip(lsData, lsDataNames) ):
        dfXYZ = dfData[lsXYZ]
        dfXYZ_noG = sd.removeGravity(dfXYZ, nStart=0, nEnd=dSamplingFreq*3)
        arrModulus = np.sqrt(np.power(dfXYZ_noG.iloc[:,0], 2.0) + 
                             np.power(dfXYZ_noG.iloc[:, 1], 2.0) + 
                             np.power(dfXYZ_noG.iloc[:, 2], 2.0) )
                             
        nRow2plot = i % nMaxRows
        nCol2Plot = i / nMaxRows
        axes[nRow2plot, nCol2Plot].plot(arrModulus, color=lsColors[i])
        axes[nRow2plot, nCol2Plot].set_xlabel(strDataName)
        axes[nRow2plot, nCol2Plot].grid()
    plt.tight_layout()
    plt.show()
        
    
def fftByAxis(lsTimeData, dSamplingFreq, nDCEnd=5):
    """
        This function performs fft on each axis on a list of data.
        
        Parameters
        ----------
        lsTimeData: a list of time series
        dSamplingFreq: sampling frequency
        nDCEnd: the number of points which are regarded as DC
        
        Returns
        ----------
        lsFFTResult: a list of data in frequency domain
        
    """
    lsFFTResult = []
    for dfTimeData in lsTimeData:
        nSamples = len(dfTimeData)
        dResolution = dSamplingFreq*1.0/nSamples
        arrFreqIndex = np.linspace(nDCEnd*dResolution, dSamplingFreq/2.0, nSamples/2-nDCEnd)
        dfFreqData = pd.DataFrame(index = arrFreqIndex)
        for col in dfTimeData.columns:
            srAxis_time = dfTimeData[col]
            # fft
            arrFreq = fftpack.fft(srAxis_time)[nDCEnd:nSamples/2]
            arrAxis_freq = np.abs(arrFreq)/(nSamples*1.0)
            
            dfFreqData[col+"_freq"] = arrAxis_freq
        lsFFTResult.append(dfFreqData)
    return lsFFTResult
    
def fftOnModulus(lsData, lsXYZColumns, dSamplingFreq, nDCEnd=5):
    """compute fft on modulus of data"""
    lsFFTResult = []
    for dfData in lsData:
        dfXYZ = dfData[lsXYZColumns]
        nSamples = len(dfXYZ)
        dResolution = dSamplingFreq*1.0/nSamples
        arrFreqIndex = np.linspace(nDCEnd*dResolution, dSamplingFreq/2.0, nSamples/2-nDCEnd)
        
        #remove gravity
        dfXYZ_noG = sd.removeGravity(dfXYZ,nStart=0, nEnd=dSamplingFreq*2)
        # compute modulus
        arrModulus = np.sqrt(np.power(dfXYZ_noG.iloc[:,0], 2.0) + 
                             np.power(dfXYZ_noG.iloc[:, 1], 2.0) + 
                             np.power(dfXYZ_noG.iloc[:, 2], 2.0) )
                             
        # fft
        arrFreq = fftpack.fft(arrModulus)[nDCEnd:nSamples/2]
        arrFreq_normalized = arrFreq/(nSamples*1.0)
        dfFreq = pd.DataFrame(arrFreq_normalized, index=arrFreqIndex, columns=['modulus'])
        lsFFTResult.append(dfFreq)
        
    return lsFFTResult
            
    
    
def loadDataEx(strWorkingDir, lsFileNames, lsColumnNames, strFileExt = '.txt'):
    """"load a list of data"""
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
    
def frfEx(lsData, lsDataName, dSamplingFreq,
          nMaxRows=3, 
          lsRespCols = ['x0', 'y0', 'z0'], 
          lsExcCols = ['x1', 'y1', 'z1'], 
          nDCEnd = 200, 
          nLowFreq = 10):
    """
        Compute and plot FRF for each data in the list
        
        Parameters
        ----------
        lsData: list of response and excitation data
        lsDataName: list of data name
        nMaxRows: max number of row in figure
        lsRespCols: columns of response data
        lsExcCols: columns of excitation data
        nDCEnd: the number of point to skip at begining
        nLowFreq: lower frequency to filter
        
        Returns
        --------
        Null
    """
    nData2Plot = len(lsData)
    nSubplotRows = nMaxRows if nData2Plot>=nMaxRows else nData2Plot
    nSubplotCols = int(math.ceil(nData2Plot*1.0/nMaxRows))
    fig, axes = plt.subplots(nrows=nSubplotRows, ncols=nSubplotCols, squeeze=False)
    
    for i, (strDataName, dfData) in enumerate(zip(lsFileNames, lsData) ):
        
        dfResp = dfData[lsRespCols]
        dfExc = dfData[lsExcCols]
        
        nData2FFT = len(dfData)
        nFFTStart = 0
        nFFTEnd = 0
        nBinSize = int(dSamplingFreq)*9999
        
        nBaseLineStart = 0
        nBaseLineEnd = dSamplingFreq * 3
        while (nFFTStart < nData2FFT ):
            nFFTEnd = min(nData2FFT, nFFTStart+nBinSize)
            
            # get raw data
            arrResp_t = sd.computeModulus( sd.removeGravity(dfResp,
                                                            nBaseLineStart,
                                                            nBaseLineEnd) )
                                                      
            arrExc_t = sd.computeModulus( sd.removeGravity(dfExc,
                                                           nBaseLineStart,
                                                           nBaseLineEnd) )
            
            nSamples = len(arrResp_t)
            
            # bandpass
            nHighFreq = int(dSamplingFreq/2.0)-1
            arrFilteredResp_t = bp_filter.butter_bandpass_filter(arrResp_t, \
                                nLowFreq, nHighFreq, dSamplingFreq, order=9)
            arrFilteredExc_t = bp_filter.butter_bandpass_filter(arrExc_t, \
                                nLowFreq, nHighFreq, dSamplingFreq, order=9)
            
            # fft
            arrResp_f = fftpack.fft(arrFilteredResp_t)
            arrNormalizedResp_f = arrResp_f/(nSamples*1.0)
            
            arrExc_f = fftpack.fft(arrFilteredExc_t)
            arrNormalizedExc_f = arrExc_f/(nSamples*1.0)
            
            # FRF
            arrFRF = arrNormalizedResp_f/arrNormalizedExc_f;
            
            # plot
            nRow2plot = i % nMaxRows        
            nCol2Plot = i / nMaxRows
            
            xf = np.linspace(nLowFreq, dSamplingFreq/2.0, nSamples/2.0-nDCEnd)
            axes[nRow2plot, nCol2Plot].plot(xf,
                                            pd.rolling_mean(
                                                abs(arrFRF[nDCEnd:nSamples/2]),
                                            window=10), 
                    lsColors[i])
    
            # setup looks
            axes[nRow2plot, nCol2Plot].set_xlabel(strDataName, 
                                                  fontname=strBasicFontName,
                                                  fontsize=nBasicFontSize)
            
            axes[nRow2plot, nCol2Plot].set_xticks(np.arange(0, 
                                                  int(dSamplingFreq/2.0)+1, 5.0) )
            
            axes[nRow2plot, nCol2Plot].set_ylim(0.0, 2.5)
            
            plt.setp(axes[nRow2plot, nCol2Plot].get_xticklabels(),
                     fontname=strBasicFontName, 
                     fontsize=nBasicFontSize, 
                     rotation=90)
            plt.setp(axes[nRow2plot, nCol2Plot].get_yticklabels(),
                     fontname=strBasicFontName, 
                     fontsize=nBasicFontSize)
            
            fig.suptitle("FRF", 
                         fontname=strBasicFontName, 
                         fontsize=nBasicFontSize)
                         
            plt.tight_layout()
            nFFTStart = nFFTEnd
        
    plt.show()
    
    


#%%
if __name__ == '__main__':
    import sys
    sys.exit(0)
    
    #%% data sets & setup
    lsColumnNames = ['x0', 'y0','z0',
                     'gx0', 'gy0','gz0',
                     'x1', 'y1','z1',
                     'gx1', 'gy1','gz1']
    
    lsBasicColors = ['r', 'g', 'b', 'c', 'm', 'y']
    lsRGB = ['r', 'g', 'b']
    
    strBasicFontName = "Times new Roman"
    
    nBasicFontSize = 16
    
    #%% data to load
    dSamplingFreq = 160.0
    strWorkingDir = ("D:\\yanglin\\baidu_cloud\\research\\my_research\\resonance_lab\\"
                     "data\\feasibility_v7\\")
    
    lsFileNames = ds.lsCYJ_t11 + ds.lsCYJ_t12 + ds.lsCYJ_t13
    
    #%% statistics of time domain
    lsData = loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)
    for i, strColName in enumerate(lsColumnNames[:3]):
        dcStats = {}
        for strFileName, dfAcc in zip(lsFileNames, lsData):
            srAxisData = dfAcc[strColName]
            dc = srAxisData.describe().to_dict()
            nRange = dc['max'] - dc['min']
            dc['range'] = nRange
            dcStats[strFileName] = dc
        dfStat = pd.DataFrame(dcStats)
        ax0 = plt.figure().add_subplot(111)
        dfStat.ix['range'].plot(ax=ax0, kind='bar', color=lsBasicColors[i])
        plt.show()
    
    #%% time domain
    lsData = loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)
    nAxesPerFig = len(lsFileNames)
    lsColors = [c for c in lsBasicColors for _ in xrange(nAxesPerFig)]
    lsColumn2Plot = ['x0', 'y0', 'z0']
    plotByDataAxis(lsData, lsFileNames, lsColumn2Plot,
                   nStartPoint=0, nEndPoint=-1, 
                   nMaxRows=3, lsColors=lsColors)
                   
    #%% modulus
    lsData = loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)
    
    lsColors = ['b']*len(lsData)
    lsColumn2Plot = ['x0', 'y0', 'z0']
    plotModolus(lsData, lsFileNames, lsColumn2Plot,
                nMaxRows=3, lsColors=lsColors)
                
    lsColumn2Plot = ['x1', 'y1', 'z1']
    plotModolus(lsData, lsFileNames, lsColumn2Plot,
                nMaxRows=3, lsColors=lsColors)
    #%% fft for modulus
    lsData = loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)
    
    lsColors = ['b']*len(lsData)
    lsXYZColumns = ['x0', 'y0', 'z0']
    lsFreqData = fftOnModulus(lsData, lsXYZColumns, dSamplingFreq, nDCEnd=50)
    
    lsMagnitudeData = []
    for dfFreqData in lsFreqData:
        dfMagnitude = dfFreqData.abs()
        lsMagnitudeData.append(dfMagnitude)

    plotByDataAxis(lsMagnitudeData, lsFileNames, 
                   lsAxis2Plot=['modulus', ], lsColors=lsColors)
        
    #%% visualize time + freq = specgram
    lsData = loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)
    plotSpecgramEx(lsData, lsFileNames, "x0")
    
    #%% FRF
    lsData = loadDataEx(strWorkingDir, lsFileNames, lsColumnNames)
    frfEx(lsData, lsFileNames, dSamplingFreq)

