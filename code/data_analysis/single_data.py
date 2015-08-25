# -*- coding: utf-8 -*-
"""
This script provides basic functions for analyzing a single data,
aslo, in its main body, it perform measurement on single data

@author: jason
"""

import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
import pandas as pd
import bp_filter
import math
import dataset as ds

lsRGB = ['r', 'g', 'b']
lsCMYK = ['c', 'm', 'y']

def computeEnvelope(arrData, nWindow, nMinPeriods=1):
    """compute upper and lower envelope for given data"""
    arrUpperEnvelope = pd.rolling_max(pd.Series(arrData), window=nWindow,
                                      min_periods=nMinPeriods)
    arrLowerEnvelope = pd.rolling_min(pd.Series(arrData), window=nWindow,
                                     min_periods=nMinPeriods)
                                     
    return arrUpperEnvelope, arrLowerEnvelope


def loadData(strWorkingDir, strFileName, lsColumnNames, strFileExt = '.txt'):
    """
        This function loads and clears a single acceleromter data
        
        Parameters
        ----------
        strWorkingDir: working directory
        strFileName: file name
        lsColumnNames: a list of column names
        strFileExt: file extention
        
        Returns
        ----------
        dfData_filtered cleared data frame
                    
    """
    # load data
    dfData = pd.read_csv(strWorkingDir+strFileName+strFileExt, dtype=np.float32)
    dfData.columns = lsColumnNames[:len(dfData.columns)]
        
    # clean data
    lsMask = [True, ]* len(dfData)
    for col in dfData.columns:
        lsMask = lsMask & (dfData[col] != -1) & (~dfData[col].isnull() )
    dfData_filtered = dfData[lsMask]
        
    return dfData_filtered

def computeModulus(dfXYZ):
    """
    Given a 3-column dataframe, computes the modulus for each row
    
    Parameters
    ----------
    dfXYS: data frame containing X, Y, Z data
    
    Returns
    ----------
    array of modulus
    """
    return np.sqrt(np.power(dfXYZ.iloc[:,0], 2.0) +
                   np.power(dfXYZ.iloc[:,1], 2.0) +
                   np.power(dfXYZ.iloc[:,2], 2.0) )
                   
def computeGravity(dfXYZ, nStart=0, nEnd=5000):
    """
        Given a 3-column acc data in some coordinates, computes the projection 
        of gravity on each axis via averaging over stable state.
        
        Parameters
        ----------
        dfXYZ: 3-column ACC data frame
        nStart: the start point of stable state
        nEnd: the end point of stable state
        
        Returns
        ----------
        projection of gravity on X, Y, Z axies, and gravity, respectively
    """
    dAvgGX = np.average(dfXYZ.iloc[nStart:nEnd, 0])
    dAvgGY = np.average(dfXYZ.iloc[nStart:nEnd, 1])
    dAvgGZ = np.average(dfXYZ.iloc[nStart:nEnd, 2])
    return dAvgGX, dAvgGY, dAvgGZ, math.sqrt(dAvgGX**2+dAvgGY**2+dAvgGZ**2)
    
def removeGravity(dfXYZ, nStart=0, nEnd=1000):
    """
        This function compute the gravity via stable states, 
        the remove it from data
        
        Parameteres
        -----------
        dfXYZ: 3-column ACC data frame
        nStart: start point of stable state
        nEnd: end point of stable state
        
        Returns
        ----------
        a 3-column ACC data frame without gravity
    """    
    # compute gravity
    dAvgGX, dAvgGY, dAvgGZ, dGravity = computeGravity(dfXYZ, nStart, nEnd)
    srGravity = pd.Series([dAvgGX, dAvgGY,dAvgGZ], index=dfXYZ.columns)
    dfXYZ_noG = dfXYZ - srGravity
    
    return dfXYZ_noG

    
    
#%%
if __name__ == '__main__':
    import sys
    sys.exit(0)
    
    #%%  setup
    dSamplingFreq = 160.0
    
    strWorkingDir = "../../data/feasibility_v7/"
    
    strFileName = ds.lsYL_t20[0]
    
    lsColumnNames = ['x0', 'y0','z0', 'gx0', 'gy0','gz0',
                     'x1', 'y1','z1', 'gx1', 'gy1','gz1']
                     
    nBasicFontSize = 16
    
    strBasicFontName = "Times new Roman"
    
    # load data
    dfData = loadData(strWorkingDir, strFileName, lsColumnNames)
    
    #%%  visualize time-domain
    lsAxis2Inspect = ['x0', 'y0', 'z0','x1', 'y1', 'z1']
    lsColors = lsRGB*6
    
    nPlotStartPoint = 0
    nPlotEndPoint = -1
    nRows= 3
    nCols = int(math.ceil(len(lsAxis2Inspect)/nRows))
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols, squeeze=False)
    for i, col in enumerate(lsAxis2Inspect):
        dfData[col].iloc[nPlotStartPoint:nPlotEndPoint].plot(color=lsColors[i],
                    ax=axes[i%nRows, int(math.ceil(i/nRows))], 
                    legend=False)
    
    fig.suptitle(strFileName + "@ time domain", 
                 fontname=strBasicFontName, 
                 fontsize=nBasicFontSize)
    plt.tight_layout()
    plt.show()
    
    #%% plot modulus
    lsAxis2Inspect = ['x0', 'y0', 'z0']
    lsColors = lsRGB*2
    
    nPlotStartPoint = 0
    nPlotEndPoint = -1
    nRows= 1
    nCols = 1
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols, squeeze=False)
    
    dfData_filtered = dfData[lsAxis2Inspect]
    for i in xrange(len(dfData_filtered.columns) ):
        if (i % 3 == 0):
            dfXYZ = dfData_filtered.iloc[:, i:i+3]
            strFormula = "sqrt(%s^2+%s^2+%s^2)" % (tuple(dfXYZ.columns) )

            dfXYZ_noG = removeGravity(dfXYZ, nStart=0, nEnd=dSamplingFreq*3)
            
            arrModulus = computeModulus(dfXYZ_noG)
            axes[i%nRows, 0].plot(arrModulus, color='b', lw=1, alpha=0.7)
            
            # plot envelope
            nWindow = 30
            arrUpperEnvelope = pd.rolling_max(pd.Series(arrModulus), window=nWindow,
                                              min_periods=1)
            arrLowerEnvelope = pd.rolling_min(pd.Series(arrModulus), window=nWindow,
                                             min_periods=1)
                                              
            axes[i%nRows, 0].plot(arrUpperEnvelope, color='r', lw=2, alpha=0.7)
            axes[i%nRows, 0].plot(arrLowerEnvelope, color='r', lw=2, alpha=0.7)
            
            axes[i%nRows, 0].set_xlabel(strFormula )
            
            
    fig.suptitle(strFileName + ": modulus", 
                 fontname=strBasicFontName, 
                 fontsize=nBasicFontSize)
    plt.tight_layout()
    plt.show()
    
    #%%  band pass filter
    nBPFilterStart = 0
    nBPFilterEnd = nBPFilterStart + (5*60*dSamplingFreq)
    
    nBandWidth = 10
    for nLowCut in xrange(0, int(dSamplingFreq/2.0), nBandWidth):
        nHighCut = nLowCut + nBandWidth
        fig, axes = plt.subplots(nrows=len(dfData.columns), ncols=1)
        for i, col in enumerate(dfData.columns):
            # bp filter
            arrFiltered= bp_filter.butter_bandpass_filter( \
            dfData[col].values[nBPFilterStart:nBPFilterEnd], \
            nLowCut, nHighCut, dSamplingFreq, order=9)
            
            #visualize
            axes[i].plot(dfData.index[nBPFilterStart:nBPFilterEnd], arrFiltered, lsColors[i])
    
        fig.suptitle("bpfilter: %d ~ %d Hz" % (nLowCut, nHighCut), \
                     fontname=strBasicFontName, 
                     fontsize=nBasicFontSize)
    plt.tight_layout()
    plt.show()
    
    #%% plot freq domain for modulus
    lsAxis2Inspect = ['x0', 'y0', 'z0', 'x1', 'y1', 'z1']
    lsColors = lsRGB*2
    
    nPlotStartPoint = 0
    nPlotEndPoint = -1
    nRows= 2
    nCols = 1
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols, squeeze=False)
    
    dfData_filtered = dfData[lsAxis2Inspect]
    for i in xrange(len(dfData_filtered.columns) ):
        if (i % 3 == 0):
            dfXYZ = dfData_filtered.iloc[:, i:i+3]
            strFormula = "sqrt(%s^2+%s^2+%s^2)" % (tuple(dfXYZ.columns) )

            dfXYZ_noG = removeGravity(dfXYZ, nStart=0, nEnd=dSamplingFreq*3)
            
            arrModulus = computeModulus(dfXYZ_noG)
            
            # bp filter
            arrFiltered = bp_filter.butter_bandpass_filter(arrModulus, 10, 75,
                                                           dSamplingFreq, order=9)
            
            # fft
            nSamples = len(arrFiltered)
            arrFreq = fftpack.fft(arrModulus)[10:nSamples/2]
            arrFreq_normalized = arrFreq/(nSamples*1.0)
            
            
            axes[i%nRows, 0].plot(np.abs(arrFreq_normalized) )
            axes[i%nRows, 0].set_xlabel(strFormula )
            
    fig.suptitle(strFileName + ": modulus", 
                 fontname=strBasicFontName, 
                 fontsize=nBasicFontSize)
    plt.tight_layout()
    plt.show()
   
#    #%%  visualize specgram
#
#    lsAxis2Inspect = ['x0', 'y0', 'z0', 'x1', 'y1', 'z1']
#    lsColors = lsRGB*2
#    
#    nPlotStartPoint = 0
#    nPlotEndPoint = -1
#    nRows= 3
#    nCols = int(math.ceil(len(lsAxis2Inspect)/nRows))
#    fig, axes = plt.subplots(nrows=nRows, ncols=nCols, squeeze=False)
#    for i, col in enumerate(lsAxis2Inspect):
#        Pxx, freqs, bins, im = axes[i%nRows, int(math.ceil(i/nRows))].specgram(
#                                dfData[col].iloc[nPlotStartPoint:nPlotEndPoint].values,
#                                NFFT=int(dSamplingFreq), 
#                                Fs=dSamplingFreq, 
#                                noverlap=int(dSamplingFreq/2.0) )
#        axes[i%nRows, int(math.ceil(i/nRows))].set_xlabel(col)
#    fig.suptitle(strFileName + "@ specgram", 
#                 fontname=strBasicFontName, 
#                 fontsize=nBasicFontSize)
#    plt.tight_layout()
#    plt.show()    
    
    # %% 
    #==============================================================================
    # FRF
    #==============================================================================
    lsRespCols = ['x0', 'y0', 'z0']
    lsExcCols = ['x1', 'y1', 'z1']
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
        
        ax = plt.figure().add_subplot(111)
        
        # get raw data
        arrResp_t = computeModulus(removeGravity(dfResp, nBaseLineStart, nBaseLineEnd) )
        arrExc_t = computeModulus(removeGravity(dfExc, nBaseLineStart, nBaseLineEnd) )
        
        plt.figure()
        plt.plot(arrResp_t)
        plt.figure()
        plt.plot(arrExc_t)
        
        nSamples = len(arrResp_t)
        nDCEnd = 10
        
        # bandpass
        nLowFreq = 10
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
        xf = np.linspace(nLowFreq, dSamplingFreq/2.0, nSamples/2.0-nDCEnd)
        ax.plot(xf, 
                pd.rolling_mean(abs(arrFRF[nDCEnd:nSamples/2]), 
                                window=10), 
                lsColors[i])

        # setup looks
        ax.set_xlabel("Freq", fontname=strBasicFontName, fontsize=nBasicFontSize)
        ax.set_ylabel("FRF", fontname=strBasicFontName, fontsize=nBasicFontSize)
        
        ax.set_xticks(np.arange(0, int(dSamplingFreq/2.0)+1, 5.0) )
        
#        axes[i].set_ylim(0.0, 100.0)
#        axes[i].set_xlim(0.0, 50.0)
        
#        axes[i].set_yscale('log');
        
        plt.setp(ax.get_xticklabels(),
                 fontname=strBasicFontName, 
                 fontsize=nBasicFontSize, 
                 rotation=90)
        plt.setp(ax.get_yticklabels(),
                 fontname=strBasicFontName, 
                 fontsize=nBasicFontSize)
        
        fig.suptitle(strFileName + ": FRF", 
                     fontname=strBasicFontName, 
                     fontsize=nBasicFontSize)
                     
        plt.tight_layout()
        nFFTStart = nFFTEnd
        
    plt.show()
    
    print("validation is over!")