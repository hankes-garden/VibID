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

lsRGB = ['r', 'g', 'b']
lsCMYK = ['c', 'm', 'y']

def computeEnvelope(arrData, nWindow, nMinPeriods=None):
    """compute upper and lower envelope for given data"""
    arrUpperEnvelope = pd.rolling_max(pd.Series(arrData), window=nWindow,
                                      min_periods=nMinPeriods, center=True)
    arrLowerEnvelope = pd.rolling_min(pd.Series(arrData), window=nWindow,
                                     min_periods=nMinPeriods, center=True)
    return arrUpperEnvelope, arrLowerEnvelope
    
def findReponseEndIndex(arrData, dSamplingFreq, nResponses, 
                        nSearchStartIndex = 100,
                        nDiscoutinousDistance = 100):
    """
        This function uses the std of data's variation range to
        determine the end indeces of responses.
        
        Parameters:
        ----
        arrValue: the data
        dSamplingRate: sampling rate of data
        nResponses: number of responses
        nSearchStartIndex: the start index of response searching
        nDiscoutinousDistance: number of points btw two humps
        
        Returns:
        ----
        lsResponseEndIndex: a list of end index of responses
        arrBandwidthSTD: the reference data to find responses
    """
    nWindowSize = 10
    arrUpperEnvelope, \
    arrLowerEnvelope = computeEnvelope(arrData, nWindowSize)
    arrBandWidth = arrUpperEnvelope - arrLowerEnvelope
    arrBandwidthSTD = pd.rolling_std(pd.Series(arrBandWidth), 
                                     nWindowSize, center=True)
    
    lsPeaks = []
    
    # compute the minimal valume of peak candidates
    dMinPeakValume = 0.10 * (np.max(arrData[int(3*dSamplingFreq):-1]) \
                    - np.min(arrData[int(3*dSamplingFreq):-1]) )  
                    
                    
    # select points whose value is larger than dMinPeakValume
    arrRespEndIndexCandidates = np.where(arrBandwidthSTD >= dMinPeakValume)[0]
    
    # only need points indexed after nSearchStartIndex
    arrRespEndIndexCandidates = \
        arrRespEndIndexCandidates[arrRespEndIndexCandidates>=nSearchStartIndex]
    
    # find discontinuous humps
    nHumpStart = 0
    nHumpEnd = nHumpStart+1
    while(nHumpEnd < len(arrRespEndIndexCandidates) ):
        if ( (arrRespEndIndexCandidates[nHumpEnd]- \
              arrRespEndIndexCandidates[nHumpEnd-1]) \
             < nDiscoutinousDistance ):
            nHumpEnd += 1 
        else:
            # note that for numpy.array, the index is 
            # not changed after slicing
            arrHumpIndexRange = arrRespEndIndexCandidates[nHumpStart:nHumpEnd]
            arrHumpValues = arrBandwidthSTD[arrHumpIndexRange] 
            # let the peak point represents this hump
            nPeakIndex = np.argmax(arrHumpValues)
            lsPeaks.append( (nPeakIndex, arrHumpValues[nPeakIndex]) )
            
            nHumpStart = nHumpEnd
            nHumpEnd = nHumpStart+1
     

    if (nHumpEnd > (nHumpStart+1) ): # last subsegment is not count yet
        # note that for numpy.array, the index does 
        # not change after slicing
        arrHumpIndexRange = arrRespEndIndexCandidates[nHumpStart:nHumpEnd]
        arrHumpValues = arrBandwidthSTD[arrHumpIndexRange] 
        # let the peak point represents this hump
        nPeakIndex = np.argmax(arrHumpValues)
        lsPeaks.append((nPeakIndex, arrHumpValues[nPeakIndex]) )
    
    # find the minimal value for top K humps
    lsSortedPeaks = sorted(lsPeaks, key=lambda x: x[1], reverse=True)
    dTopPeakThreshold = lsSortedPeaks[nResponses-1][1] 
    
    # find top k peaks --> their index is the end index of response
    lsResponseEndIndex = [peak[0] for peak in lsPeaks \
                          if peak[1] >= dTopPeakThreshold]
     
    
    return lsResponseEndIndex, arrBandwidthSTD
            
            
    
def splitData(arrData, dSamplingFreq, nResponses, 
              nSegmentsPerRespsonse = 5, 
              dVibrationDuration = 1.4, 
              dIntervalDuration = 0.0, 
              dRestDuration = 1.0):
    """
        Split data into responses and then find segments in 
        each response.
        
        Parameters:
        ----
        arrData: the data
        dSamplingFreq: sampling frequency of data
        nResponses: number of responses in this data
        nSegmentPerResponse: number of segement per response
        dVibrationDuration: duration of each vibration in seconds
        dIntervalDuration: static duration btw vibrations in seconds
        dRestDuration: rest duration btw responses in seconds
        
        Returns:
        ----
        lsResponses: list of segment lists, each of which represent an response
        arrResponseEndIndex: the ending index of each responses
        arrBandwidthSTD: the std of data variation range 
    """
    # find the end of response via the std of variation
    arrResponseEndIndex, \
    arrBandwidthSTD = findReponseEndIndex(arrData,
                                          dSamplingFreq,
                                          nResponses,
                                          int(2*dSamplingFreq),
                                          int(1*dSamplingFreq) )
                                       
    lsResponses = []
    for nRespEndIndex in arrResponseEndIndex:
        if ( (nRespEndIndex- \
              nSegmentsPerRespsonse*(dVibrationDuration+dIntervalDuration) \
              *dSamplingFreq ) \
            < 0.0 ): 
            raise ValueError("Invalid end index of response.")
            
        
        lsSegments = []
        nSegmentEnd = nRespEndIndex
        nSegmentStart = nRespEndIndex
        nCount = 0
        while(nCount < nSegmentsPerRespsonse):
            nSegmentStart = int(nSegmentEnd - dSamplingFreq*dVibrationDuration ) 
            lsSegments.append( (nSegmentStart, nSegmentEnd) )
            nCount += 1
            
            nSegmentEnd = int(nSegmentStart - dSamplingFreq*dIntervalDuration)
            
        lsSegments.reverse()
        lsResponses.append(lsSegments)
        
    return lsResponses, arrResponseEndIndex, arrBandwidthSTD
    


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
#
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
    
    
if __name__ == '__main__':
    
    dSamplingFreq = 320.0

    lsColumnNames = ['x0', 'y0','z0', 'gx0', 'gy0','gz0',
                     'x1', 'y1','z1', 'gx1', 'gy1','gz1']

    nBasicFontSize = 16
    strBasicFontName = "Times new Roman"

    import sys
    sys.exit(0)

#%%  x, y, z @ time domain
    strWorkingDir = "../../data/experiment/user_identification_v2/"
    strFileName = "cyj_t18_l2_p0_0"
    dfData = loadData(strWorkingDir, strFileName, lsColumnNames)
    
    lsAxis2Inspect = ['x0', 'y0', 'z0']
    

    bPlotRawData = False
    bPlotSegmentLine = True
    
    lsColors = lsRGB*6
    nRows= len(lsAxis2Inspect)
    nCols = 2 if bPlotRawData is True else 1
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols, squeeze=False)
    for i, col in enumerate(lsAxis2Inspect):
        arrData = dfData[col].values
        arrFiltered = bp_filter.butter_bandpass_filter(arrData, lowcut=20,
                                                       highcut=120,
                                                       fs=dSamplingFreq,
                                                       order=7)
        axes[i%nRows, 0].plot(arrFiltered, color=lsColors[i])
        axes[i%nRows, 0].set_xlabel("%s_filtered" % col)
        
        
        # plot raw data
        if(bPlotRawData is True):
            axes[i%nRows, 1].plot(arrData, color=lsColors[i])
            axes[i%nRows, 1].set_xlabel(col)
            
        # plot segement lines
        if(bPlotSegmentLine is True):
            lsResponses, arrResponseEndIndex, \
            arrBandwidthSTD = splitData(arrFiltered,
                                        dSamplingFreq, 
                                        nResponses=3,
                                        nSegmentsPerRespsonse=13,
                                        dVibrationDuration=1.4,
                                        dIntervalDuration=0.0,
                                        dRestDuration=1.0)
                                     
            axes[i%nRows, 0].plot(arrBandwidthSTD, color='k')
            
            for subsegment in lsResponses:
                for begin, end in subsegment:
                    axes[i%nRows, 0].axvline(begin, ls="--", color='c', lw=1)
                    axes[i%nRows, 0].axvline(end, ls="--", color='m', lw=2)
                
            
                           
    fig.suptitle(strFileName + "@ time domain",
                 fontname=strBasicFontName,
                 fontsize=nBasicFontSize)
    plt.tight_layout()
    plt.show()
    
#%% plot modulus
    lsAxis2Inspect = ['x0', 'y0', 'z0', 'x1', 'y1', 'z1']
    lsColors = lsRGB*2

    nFFTBatchStart = 0
    nFFTBatchEnd = -1
    

    dfData_filtered = dfData[lsAxis2Inspect]
    for i in xrange(len(dfData_filtered.columns) ):
        if (i % 3 == 0):
            nRows= 1
            nCols = 1
            fig, axes = plt.subplots(nrows=nRows, ncols=nCols, squeeze=False)
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
    lsColors = lsRGB * 10
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
            axes[i].plot(dfData.index[nBPFilterStart:nBPFilterEnd], 
                         arrFiltered, lsColors[i])

        fig.suptitle("bpfilter: %d ~ %d Hz" % (nLowCut, nHighCut), \
                     fontname=strBasicFontName,
                     fontsize=nBasicFontSize)
    plt.tight_layout()
    plt.show()

#%% fft & plot
    strWorkingDir = "../../data/experiment/feasibility/temp/"
    strFileName = "motor_vol_110"
    dfData = loadData(strWorkingDir, strFileName, lsColumnNames)
    
    lsAxis2Inspect = ['x0', 'y0', 'z0']
    lsColors = lsRGB * int(math.ceil(len(lsAxis2Inspect)/3.0 ))
    
    dfData2FFT = dfData[lsAxis2Inspect]
    
    nDCEnd = 5
    nFFTBatchStart = 0
    nFFTBatchEnd = nFFTBatchStart
    nFFTBatchSize = int(dSamplingFreq*200)
    
    nDataLen = dfData2FFT.shape[0]
    while(nFFTBatchStart<nDataLen ):
        nFFTBatchEnd = min( (nFFTBatchStart+nFFTBatchSize),nDataLen)
        
        print nFFTBatchStart, nFFTBatchEnd
        fig, axes = plt.subplots(nrows=len(lsAxis2Inspect), 
                                 ncols=1, squeeze=False)
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
                            color=lsColors[i])
            axes[i, 0].set_xticks(range(0, int(dSamplingFreq/2), 1) )
            axes[i, 0].set_xlabel(lsAxis2Inspect[i] )
    
        fig.suptitle( "%s: %d - %d second" % (strFileName,
                                              nFFTBatchStart/dSamplingFreq,
                                              nFFTBatchEnd/dSamplingFreq),
                     fontname=strBasicFontName,
                     fontsize=nBasicFontSize)
        fig.tight_layout()
        plt.show()
        nFFTBatchStart = nFFTBatchEnd

#%%  visualize specgram
    strWorkingDir = "../../data/experiment/feasibility/temp/"
    strFileName = "yl_trail_r1"
    dfData = loadData(strWorkingDir, strFileName, lsColumnNames)
    
    lsAxis2Inspect = ['x1', 'y1', 'z1']
    lsColors = lsRGB * int(math.ceil(len(lsAxis2Inspect)/3.0 ))

    nFFTBatchStart = 0
    nFFTBatchEnd = -1
    fig, axes = plt.subplots(nrows=len(lsAxis2Inspect), 
                             ncols=1, squeeze=False)
    for i, col in enumerate(lsAxis2Inspect):
        srData2Analyse = dfData.ix[:, col]
        Pxx, freqs, bins, im = axes[i, 0].specgram(
            srData2Analyse.iloc[nFFTBatchStart:nFFTBatchEnd].values,
            mode='psd',
            NFFT=int(dSamplingFreq*1),
            Fs=dSamplingFreq,
            noverlap=0)
        axes[i, 0].set_xlabel(col)
    fig.suptitle(strFileName + "@ specgram",
                 fontname=strBasicFontName,
                 fontsize=nBasicFontSize)
    plt.tight_layout()
    plt.show()

# %% FRF
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

        # get raw data
        arrResp_t = removeGravity(dfResp, nBaseLineStart, nBaseLineEnd).iloc[:, 0]
        arrExc_t = removeGravity(dfExc, nBaseLineStart, nBaseLineEnd).iloc[:, 0]

        nSamples = len(arrResp_t)
        nDCEnd = 10

        # bandpass
        nLowFreq = 5
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
        
        plt.figure()
        plt.plot(np.abs(arrNormalizedResp_f[nDCEnd:nSamples/2]) )
        plt.suptitle("arrNormalizedResp_f")
        
        plt.figure()
        plt.plot(np.abs(arrNormalizedExc_f[nDCEnd:nSamples/2]) )
        plt.suptitle("arrNormalizedExc_f")

        # FRF
        arrFRF = arrResp_f/arrExc_f;

        # plot
        fig = plt.figure()
        
        # magnitude
        ax0 = fig.add_subplot(311)
        xf = np.linspace(nLowFreq, dSamplingFreq/2.0, nSamples/2.0-nDCEnd)
        ax0.plot(xf, pd.rolling_mean(np.abs(arrFRF[nDCEnd:nSamples/2]),
                                     window=100, min_periods=1) )
        ax0.set_ylabel("magnitude")
        
        # phase
        ax1 = fig.add_subplot(312)
        ax1.plot(xf, pd.rolling_mean(np.imag(arrFRF[nDCEnd:nSamples/2]),
                                     window=100, min_periods=1) )
        ax1.set_ylabel("phase")
        
        # complex FRF
        ax2 = fig.add_subplot(313)
        ax2.plot(xf, pd.rolling_mean(arrFRF[nDCEnd:nSamples/2],
                                     window=100, min_periods=1) )
        ax2.set_ylabel("FRF")

        # setup looks

        ax0.set_xticks(np.arange(0, int(dSamplingFreq/2.0)+1, 5.0) )
        ax1.set_xticks(np.arange(0, int(dSamplingFreq/2.0)+1, 5.0) )
        ax2.set_xticks(np.arange(0, int(dSamplingFreq/2.0)+1, 5.0) )

        fig.suptitle(strFileName + ": FRF",
                     fontname=strBasicFontName,
                     fontsize=nBasicFontSize)

        plt.tight_layout()
        nFFTStart = nFFTEnd

    plt.show()

    print("validation is over!")
