# -*- coding: utf-8 -*-
"""
Feasibility Test
Created on Fri Oct 02 21:02:37 2015

@author: jason
"""
import single_data as sd
import bp_filter

import matplotlib.pyplot as plt
import math
import scipy.fftpack as fftpack
import numpy as np



if __name__ == "__main__":
#%% setting
    dSamplingFreq = 320.0

    lsColumnNames = ['x0', 'y0','z0', 'gx0', 'gy0','gz0',
                     'x1', 'y1','z1', 'gx1', 'gy1','gz1']

    nBasicFontSize = 16
    strBasicFontName = "Times new Roman"
    
    lsRGB = ['r', 'g', 'b']
    
#%% temporal data
    strWorkingDir = "../../data/experiment/user_identification_v2/"
    lsFileNames = ["cyj_t17_l2_p0_0"]
    
    lsAxis2Inspect = ['x0', 'y0', 'z0']
    lsColors = lsRGB*6
    nRows = len(lsAxis2Inspect)
    nCols = len(lsFileNames)
    nFigWidth, nFigHeight = plt.figaspect(4.0/3.0)
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols, 
                             squeeze=False, sharex=True,
                             sharey=True, 
                             figsize=(nFigWidth, nFigHeight) )
    
    for nDataID, strFileName in enumerate(lsFileNames):
        dfData = sd.loadData(strWorkingDir, strFileName, lsColumnNames)

        for nColID, strCol in enumerate(lsAxis2Inspect):
            arrData = dfData[strCol].values
            arrFiltered = bp_filter.butter_bandpass_filter(arrData,
                                                           lowcut=20,
                                                           highcut=120,
                                                           fs=dSamplingFreq,
                                                           order=10)
            axes[nColID, nDataID].plot(arrFiltered, color=lsColors[nColID])
            axes[nColID, nDataID].set_ylim(-5000, 5000)
            axes[nColID, nDataID].set_ylabel("%s" % strCol,
                                             rotation=0,
                                             fontname=strBasicFontName,
                                             fontsize=nBasicFontSize)
            axes[nColID, nDataID].set_xlabel("Time(sec)",
                                             rotation=0,
                                             fontname=strBasicFontName,
                                             fontsize=nBasicFontSize)
            xTickLabels = axes[nColID, nDataID].xaxis.get_ticklabels()
            plt.setp(xTickLabels, fontname=strBasicFontName,
                     size=nBasicFontSize)
            yTickLabels = axes[nColID, nDataID].yaxis.get_ticklabels()
            plt.setp(yTickLabels, fontname=strBasicFontName,
                     size=nBasicFontSize)
            axes[nColID, nDataID].grid(True)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0) )
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0) )
    plt.tight_layout()
    plt.show()
    
#%% freq. data
    strWorkingDir = "../../data/experiment/user_identification_v2/"
    strFileName = "cyj_t17_l2_p0_0"
    dfData = sd.loadData(strWorkingDir, strFileName, lsColumnNames)
    
    lsAxis2Inspect = ['x0', 'y0', 'z0']
    lsColors = lsRGB * int(math.ceil(len(lsAxis2Inspect)/3.0) )
    dcYLim = {'x0': 100, 'y0':150, 'z0':100}
    
    dfData2FFT = dfData[lsAxis2Inspect]
    
    nDCEnd = 5
    nFFTStart = 0
    nFFTEnd = 7100
    
    nFFTBatchStart = nFFTStart
    nFFTBatchEnd = nFFTBatchStart
    nFFTBatchSize = int(dSamplingFreq*200)
    
    while(nFFTBatchStart < nFFTEnd ):
        nFFTBatchEnd = min( (nFFTBatchStart+nFFTBatchSize), nFFTEnd)
        
        print nFFTBatchStart, nFFTBatchEnd
        nFigWidth, nFigHeight = plt.figaspect(4.0/3.0)
        fig, axes = plt.subplots(nrows=len(lsAxis2Inspect), 
                                 ncols=1, squeeze=False,
                                 figsize=(nFigWidth, nFigHeight) )
        for nColID, strCol in enumerate(dfData2FFT.columns):
            srAxis = dfData2FFT.ix[nFFTBatchStart:nFFTBatchEnd, strCol]
            arrTemproalData = srAxis.values
            
            # bandpass
            arrFiltered = bp_filter.butter_bandpass_filter(arrTemproalData,
                                                           lowcut=20,
                                                           highcut=120,
                                                           fs=dSamplingFreq,
                                                           order=10)
            # fft
            nSamples = len(arrFiltered)
            arrFreqData = fftpack.fft(arrFiltered)
            arrNormalizedPower = abs(arrFreqData)/(nSamples*1.0)
    
            dResolution = dSamplingFreq/(nSamples*1.0)
            arrFreqIndex = np.linspace(nDCEnd*dResolution, 
                                       dSamplingFreq/2.0, 
                                       nSamples/2-nDCEnd)
            axes[nColID, 0].plot(arrFreqIndex, 
                            arrNormalizedPower[nDCEnd:nSamples/2],
                            color=lsColors[nColID])
            
            tpYLim = (0, dcYLim[strCol])
            axes[nColID, 0].set_ylim(tpYLim)
            axes[nColID, 0].set_ylabel("%s \n Power" % strCol,
                                  rotation=90,
                                  fontname=strBasicFontName,
                                  fontsize=nBasicFontSize)
            axes[nColID, 0].set_xlabel("Frequency (Hz)",
                                  rotation=0,
                                  fontname=strBasicFontName,
                                  fontsize=nBasicFontSize)
        fig.tight_layout()
        plt.show()
        nFFTBatchStart = nFFTBatchEnd