# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:13:40 2015

@author: jason
"""

#%%
import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
import pandas as pd
import bp_filter
#%%
dSamplingFreq = 200.0

lsColors = ['r', 'g', 'b', 'r', 'g', 'b']

strWorkingDir = "D:\\yanglin\\baidu_cloud\\research\\my_research\\resonance_lab\\data\\experiment_on_locations\\"
strFileName = "ww_4_45.txt"


#dfAcc = pd.read_csv(strWorkingDir+strFileName, index_col=0, header=0)
dfAcc = pd.read_csv(strWorkingDir+strFileName, \
    names=['x0', 'y0','z0','x1', 'y1','z1'], dtype=np.float32)

                
##==============================================================================
## change index and data to required format
##==============================================================================
#timeIndex = np.add(dfAcc.index.tolist(), 8*60*60*1000)
#dfAcc.set_index(pd.to_datetime(timeIndex, unit='ms', utc=True), inplace=True)
#dfAcc['x'] = dfAcc['x'].astype(np.float32)
#dfAcc['y'] = dfAcc['y'].astype(np.float32)
#dfAcc['z'] = dfAcc['z'].astype(np.float32)

#==============================================================================
# clear data
#==============================================================================
lsMask = [True ]* len(dfAcc)
for col in dfAcc.columns:
    lsMask = lsMask & (dfAcc[col] != -1) & (~dfAcc[col].isnull() )
dfAcc = dfAcc[lsMask]
#%% 
#==============================================================================
# visualize time-domain
#==============================================================================
#nCol2Plot = -1
nCol2Plot = 3
fig, axes = plt.subplots(nrows=len(dfAcc.columns[:nCol2Plot]), ncols=1)
nTDPlotStart = 0
#nTDPlotEnd = nTDPlotStart + (1*60*dSamplingFreq)
nTDPlotEnd = -1

for i, col in enumerate(dfAcc.columns[:nCol2Plot]):
    dfAcc[col].iloc[nTDPlotStart:nTDPlotEnd].plot(color=lsColors[i], ax=axes[i], legend=False)

fig.suptitle(strFileName + "@ time domain", fontname='Times new Roman', fontsize=16)
plt.tight_layout()
plt.show()

#%%
#==============================================================================
# band pass filter
#==============================================================================
nBPFilterStart = 0
nBPFilterEnd = nBPFilterStart + (5*60*dSamplingFreq)

nBandWidth = 10
for nLowCut in xrange(0, int(dSamplingFreq/2.0), nBandWidth):
    nHighCut = nLowCut + nBandWidth
    fig, axes = plt.subplots(nrows=len(dfAcc.columns), ncols=1)
    for i, col in enumerate(dfAcc.columns):
        # bp filter
        arrFiltered= bp_filter.butter_bandpass_filter( \
        dfAcc[col].values[nBPFilterStart:nBPFilterEnd], \
        nLowCut, nHighCut, dSamplingFreq, order=9)
        
        #visualize
        axes[i].plot(dfAcc.index[nBPFilterStart:nBPFilterEnd], arrFiltered, lsColors[i])

    fig.suptitle("bpfilter: %d ~ %d Hz" % (nLowCut, nHighCut), \
                 fontname='Times new Roman', fontsize=16)
plt.tight_layout()
plt.show()

##%% 
##==============================================================================
## visualized freq-domain
##==============================================================================
#nFFTDataLen = len(dfAcc)
##nFFTDataLen = int(5*60*dSamplingFreq)
#
#nFFTStart = 0
#nFFTEnd = 0
#nBinSize = int(dSamplingFreq)*1000
#while (nFFTStart < nFFTDataLen ):
#    nFFTEnd = min(nFFTDataLen, nFFTStart+nBinSize)
#    
#    fig, axes = plt.subplots(nrows=len(dfAcc.columns), ncols=1)
#    for i, col in enumerate(dfAcc.columns):
#        arrTimeDataSlice = dfAcc[col].values[nFFTStart:nFFTEnd]
#        
#        nSamples = len(arrTimeDataSlice)
#        nFDStart = 3
#        
#        # high pass filter
#        arrFiltered= bp_filter.butter_bandpass_filter( \
#        arrTimeDataSlice, 50, 84, dSamplingFreq, order=9)
#        
#        # fft
#        arrFreqData = fftpack.fft(arrFiltered)
#        arrFreqData_normalized = arrFreqData/(nSamples*1.0)
#        
#        xf = np.linspace(float(nFDStart), dSamplingFreq/2.0, nSamples/2.0-nFDStart)
#        
#        # plot
#        axes[i].plot(xf, np.abs(arrFreqData_normalized[nFDStart:nSamples/2]), lsColors[i])
#
#        # setup looks
##        axes[i].set_xlabel("Freq", fontname='Times new Roman', fontsize=18)
##        axes[i].set_ylabel("Power", fontname='Times new Roman', fontsize=18)
#        
#        axes[i].set_xticks(np.arange(0, int(dSamplingFreq/2.0)+1, 5.0) )
#        
##        axes[i].set_ylim(0.0, 100.0)
##        axes[i].set_xlim(0.0, 50.0)
#        
##        axes[i].set_yscale('log');
#        
#        plt.setp(axes[i].get_xticklabels(), fontname='Times new Roman', fontsize=16, rotation=90)
#        plt.setp(axes[i].get_yticklabels(), fontname='Times new Roman', fontsize=16)
#    
#    fig.suptitle(strFileName + "@ frequency domain", fontname='Times new Roman', fontsize=16)
#    plt.tight_layout()
#    nFFTStart = nFFTEnd
#    
#plt.show()
#
#
#
##%%
##==============================================================================
## visualize specgram
##==============================================================================
#fig = plt.figure()
#ax = fig.add_subplot(111)
#Pxx, freqs, bins, im = ax.specgram(dfAcc['y1'].values, NFFT=int(dSamplingFreq), Fs=dSamplingFreq, noverlap=int(dSamplingFreq/2.0))
#ax.set_xlabel("Time", fontname='Times new Roman', fontsize=18)
#ax.set_ylabel("Frequency", fontname='Times new Roman', fontsize=18)
#fig.colorbar(im).set_label('power')
#plt.tight_layout()
#
## %% 
##==============================================================================
## FRF
##==============================================================================
#nFFTDataLen = len(dfAcc)
##nFFTDataLen = int(5*60*dSamplingFreq)
#
#nFFTStart = 0
#nFFTEnd = 0
#nBinSize = int(dSamplingFreq)*1000
#while (nFFTStart < nFFTDataLen ):
#    nFFTEnd = min(nFFTDataLen, nFFTStart+nBinSize)
#    
#    fig, axes = plt.subplots(nrows=(len(dfAcc.columns)/2), ncols=1)
#    for i in xrange((len(dfAcc.columns)/2)):
#        arrTimeDataSlice_resp = dfAcc.iloc[:, i].values[nFFTStart:nFFTEnd]
#        arrTimeDataSlice_exc = dfAcc.iloc[:, i+3].values[nFFTStart:nFFTEnd]
#        
#        nSamples = len(arrTimeDataSlice_resp)
#        nFDStart = 3
#        
#        # fft of response
#        arrFiltered_resp = bp_filter.butter_bandpass_filter(arrTimeDataSlice_resp, \
#                            50, 84, dSamplingFreq, order=9)
#        arrFreqData_resp = fftpack.fft(arrFiltered_resp)
#        arrFreqData_resp = arrFreqData_resp/(nSamples*1.0)
#        
#        # fft of input
#        arrFiltered_exc = bp_filter.butter_bandpass_filter(arrTimeDataSlice_exc, \
#                            50, 84, dSamplingFreq, order=9)
#        arrFreqData_exc = fftpack.fft(arrFiltered_exc)
#        arrFreqData_exc = arrFreqData_exc/(nSamples*1.0)
#        
#        # FRF
#        arrFRF = arrFreqData_resp/arrFreqData_exc;
#        
#        # plot
#        xf = np.linspace(float(nFDStart), dSamplingFreq/2.0, nSamples/2.0-nFDStart)
#        axes[i].plot(xf, abs(arrFRF[nFDStart:nSamples/2]), lsColors[i])
#
##        # setup looks
##        axes[i].set_xlabel("Freq", fontname='Times new Roman', fontsize=18)
##        axes[i].set_ylabel("FRF", fontname='Times new Roman', fontsize=18)
#        
#        axes[i].set_xticks(np.arange(0, int(dSamplingFreq/2.0)+1, 5.0) )
#        
##        axes[i].set_ylim(0.0, 100.0)
##        axes[i].set_xlim(0.0, 50.0)
#        
##        axes[i].set_yscale('log');
#        
#        plt.setp(axes[i].get_xticklabels(), fontname='Times new Roman', fontsize=16, rotation=90)
#        plt.setp(axes[i].get_yticklabels(), fontname='Times new Roman', fontsize=16)
#    
#    fig.suptitle(strFileName + ": FRF", fontname='Times new Roman', fontsize=16)
#    plt.tight_layout()
#    nFFTStart = nFFTEnd
#    
#plt.show()

print("validation is over!")