## Please note that this script requires the output from "preprocessingEEGData". Please adjust the loadpath of files accordingly.
## It also requires the processed behavioral data csvs.

#%% load modules and packages
import mne
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from EEG_auxiliary_module_guggenmos import run_eeg_svm_guggenmos_SVR #, run_eeg_distance_guggenmos
import pdb
import random

# v3 - with higher npsuedo than 5

#%% study info and load preprocessed data
#study info and filepath 
study        = 'MADeEEG1'
studyPathInd = os.getcwd().find(study)

#%% main ffunction that calls the classification functions, gets results, plots and saves
def getandsaveResults(subj,X,Y,condition,event,times,timebin, figNum, n_permutation=1, n_jobs=-1,jitCorrect = None,n_pseudo=5):
 ### subj: Subject Number
 ### X: Selected epochs
 ### Y: Labels
 ### condition: Which conditions to extract (Faces, Colors, FaceValues, ColorValues, Values, Response)
 ### event: Which epochs to extract (Stimulus or Dot)
 ### times: Time range
 ### timebin: Whether time was averaged into bins or not.
 ### figNum: Iterates over to make new figures
 ### n_permutation:
 ### n_jobs: Number of jobs to run in parallel. Default of -1 sets it equal to numnber of CPU cores.
 ### jitCorrect: Creates different save folders depending on whether jitter correction has been applied or not.
    
        # Calculate classification score
        score, coefs, coefs_scaled = run_eeg_svm_guggenmos_SVR(X,Y,n_pseudo = n_pseudo,n_perm=20,n_jobs=n_jobs)

        # Create figure of classification accuracy
        print ('subject'+subj+'-'+ condition + " done.")
        plt.figure(figNum)
        figNum+=1
        plt.plot(np.arange(times[0]*1000,times[1]*1000,timebin),score,label = condition)
        plt.xlabel('time')
        plt.ylabel('accuracy')  
        
        # Define and save figures and metrics.
        dataPath     = op.join('Results',subj)
        savefile  = 'stimClassification_'+condition+'_'+event+'_'+str(timebin)+'ms_SVR_reref.npy'
        savecoefs = 'coefs_'+condition+'_'+event+'_'+str(timebin)+'ms_SVR.npy'
        savecoefs_scaled = 'coefs_scaled_'+condition+'_'+event+'_'+str(timebin)+'ms_SVR_reref.npy'
        savefig   = 'stimClassification_'+condition+'_'+event+'_'+str(timebin)+'ms_SVR_reref.png'
        if jitCorrect == None:
            savepath1 = op.join(dataPath,'classification_guggenmos_noJitFix')
            savepath2 = op.join(dataPath,'figures_guggenmos_noJitFix')
        else:
            savepath1 = op.join(dataPath,'classification_guggenmos')
            savepath2 = op.join(dataPath,'figures_guggenmos')

        if not op.isdir(savepath1):
            os.mkdir(savepath1)
        np.save(op.join(savepath1,savefile),score)
        np.save(op.join(savepath1,savecoefs),coefs)
        np.save(op.join(savepath1,savecoefs_scaled),coefs_scaled)
        if not op.isdir(savepath2):
            os.mkdir(savepath2)
        plt.savefig(op.join(savepath2,savefig))

        return figNum
    
#%% start classification
def runClassification (subj,condition,timebin = 0,event = 'Stimulus',n_permutation=0,n_jobs=-1,jitCorrect = None):
 ### subj: subject number
 ### condition: which condition to extract (All, Faces, Colors, FaceValues, ColorValues, Values, Response)
 ### timebin: Size of timebin to average by (default = 0)
 ### event: Which epochs to extract (Stimulus or Dot; default = Stimulus)
 ### n_permutation: 
 ### n_jobs:  Number of jobs to run in parallel. Default of -1 sets it equal to numnber of CPU cores.
 ### jitCorrect: Whether to use jitter-corrected files or not (Default = not).

    #load in the preprocessed data
    dataPath     = op.join('Results',subj)
    file_name= 'stimData-epo_v2_reref.fif'
    fname = op.join(dataPath,file_name)
    epochs = mne.read_epochs(fname)
    
    # get trial info
    behavPath = op.join(os.getcwd()[:studyPathInd+len(study)],'Analysis','Matlab','Behavior','csvfiles')
    behavData = pd.read_csv(op.join(behavPath, subj+'.csv'))
    figNum = 1
    
    #determine time window and crop 
    if event == 'Stimulus':
        times = (-.2,1)
    elif event == 'Dot':
        times = (.8,2)
    elif event == 'Stimulus_long':
        times = (-.2,2)
        
    if jitCorrect == None:
        AllEpochs = epochs.copy().crop(times[0],times[1]).get_data()
    else:
        jitCorrectDataFile = op.join(dataPath,'stimJitterCorrected50.npy')
        temp = np.load(jitCorrectDataFile)
        timeVect = np.linspace(-.2,2.2,temp.shape[2])
        t0 = find_nearest_idx(timeVect,times[0])
        tE = find_nearest_idx(timeVect,times[1])
        AllEpochs = temp[:,:,t0:tE+1]
        del temp,timeVect
    
    # crop the time window of interest
  
    #%%####------------------------- Values-2 -----------------------------------------
    if condition == 'Value-2' or condition == 'All':
        newValues = set(behavData.Value)
        newValues

        valList = []

        for v in newValues:
         # if v <0:
            epochsValue = AllEpochs[(behavData.Value == v),:,:]

        
            Y = np.empty((0))
            for curr_v,v in enumerate(newValues):
                temp = AllEpochs[(behavData.Value == v) ,:,:] #-1 to start trial index from 0
                #average trials
                n_avg = 2
                nChunk  = int(temp.shape[0]/n_avg)
                if timebin == 0:
                    valueAvg = np.zeros((nChunk,temp.shape[1],temp.shape[2]))
                    for c in range(nChunk):
                        valueAvg[c,:,:] = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                        Y = np.append(Y,v)
                else:
                    nBin   = int((times[1]-times[0])*1000/timebin) # number of bins
                    n_tpnt = int(timebin/(1000/epochs.info['sfreq']))
                    valueAvg= np.zeros((nChunk,temp.shape[1],nBin))
                    for c in range(nChunk):
                        temp1 = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                        for t in range(nBin):
                            valueAvg[c,:,t] = np.mean(temp1[:,t*n_tpnt:(t+1)*n_tpnt],1)    
                        Y = np.append(Y,v)
                if curr_v>0:
                    X = np.concatenate((X,valueAvg), axis = 0)
                else:
                    X = valueAvg

        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X,Y,'Value-2',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 5)

        del X,Y
        
     #%%####------------------------- Face Values -----------------------------------------
    if condition == 'FaceValues' or condition == 'All':
        newValues = set(behavData.FaceValue)
        newValues


        valList = []

        for v in newValues:
            epochsValue = AllEpochs[(behavData.FaceValue == v),:,:]
            Y = np.empty((0))
            for curr_v,v in enumerate(newValues):
                temp = AllEpochs[(behavData.FaceValue == v) ,:,:] #-1 to start trial index from 0
                #average trials
                n_avg = 2
                nChunk  = int(temp.shape[0]/n_avg)
                if timebin == 0:
                    valueAvg = np.zeros((nChunk,temp.shape[1],temp.shape[2]))
                    for c in range(nChunk):
                        valueAvg[c,:,:] = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                        Y = np.append(Y,v)
                else:
                    nBin   = int((times[1]-times[0])*1000/timebin) # number of bins
                    n_tpnt = int(timebin/(1000/epochs.info['sfreq']))
                    valueAvg= np.zeros((nChunk,temp.shape[1],nBin))
                    for c in range(nChunk):
                        temp1 = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                        for t in range(nBin):
                            valueAvg[c,:,t] = np.mean(temp1[:,t*n_tpnt:(t+1)*n_tpnt],1)    
                        Y = np.append(Y,v)
                if curr_v>0:
                    X = np.concatenate((X,valueAvg), axis = 0)
                else:
                    X = valueAvg
        
        figNum = getandsaveResults(subj,X,Y,'FaceValues',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 60)
        del X,Y    
    #%%####------------------------- Color Values -----------------------------------------
    if condition == 'ColorValues' or condition == 'All':
        newValues = set(behavData.ColorValue)
        newValues

        valList = []

        for v in newValues:
            epochsValue = AllEpochs[(behavData.ColorValue == v),:,:]

        
            # n_values = len(values)
            Y = np.empty((0))
            for curr_v,v in enumerate(newValues):
                temp = AllEpochs[(behavData.ColorValue == v) ,:,:] #-1 to start trial index from 0
                #average trials
                n_avg = 2
                nChunk  = int(temp.shape[0]/n_avg)
                if timebin == 0:
                    valueAvg = np.zeros((nChunk,temp.shape[1],temp.shape[2]))
                    for c in range(nChunk):
                        valueAvg[c,:,:] = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                        Y = np.append(Y,v)
                else:
                    nBin   = int((times[1]-times[0])*1000/timebin) # number of bins
                    n_tpnt = int(timebin/(1000/epochs.info['sfreq']))
                    valueAvg= np.zeros((nChunk,temp.shape[1],nBin))
                    for c in range(nChunk):
                        temp1 = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                        for t in range(nBin):
                            valueAvg[c,:,t] = np.mean(temp1[:,t*n_tpnt:(t+1)*n_tpnt],1)    
                        Y = np.append(Y,v)
                if curr_v>0:
                    X = np.concatenate((X,valueAvg), axis = 0)
                else:
                    X = valueAvg
        
        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X,Y,'ColorValues',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 60)

        del X,Y       
    
#%%####-------------------------  -----------------------------------------

def find_nearest_idx(array, value):
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
