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
from EEG_auxiliary_module_guggenmos import run_eeg_svm_guggenmos #, run_eeg_distance_guggenmos
import pdb
import random

#%% study info and load preprocessed data
#study info and filepath 
study        = 'MADeEEG1'
studyPathInd = os.getcwd().find(study)

    
#%% main ffunction that calls the classification functions, gets results, plots and saves
def getandsaveResults(subj,X1,Y1,X2,Y2,condition,event,times,timebin, figNum, n_permutation=1, n_jobs=-1,jitCorrect = None,n_pseudo=5):
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
        acc1, rdm1, coefs1, coefs_scaled1, duration1 = run_eeg_svm_guggenmos(X1,Y1,n_pseudo = 5,n_perm=n_permutation,n_jobs=n_jobs)

        acc2, rdm2, coefs2, coefs_scaled2, duration2 = run_eeg_svm_guggenmos(X2,Y2,n_pseudo = 5,n_perm=n_permutation,n_jobs=n_jobs)
        
        # Create figure of classification accuracy
        print ('subject'+subj+'-'+condition + " done. %.2f minutes" % duration1)
        acc1 = 100*acc1+50
        plt.figure(figNum)
        figNum+=1
        plt.plot(np.arange(times[0]*1000,times[1]*1000,timebin),np.nanmean(acc1,axis=(0,1)),label = condition)
        plt.xlabel('time')
        plt.ylabel('accuracy')  
        
        # Define and save figures and metrics.
        dataPath     = op.join('Results',subj)
        savefile_half1  = 'stimClassification_'+condition+'_'+event+'_'+str(timebin)+'ms_half1_reref.npy'
        saverdm_half1   = 'rdm_'+condition+'_'+event+'_'+str(timebin)+'ms_half1_reref.npy'
        savecoefs_half1 = 'coefs_'+condition+'_'+event+'_'+str(timebin)+'ms_half1_reref.npy'
        savecoefs_scaled_half1 = 'coefs_scaled_'+condition+'_'+event+'_'+str(timebin)+'ms_half1_reref.npy'
        savefig_half1   = 'stimClassification_'+condition+'_'+event+'_'+str(timebin)+'ms_half1_reref.png'
        
        savefile_half2  = 'stimClassification_'+condition+'_'+event+'_'+str(timebin)+'ms_half2_reref.npy'
        saverdm_half2   = 'rdm_'+condition+'_'+event+'_'+str(timebin)+'ms_half2_reref.npy'
        savecoefs_half2 = 'coefs_'+condition+'_'+event+'_'+str(timebin)+'ms_half2_reref.npy'
        savecoefs_scaled_half2 = 'coefs_scaled_'+condition+'_'+event+'_'+str(timebin)+'ms_half2_reref.npy'
        savefig_half2   = 'stimClassification_'+condition+'_'+event+'_'+str(timebin)+'ms_half2_reref.png'

        if jitCorrect == None:
            savepath1 = op.join(dataPath,'classification_guggenmos_noJitFix')
            savepath2 = op.join(dataPath,'figures_guggenmos_noJitFix')
        else:
            savepath1 = op.join(dataPath,'classification_guggenmos')
            savepath2 = op.join(dataPath,'figures_guggenmos')

        if not op.isdir(savepath1):
            os.mkdir(savepath1)
        np.save(op.join(savepath1,savefile_half1),acc1)
        np.save(op.join(savepath1,saverdm_half1),rdm1)
        np.save(op.join(savepath1,savecoefs_half1),coefs1)
        np.save(op.join(savepath1,savecoefs_scaled_half1),coefs_scaled1)      
        np.save(op.join(savepath1,savefile_half2),acc2)
        np.save(op.join(savepath1,saverdm_half2),rdm2)
        np.save(op.join(savepath1,savecoefs_half2),coefs2)
        np.save(op.join(savepath1,savecoefs_scaled_half2),coefs_scaled2)
        if not op.isdir(savepath2):
            os.mkdir(savepath2)
        plt.savefig(op.join(savepath2,savefig_half1))
        plt.savefig(op.join(savepath2,savefig_half2))

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
    ####------------------------- Faces -----------------------------------------
    if condition == 'Faces'or condition == 'All':
        # Define basic condition information
        faces = [1,2,3,4,5]
        n_faces = 5

        #create averages based on trial types
        chan = epochs.ch_names.index('PO8')
        ttime = np.linspace(times[0],times[1],AllEpochs.shape[2])
        valList = []        
        plt.figure(figNum)
        figNum+=1
              
        #find trials with the same face, and then average them together in "chunks". Becomes "X" variable later on.
        n_faces = 5
        for f in range(n_faces):
            behavData1 = behavData[:300]
            temp1 = AllEpochs[behavData1.Trial[behavData1.Face == f+1]-1,:,:] #-1 to start trial index from 0
            n_avg = 2
            nChunk  = int(temp1.shape[0]/n_avg)
            
            if timebin == 0:
                faceAvg = np.zeros((nChunk,temp1.shape[1],temp1.shape[2]))
                for c in range(nChunk):
                    faceAvg[c,:,:] = np.mean(temp1[c*n_avg:(c+1)*n_avg,:,:],0)
                
            else:
                nBin   = int((times[1]-times[0])*1000/timebin) # number of bins
                n_tpnt = int(timebin/(1000/epochs.info['sfreq']))
                faceAvg= np.zeros((nChunk,temp1.shape[1],nBin))
                for c in range(nChunk):
                    temp = np.mean(temp1[c*n_avg:(c+1)*n_avg,:,:],0)
                    for t in range(nBin):
                        faceAvg[c,:,t] = np.mean(temp[:,t*n_tpnt:(t+1)*n_tpnt],1)           
            if f>0:
                X1 = np.concatenate((X1,faceAvg), axis = 0)
            else:
                X1 = faceAvg
                
        # Set up label matrix. 
        Y1 = np.concatenate((np.ones((1,nChunk)),2*np.ones((1,nChunk)),3*np.ones((1,nChunk)),
                                    4*np.ones((1,nChunk)),5*np.ones((1,nChunk))),axis = 1)                 
        Y1 = np.int32(Y1[0])


        for f in range(n_faces):
            behavData2 = behavData[300:]
            temp2 = AllEpochs[behavData2.Trial[behavData2.Face == f+1]-1,:,:] #-1 to start trial index from 0
            n_avg = 2
            nChunk  = int(temp2.shape[0]/n_avg)
            
            if timebin == 0:
                faceAvg = np.zeros((nChunk,temp2.shape[1],temp2.shape[2]))
                for c in range(nChunk):
                    faceAvg[c,:,:] = np.mean(temp2[c*n_avg:(c+1)*n_avg,:,:],0)
                
            else:
                nBin   = int((times[1]-times[0])*1000/timebin) # number of bins
                n_tpnt = int(timebin/(1000/epochs.info['sfreq']))
                faceAvg= np.zeros((nChunk,temp2.shape[1],nBin))
                for c in range(nChunk):
                    temp = np.mean(temp2[c*n_avg:(c+1)*n_avg,:,:],0)
                    for t in range(nBin):
                        faceAvg[c,:,t] = np.mean(temp[:,t*n_tpnt:(t+1)*n_tpnt],1)           
            if f>0:
                X2 = np.concatenate((X2,faceAvg), axis = 0)
            else:
                X2 = faceAvg
                
        # Set up label matrix. 
        Y2 = np.concatenate((np.ones((1,nChunk)),2*np.ones((1,nChunk)),3*np.ones((1,nChunk)),
                                    4*np.ones((1,nChunk)),5*np.ones((1,nChunk))),axis = 1)                 
        Y2 = np.int32(Y2[0])
        
        
        print(n_permutation)
        
        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X1,Y1,X2,Y2,'Faces',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect= jitCorrect,n_pseudo = 5)
        del X1,Y1, X2, Y2
        
    #%%####------------------------- Colors -----------------------------------------
    if condition == 'Colors' or condition == 'All':
        # Define basic condition information
        colors = [1,2,3,4,5]
        n_colors = 5

        #create averages based on trial types
        chan = epochs.ch_names.index('PO8')
        ttime = np.linspace(times[0],times[1],AllEpochs.shape[2])
        valList = []        
        plt.figure(figNum)
        figNum+=1
        
        
        #find trials with the same color, and then average them together in "chunks". Becomes "X" variable later on.
        n_colors = 5
        for f in range(n_colors):
            behavData1 = behavData[:300]
            temp1 = AllEpochs[behavData1.Trial[behavData1.Color == f+1]-1,:,:] #-1 to start trial index from 0
            n_avg = 2
            nChunk  = int(temp1.shape[0]/n_avg)
            
            if timebin == 0:
                colorAvg = np.zeros((nChunk,temp1.shape[1],temp1.shape[2]))
                for c in range(nChunk):
                    colorAvg[c,:,:] = np.mean(temp1[c*n_avg:(c+1)*n_avg,:,:],0)
                
            else:
                nBin   = int((times[1]-times[0])*1000/timebin) # number of bins
                n_tpnt = int(timebin/(1000/epochs.info['sfreq']))
                colorAvg= np.zeros((nChunk,temp1.shape[1],nBin))
                for c in range(nChunk):
                    temp = np.mean(temp1[c*n_avg:(c+1)*n_avg,:,:],0)
                    for t in range(nBin):
                        colorAvg[c,:,t] = np.mean(temp[:,t*n_tpnt:(t+1)*n_tpnt],1)           
            if f>0:
                X1 = np.concatenate((X1,colorAvg), axis = 0)
            else:
                X1 = colorAvg
                
        # Set up label matrix. 
        Y1 = np.concatenate((np.ones((1,nChunk)),2*np.ones((1,nChunk)),3*np.ones((1,nChunk)),
                                    4*np.ones((1,nChunk)),5*np.ones((1,nChunk))),axis = 1)                 
        Y1 = np.int32(Y1[0])


        for f in range(n_colors):
            behavData2 = behavData[300:]
            temp2 = AllEpochs[behavData2.Trial[behavData2.Color == f+1]-1,:,:] #-1 to start trial index from 0
            n_avg = 2
            nChunk  = int(temp2.shape[0]/n_avg)
            
            if timebin == 0:
                colorAvg = np.zeros((nChunk,temp2.shape[1],temp2.shape[2]))
                for c in range(nChunk):
                    colorAvg[c,:,:] = np.mean(temp2[c*n_avg:(c+1)*n_avg,:,:],0)
                
            else:
                nBin   = int((times[1]-times[0])*1000/timebin) # number of bins
                n_tpnt = int(timebin/(1000/epochs.info['sfreq']))
                colorAvg= np.zeros((nChunk,temp2.shape[1],nBin))
                for c in range(nChunk):
                    temp = np.mean(temp2[c*n_avg:(c+1)*n_avg,:,:],0)
                    for t in range(nBin):
                        colorAvg[c,:,t] = np.mean(temp[:,t*n_tpnt:(t+1)*n_tpnt],1)           
            if f>0:
                X2 = np.concatenate((X2,colorAvg), axis = 0)
            else:
                X2 = colorAvg
                
        # Set up label matrix. 
        Y2 = np.concatenate((np.ones((1,nChunk)),2*np.ones((1,nChunk)),3*np.ones((1,nChunk)),
                                    4*np.ones((1,nChunk)),5*np.ones((1,nChunk))),axis = 1)                 
        Y2 = np.int32(Y2[0])
        
        
        
        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X1,Y1,X2,Y2,'Colors',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect= jitCorrect,n_pseudo = 5)
        del X1,Y1, X2, Y2
    
    #%%####------------------------- Values-2 -----------------------------------------

def find_nearest_idx(array, value):
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
