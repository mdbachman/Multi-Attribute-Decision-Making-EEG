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
        acc, rdm, coefs, coefs_scaled, duration = run_eeg_svm_guggenmos(X,Y,n_pseudo = n_pseudo,n_perm=n_permutation,n_jobs=n_jobs)

        # Create figure of classification accuracy
        print ('subject'+subj+'-'+condition + " done. %.2f minutes" % duration)
        acc = 100*acc+50
        plt.figure(figNum)
        figNum+=1
        plt.plot(np.arange(times[0]*1000,times[1]*1000,timebin),np.nanmean(acc,axis=(0,1)),label = condition)
        plt.xlabel('time')
        plt.ylabel('accuracy')  
        
        # Define and save figures and metrics.
        dataPath     = op.join('Results',subj)
        savefile  = 'stimClassification_'+condition+'_'+event+'_'+str(timebin)+'ms_v2_reref.npy'
        saverdm   = 'rdm_'+condition+'_'+event+'_'+str(timebin)+'ms_v2_reref.npy'
        savecoefs = 'coefs_'+condition+'_'+event+'_'+str(timebin)+'ms_v2_reref.npy'
        savecoefs_scaled = 'coefs_scaled_'+condition+'_'+event+'_'+str(timebin)+'ms_v2_reref.npy'
        savefig   = 'stimClassification_'+condition+'_'+event+'_'+str(timebin)+'ms_v2_reref.png'
        if jitCorrect == None:
            savepath1 = op.join(dataPath,'classification_guggenmos_noJitFix')
            savepath2 = op.join(dataPath,'figures_guggenmos_noJitFix')
        else:
            savepath1 = op.join(dataPath,'classification_guggenmos')
            savepath2 = op.join(dataPath,'figures_guggenmos')

        if not op.isdir(savepath1):
            os.mkdir(savepath1)
        np.save(op.join(savepath1,savefile),acc)
        np.save(op.join(savepath1,saverdm),rdm)
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
            temp = AllEpochs[behavData.Trial[behavData.Face == f+1]-1,:,:] #-1 to start trial index from 0
            n_avg = 2
            nChunk  = int(temp.shape[0]/n_avg)
            
            if timebin == 0:
                faceAvg = np.zeros((nChunk,temp.shape[1],temp.shape[2]))
                for c in range(nChunk):
                    faceAvg[c,:,:] = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                
            else:
                nBin   = int((times[1]-times[0])*1000/timebin) # number of bins
                n_tpnt = int(timebin/(1000/epochs.info['sfreq']))
                faceAvg= np.zeros((nChunk,temp.shape[1],nBin))
                for c in range(nChunk):
                    temp1 = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                    for t in range(nBin):
                        faceAvg[c,:,t] = np.mean(temp1[:,t*n_tpnt:(t+1)*n_tpnt],1)           
            if f>0:
                X = np.concatenate((X,faceAvg), axis = 0)
            else:
                X = faceAvg
                
        # Set up label matrix. 
        Y = np.concatenate((np.ones((1,nChunk)),2*np.ones((1,nChunk)),3*np.ones((1,nChunk)),
                                    4*np.ones((1,nChunk)),5*np.ones((1,nChunk))),axis = 1)                 
        Y = np.int32(Y[0])
        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X,Y,'Faces',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect= jitCorrect,n_pseudo = 5)
        del X,Y
        
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
            temp = AllEpochs[behavData.Trial[behavData.Color == f+1]-1,:,:] #-1 to start trial index from 0
            n_avg = 2
            nChunk  = int(temp.shape[0]/n_avg)
            # pdb.set_trace() 
            if timebin == 0:
                colorAvg = np.zeros((nChunk,temp.shape[1],temp.shape[2]))
                for c in range(nChunk):
                    colorAvg[c,:,:] = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                
            else:
                nBin   = int((times[1]-times[0])*1000/timebin) # number of bins
                n_tpnt = int(timebin/(1000/epochs.info['sfreq']))
                colorAvg= np.zeros((nChunk,temp.shape[1],nBin))
                for c in range(nChunk):
                    temp1 = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                    for t in range(nBin):
                        colorAvg[c,:,t] = np.mean(temp1[:,t*n_tpnt:(t+1)*n_tpnt],1)           
            if f>0:
                X = np.concatenate((X,colorAvg), axis = 0)
            else:
                X = colorAvg
                
         # Set up label matrix. 
        Y = np.concatenate((np.ones((1,nChunk)),2*np.ones((1,nChunk)),3*np.ones((1,nChunk)),
                                    4*np.ones((1,nChunk)),5*np.ones((1,nChunk))),axis = 1)                 
        Y = np.int32(Y[0])
        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X,Y,'Colors',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 5)
        del X,Y
    
    #%%####------------------------- Values-2 -----------------------------------------
    if condition == 'Value-2' or condition == 'All':
        # Define basic condition information
        newValues = [0]*5
        newValues[0] = [-100,-70] # [-100,-70,-50]
        newValues[1] = [-30,-20]# [-40,-30,-20]
        newValues[2] = [0,0,0]
        newValues[3] = [20,30]#[20,30,40]
        newValues[4] = [70,100] ##[50,70,100]
        # newValues[5] = [40,50]
        # newValues[6] = [70,100]

        
        plt.figure(figNum)
        figNum+=1
        chan = epochs.ch_names.index('Pz')
        ttime = np.linspace(times[0],times[1],AllEpochs.shape[2])
        valList = []

        

        Y = np.empty((0))
        for v in range(len(newValues)):
            temp = AllEpochs[(behavData.Value == newValues[v][0])|(behavData.Value == newValues[v][1]) ,:,:]
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
            if v>0:
                X = np.concatenate((X,valueAvg), axis = 0)
            else:
                X = valueAvg

        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X,Y,'Value-2',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 5)

        del X,Y
        
    if condition == 'Value-3' or condition == 'All':
        # Define basic condition information
        newValues = [0]*5
        newValues[0] = [-100,-70,-50] # [-100,-70,-50]
        newValues[1] = [-40,-30,-20]# [-40,-30,-20]
        newValues[2] = [0,0,0]
        newValues[3] = [20,30,40]#[20,30,40]
        newValues[4] = [50,70,100] ##[50,70,100]
        # newValues[5] = [40,50]
        # newValues[6] = [70,100]

        
        plt.figure(figNum)
        figNum+=1
        chan = epochs.ch_names.index('Pz')
        ttime = np.linspace(times[0],times[1],AllEpochs.shape[2])
        valList = []


        Y = np.empty((0))
        for v in range(len(newValues)):
            temp = AllEpochs[(behavData.Value == newValues[v][0])|(behavData.Value == newValues[v][1])|(behavData.Value == newValues[v][2])  ,:,:]
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
            if v>0:
                X = np.concatenate((X,valueAvg), axis = 0)
            else:
                X = valueAvg

        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X,Y,'Value-3',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 5)

        del X,Y 

     #%%####------------------------- Values binary ----------------------------------
    if condition == 'Value_binary' or condition == 'All':
        # Define basic condition information
        newValues = [0]*2
        newValues[0] = [-100, -70, -50, -40, -30, -20] # [-100,-70,-50]
        newValues[1] = [100, 70, 50, 40, 30, 20]# [-40,-30,-20]


        
        plt.figure(figNum)
        figNum+=1
        chan = epochs.ch_names.index('Pz')
        ttime = np.linspace(times[0],times[1],AllEpochs.shape[2])
        valList = []

        

        Y = np.empty((0))
        for v in range(len(newValues)):
            temp = AllEpochs[(behavData.Value == newValues[v][0])|(behavData.Value == newValues[v][1]) |(behavData.Value == newValues[v][2])|(behavData.Value == newValues[v][3])|(behavData.Value == newValues[v][4])|(behavData.Value == newValues[v][5]),:,:]            #average trials
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
            if v>0:
                X = np.concatenate((X,valueAvg), axis = 0)
            else:
                X = valueAvg

        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X,Y,'Value_binary',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 5)

        del X,Y

      #%%####------------------------- Values binary ----------------------------------
    if condition == 'Value_binary_v2' or condition == 'All':
        # Define basic condition information
        newValues = [0]*3
        newValues[0] = [-100, -70, -50, -40, -30, -20] # [-100,-70,-50]
        newValues[1] = [100, 70, 50, 40, 30, 20]# [-40,-30,-20]
        newValues[2] = [0,0,0,0,0,0]# [-40,-30,-20]



        
        plt.figure(figNum)
        figNum+=1
        chan = epochs.ch_names.index('Pz')
        ttime = np.linspace(times[0],times[1],AllEpochs.shape[2])
        valList = []


        Y = np.empty((0))
        for v in range(len(newValues)):
            temp = AllEpochs[(behavData.Value == newValues[v][0])|(behavData.Value == newValues[v][1]) |(behavData.Value == newValues[v][2])|(behavData.Value == newValues[v][3])|(behavData.Value == newValues[v][4])|(behavData.Value == newValues[v][5]),:,:]
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
            if v>0:
                X = np.concatenate((X,valueAvg), axis = 0)
            else:
                X = valueAvg

        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X,Y,'Value_binary_v2',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 5)

        del X,Y
        
     #%%####------------------------- Values binary ----------------------------------
    if condition == 'Value_absMag' or condition == 'All':
        # Define basic condition information
        newValues = [0]*2
        newValues[0] = [-100, -70, -50, 50, 70, 100] # [-100,-70,-50]
        newValues[1] = [-40, -30, -20, 20, 30, 40]# [-40,-30,-20]


        
        plt.figure(figNum)
        figNum+=1
        chan = epochs.ch_names.index('Pz')
        ttime = np.linspace(times[0],times[1],AllEpochs.shape[2])
        valList = []



        Y = np.empty((0))
        for v in range(len(newValues)):
            temp = AllEpochs[(behavData.Value == newValues[v][0])|(behavData.Value == newValues[v][1]) |(behavData.Value == newValues[v][2])|(behavData.Value == newValues[v][3])|(behavData.Value == newValues[v][4])|(behavData.Value == newValues[v][5]),:,:]            #average trials
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
            if v>0:
                X = np.concatenate((X,valueAvg), axis = 0)
            else:
                X = valueAvg
        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X,Y,'Value_absMag',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 5)

        del X,Y

    if condition == 'Value_absMag_v2' or condition == 'All':
        # Define basic condition information
        newValues = [0]*3
        newValues[0] = [-100, -70, -50, 50, 70, 100] # [-100,-70,-50]
        newValues[1] = [-40, -30, -20, 20, 30, 40]# [-40,-30,-20]
        newValues[2] = [0, 0, 0, 0, 0, 0]# [-40,-30,-20]


        
        plt.figure(figNum)
        figNum+=1
        chan = epochs.ch_names.index('Pz')
        ttime = np.linspace(times[0],times[1],AllEpochs.shape[2])
        valList = []



        Y = np.empty((0))
        for v in range(len(newValues)):
            temp = AllEpochs[(behavData.Value == newValues[v][0])|(behavData.Value == newValues[v][1]) |(behavData.Value == newValues[v][2])|(behavData.Value == newValues[v][3])|(behavData.Value == newValues[v][4])|(behavData.Value == newValues[v][5]),:,:]            #average trials
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
            if v>0:
                X = np.concatenate((X,valueAvg), axis = 0)
            else:
                X = valueAvg
        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X,Y,'Value_absMag_v2',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 5)

        del X,Y

        
    #%%####------------------------- Face Values -----------------------------------------
    if condition == 'FaceValues' or condition == 'All':
        # Define basic condition information
        values = np.unique(behavData.FaceValue)
        plt.figure(figNum)
        figNum+=1
        chan = epochs.ch_names.index('Pz')
        ttime = np.linspace(times[0],times[1],AllEpochs.shape[2])
        valList = []
        

        
        # Set up label matrix (Y) while averaging together epochs for X.       
        Y = np.empty((0))
        for v in range(len(values)):
            temp = AllEpochs[behavData.Trial[behavData.FaceValue == values[v]]-1,:,:] #-1 to start trial index from 0
            #average trials
            n_avg = 2
            nChunk  = int(temp.shape[0]/n_avg)
            if timebin == 0:
                valueAvg = np.zeros((nChunk,temp.shape[1],temp.shape[2]))
                for c in range(nChunk):
                    valueAvg[c,:,:] = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                    Y = np.append(Y,values[v])
            else:
                nBin   = int((times[1]-times[0])*1000/timebin) # number of bins
                n_tpnt = int(timebin/(1000/epochs.info['sfreq']))
                valueAvg= np.zeros((nChunk,temp.shape[1],nBin))
                for c in range(nChunk):
                    temp1 = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                    for t in range(nBin):
                        valueAvg[c,:,t] = np.mean(temp1[:,t*n_tpnt:(t+1)*n_tpnt],1)    
                    Y = np.append(Y,values[v])
            if v>0:
                X = np.concatenate((X,valueAvg), axis = 0)
            else:
                X = valueAvg
        
        figNum = getandsaveResults(subj,X,Y,'FaceValues',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 5)
        del X,Y    
    #%%####------------------------- Color Values -----------------------------------------
    if condition == 'ColorValues' or condition == 'All':
        # Define basic condition information
        values = np.unique(behavData.ColorValue)
        #create averages based on trial types
        chan = epochs.ch_names.index('Pz')
        ttime = np.linspace(times[0],times[1],AllEpochs.shape[2])
        valList = []
        plt.figure(figNum)
        figNum+=1

        # Set up label matrix (Y) while averaging together epochs for X.       
        Y = np.empty((0))
        for v in range(len(values)):
            temp = AllEpochs[behavData.Trial[behavData.ColorValue == values[v]]-1,:,:] #-1 to start trial index from 0
            #average trials
            n_avg = 2
            nChunk  = int(temp.shape[0]/n_avg)
            if timebin == 0:
                valueAvg = np.zeros((nChunk,temp.shape[1],temp.shape[2]))
                for c in range(nChunk):
                    valueAvg[c,:,:] = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                    Y = np.append(Y,values[v])
            else:
                nBin   = int((times[1]-times[0])*1000/timebin) # number of bins
                n_tpnt = int(timebin/(1000/epochs.info['sfreq']))
                valueAvg= np.zeros((nChunk,temp.shape[1],nBin))
                for c in range(nChunk):
                    temp1 = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                    for t in range(nBin):
                        valueAvg[c,:,t] = np.mean(temp1[:,t*n_tpnt:(t+1)*n_tpnt],1)    
                    Y = np.append(Y,values[v])
            if v>0:
                X = np.concatenate((X,valueAvg), axis = 0)
            else:
                X = valueAvg
        
        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X,Y,'ColorValues',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 5)

        del X,Y       
    
    #%%####------------------------- Response -----------------------------------------
    if condition == 'Response' or condition == 'All':
        # Define basic condition information
        resps = [0,1]
        plt.figure(figNum)
        figNum+=1
        chan = epochs.ch_names.index('Fz')
        ttime = np.linspace(times[0],times[1],AllEpochs.shape[2])
        respList = ['Reject','Accept']
        

        # Set up label matrix (Y) while averaging together epochs for X.       
        Y = np.empty((0))
        for r in range(len(resps)):
            temp = AllEpochs[behavData.Resp == resps[r] ,:,:]
            print(temp.shape[0])
            #average trials
            n_avg = 2
            nChunk  = int(temp.shape[0]/n_avg)
            if timebin == 0:
                respAvg = np.zeros((nChunk,temp.shape[1],temp.shape[2]))
                for c in range(nChunk):
                    respAvg[c,:,:] = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                    Y = np.append(Y,resps[r])
            else:
                nBin   = int((times[1]-times[0])*1000/timebin) # number of bins
                n_tpnt = int(timebin/(1000/epochs.info['sfreq']))
                respAvg= np.zeros((nChunk,temp.shape[1],nBin))
                for c in range(nChunk):
                    temp1 = np.mean(temp[c*n_avg:(c+1)*n_avg,:,:],0)
                    for t in range(nBin):
                        respAvg[c,:,t] = np.mean(temp1[:,t*n_tpnt:(t+1)*n_tpnt],1)    
                    Y = np.append(Y,resps[r])
            if r>0:
                X = np.concatenate((X,respAvg), axis = 0)
            else:
                X = respAvg
        
        # ### Running discrimination analysis
        figNum = getandsaveResults(subj,X,Y,'Response-0',event,times,timebin, figNum,n_permutation=n_permutation,n_jobs=n_jobs,jitCorrect=jitCorrect,n_pseudo = 5)
        del X,Y   
#%%####-------------------------  -----------------------------------------

def find_nearest_idx(array, value):
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
