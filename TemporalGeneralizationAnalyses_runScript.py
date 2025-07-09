## This is the main script used to create the temporal generalization analyses
# The general procedure of this script is to take each subject and condition, develop a classifier, and test it on past/present time points.
# The subscripts go into more detail on each.

# Please note that these analyses require both the processed EEG data and behavioral csv files.

from  matplotlib import pyplot as plt

# The list of subjects used in the final analyses
subjects = [str(s) for s in [150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,166,167,168,169,170,171,173,174,176,177,178,180,181,182,183,185,186,188,189,190,191,192,193,194,195]]

# Import custom function.
import EEG_tempGen_Stim_guggenmos_reref as tempGenModule


# Define which epochs to test. Default is "Stimulus_long, which covers -200 to 2000ms from trial onset.
events   = ['Stimulus_long']


# Change these variables depending on your goals. Default is "All", but can also do "Faces", "Colors", "FaceValues", "ColorValues", 
#"Value-2", and several others.
conditions = ['All']

# Size of timebin to average by. 
timebin = 10

# Main Loop.
for s in subjects:
    for c,cond in enumerate(conditions):
        for e,event in enumerate(events):
            tempGenModule.runClassification(s,cond,timebin,event,n_permutation=20,n_jobs=-1,jitCorrect =None)
            print ('Subject: '+s+', Condition: '+cond+', Event: '+event+', complete ...\n')
            plt.close('all')
