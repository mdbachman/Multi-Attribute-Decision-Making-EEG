## This is the main script used to create the temporally-distinct classifiers, SVRs, and the data for the value-boosted perceptual analyses.
# The general procedure of this script is to take each subject and condition, and develop and test a  classifier.
# The subscripts go into more detail on each.

# Please note that these analyses require both the processed EEG data and behavioral csv files.

from  matplotlib import pyplot as plt

# The list of subjects used in the final analyses
subjects = [str(s) for s in [150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,166,167,168,169,170,171,173,174,176,177,178,180,181,182,183,185,186,188,189,190,191,192,193,194,195]]

# Define which epochs to test. Default is "Stimulus_long, which covers -200 to 2000ms from trial onset.
events   = ['Stimulus_long']

# Import custom functions and scripts.
# For the temporally-specific classifiers, with cond = "All" and timebin = 10
import TemporallySpecificAnalyses_SVC_mainFunctions as ClassificationModule # Used for the temporally-specific classifiers.
import TemporallySpecificAnalyses_SVC_mainFunctions_byBlock as ClassificationModule_halves # Used for the value-boosted analyses.
import TemporallySpecificAnalyses_SVR_mainFunctions as ClassificationModule_SVR # Used for support-vector regression analyses

#---------------------------------------------------------
# CHANGE VARIABLES HERE
#---------------------------------------------------------

# Change these variables depending on your goals. Default is "All", but can also do "Faces", "Colors", "FaceValues", "ColorValues", 
#"Value-2", and several others.
conditions = ['All']


#---------------------------------------------------------
# END OF CHANGE VARIABLES.
#---------------------------------------------------------

# Main Loop.
for s in subjects:
    for c,cond in enumerate(conditions):
        for e,event in enumerate(events):
            # 10ms bins for most analyses.
            timebin = 10
            ClassificationModule.runClassification(s,cond,timebin,event,n_permutation=20,n_jobs=-1,jitCorrect =None)
            ClassificationModule_SVR.runClassification(s,cond,timebin,event,n_permutation=20,n_jobs=-1,jitCorrect =None)
            
            # 20ms timebin only for the "halves" script, for the value-boosted analyses.
            timebin = 20
            ClassificationModule_halves.runClassification(s,cond,timebin,event,n_permutation=20,n_jobs=-1,jitCorrect =None)
            print ('Subject: '+s+', Condition: '+cond+', Event: '+event+', complete ...\n')
            plt.close('all')
            
       

