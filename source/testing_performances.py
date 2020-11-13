import partitura
import numpy as np
import itertools
from statistics import mean
import sys, os
from chords import chord_to_intervalVector
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from basismixer.performance_codec import to_matched_score
from tensorly.decomposition import robust_pca
import scipy.misc


MATCHSCORE_DIR = os.path.dirname(os.path.dirname(os.getcwd())) + "\\alignments\\match_4\\"
ARTIFACT_DIR = os.path.dirname(os.path.dirname(os.getcwd())) + "\\alignments\\ssm\\"
NUMBER_OF_WINDOWS = 6
NUMBER_OF_STATS = 6
SIZE_OF_INTVEC = 6
step = 0.5

def structure_analysis(note_array, duration, forward_step_lim, step=1):
    '''
    Does the intervalic analysis of a piece.
    
    Parameters:
    -----------
    note_array : array(tuples)
        The note_array from the partitura analysis
    
    duration : float
        The duration of a piece in seconds
    
    step : float
        The step for the analysis window.
    
    Returns:
    --------
    X : array(int)
        An array of the piece analysis which outputs interval vectors.
        The size of the array is len(duration of analysis) x 6 x 6.
    '''    
    # normalize duration
    duration = duration/step
    step_unit = 1
    dim = int(round((duration-(NUMBER_OF_WINDOWS*step_unit))/step_unit)+1)
    # Experimenting with array resolution
    X = np.zeros((dim, NUMBER_OF_WINDOWS, SIZE_OF_INTVEC))
    index = 0
    for i in range(1 , dim-1, step_unit):
        fix_start = i*step
        for j in range(1, NUMBER_OF_WINDOWS + 1):
            x = list()
            ind_list = list()
            if len(note_array[index:]) > forward_step_lim*j:
                look_in = forward_step_lim*j+index
            else :
                look_in = len(note_array)
            for ind, note in enumerate(note_array[index:look_in]):
                note_start = note[0] #onset
                note_end = note[0] + note[1] #onset + duration
                fix_end = fix_start + (j*step)                
                # check if note is in window
                if  fix_start <= note_start <= fix_end :
                    ind_list.append(ind)
                    x.append(note[2]) # pitch
                elif note_start <= fix_start and note_end >= fix_start:
                    ind_list.append(ind)
                    x.append(note[2]) # pitch
            if x != []:
                interval = chord_to_intervalVector(x)
                X[i-1][j-1] = interval
        if ind_list != [] : 
            index += min(ind_list)
    return X


def dynamics_analysis(note_array, duration, forward_step_lim, step=1 ):
    '''
    Does the dynamic analysis of a piece.
    
    Parameters:
    -----------
    note_array : array(tuples)
        The note_array from the partitura analysis
    
    duration : float
        The duration of a piece in seconds
    
    step : float
        The step for the analysis window.
    
    Returns:
    --------
    X : array(int)
        An array of size len(duration of analysis) x 128. 
        The piece analysis which outputs the sum of velocitys for all midi pitch for each window.
    '''
    # normalize duration
    duration = duration/step
    step_unit = 1
    dim = int(round((duration-(NUMBER_OF_WINDOWS*step_unit))/step_unit)+1)
    # Experimenting with array resolution
    X = np.zeros((dim, NUMBER_OF_WINDOWS, NUMBER_OF_STATS))
    index = 0
    for i in range(1 , dim-1, step_unit):
        fix_start = i*step
        for j in range(1, NUMBER_OF_WINDOWS + 1):
            x = list()
            ind_list = list()
            if len(note_array[index:]) > forward_step_lim*j:
                look_in = forward_step_lim*j+index
            else :
                look_in = len(note_array)
            for ind, note in enumerate(note_array[index:look_in]):
                note_start = note[0] #onset
                note_end = note[0] + note[1] #onset + duration
                fix_end = fix_start + (j*step)                
                # check if note is in window
                if  fix_start <= note_start <= fix_end :
                    ind_list.append(ind)
                    x.append(note[5]*note[1]) # Velocity
                elif note_start <= fix_start and note_end >= fix_start:
                    ind_list.append(ind)
                    x.append(note[5]*note[1]) # velocity
            if x != []:
                X[i-1][j-1] = [np.mean(x), np.var(x), np.std(x), min(x), max(x), max(x) - min(x)]
        if ind_list != [] : 
            index += min(ind_list)
    return X


for file in os.listdir(MATCHSCORE_DIR):
    if file.endswith("match"):
        ppart, alignment, spart = partitura.load_match(MATCHSCORE_DIR + file, create_part=True, old_part_generator=True)
        note_array, snote_ids = to_matched_score(spart, ppart, alignment)
        durations = [n[1] for n in note_array if n[1]!=0]
        min_duration = min(durations)
        max_duration = max(durations)
        max_polyphony = max([len(list(item[1])) for item in itertools.groupby(note_array, key=lambda x: x[0])])
        forward_step_lim = int(max_duration/min_duration + max_polyphony)
        duration = note_array[-1][0]
        note_array = sorted(note_array, key=lambda note: note[0])
        X = structure_analysis(note_array, duration, forward_step_lim, step)
        low_rank_part, sparse_part = robust_pca(X, reg_E=0.04, learning_rate=1.2, n_iter_max=20)
        structure_SSM = np.tensordot(low_rank_part, low_rank_part.T)
        Y = dynamics_analysis(note_array, duration, forward_step_lim, step)
        low_rank_part, sparse_part = robust_pca(Y, reg_E=0.04, learning_rate=1.2, n_iter_max=20)
        performance_SSM = np.tensordot(low_rank_part, low_rank_part.T)

        np.savetxt(ARTIFACT_DIR + "Structure_" + os.path.splitext(file)[0] + ".csv", structure_SSM, delimiter=',')
        np.savetxt(ARTIFACT_DIR + "Performance_" + os.path.splitext(file)[0] + ".csv", performance_SSM, delimiter=',')
