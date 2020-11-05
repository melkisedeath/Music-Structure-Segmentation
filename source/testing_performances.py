import partitura
import numpy as np
import itertools
from statistics import mean
import sys, os
from chords import chord_to_intervalVector
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from basismixer.performance_codec import to_matched_score
from PIL import Image


MATCHSCORE_DIR = os.path.dirname(os.getcwd()) + "\\samples\\data\\match\\"
ARTIFACT_DIR = os.path.dirname(os.getcwd()) + "\\artifacts\\images\\"
NUMBER_OF_WINDOWS = 6
SIZE_OF_INTVEC = 6

MIDI_VEL_SIZE = 128

parts = {
    "a1" : (0, 20),
    "a2" : (20, 32),
    "g1" : (32, 58),
    "b" : (58, 84),
    "b_ext" : (84, 92),
    "g2" : (92, 121),
    "a1'" : (224,244),
    "a1'_ext" : (244, 251),
    "g1'" : (250, 276),
    "b'" : (276, 303),
    "b'_ext" : (303, 311),
    "g2'" : (311, 336)
}

sections = {
    "Exp1" : (0,121),
    "Dev" : (121,224),
    "Exp2" : (224, 336)
}

step = 0.5
time_sign = 2
pick_up = 1
overlap = NUMBER_OF_WINDOWS*step
measure = time_sign/step

def structure_analysis(note_array, duration, step=1):
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
#     X = np.zeros((dim, NUMBER_OF_WINDOWS, SIZE_OF_INTVEC), dtype=np.uint8)
    for i in range(1 , dim-1, step_unit):
        fix_start = i*step
        for j in range(1, NUMBER_OF_WINDOWS + 1):
            x = list()
            for note in note_array:
                note_start = note[0] #onset
                note_end = note[0] + note[1] #onset + duration
                fix_end = fix_start + (j*step)
                # check if note is in window
                if  fix_start <= note_start <= fix_end :
                    x.append(note[2]) # pitch
                elif note_start <= fix_start and note_end >= fix_start:
                    x.append(note[2]) # pitch
            if x != []:
                interval = chord_to_intervalVector(x)
                X[i-1][j-1] = interval
    return X


def dynamics_analysis(note_array, duration, step=1):
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
    duration = duration/step
    step_unit = 1
    dim = int(round((duration-(NUMBER_OF_WINDOWS*step_unit))/step_unit)+1)
#     X = np.zeros((dim, 128))
    X = np.zeros((dim, MIDI_VEL_SIZE), dtype=np.uint)
    for i in range(1 , dim-1, step_unit):
        fix_start = i*step
        for note in note_array:
            note_start = note[0] #onset
            note_end = note[0] + note[1] #onset + duration
            fix_end = fix_start + NUMBER_OF_WINDOWS*step 
            if  fix_start <= note_start <= fix_end :
                X[i][note[2]] += note[-1]
            elif note_start <= fix_start and note_end >= fix_start:
                X[i][note[2]] += note[-1]
    return X


for file in os.listdir(MATCHSCORE_DIR):
	if file.endswith("match"):
		ppart, alignment, spart = partitura.load_match(MATCHSCORE_DIR + file, create_part=True, old_part_generator=True)
		note_array, snote_ids = to_matched_score(spart, ppart, alignment)
		duration = note_array[-1][0]
		X_full = structure_analysis(note_array, duration, step)
		X = np.reshape(X_full, (X_full.shape[0], X_full.shape[1]*X_full.shape[2]))
		Y = dynamics_analysis(note_array, duration, step)
		pca = PCA(n_components=3, svd_solver='auto')
		A = pca.fit_transform(X)
		B = pca.fit_transform(Y)


		im = Image.fromarray(A@A.T)
		im.convert('RGB').save(ARTIFACT_DIR + "Intervalic_" + os.path.splitext(file)[0] + ".png")
		im = Image.fromarray(B@B.T)
		im.convert('RGB').save(ARTIFACT_DIR + "Dynamics_" + os.path.splitext(file)[0] + ".png")