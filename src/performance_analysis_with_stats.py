import partitura
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
from basismixer.performance_codec import PerformanceCodec, get_performance_codec, to_matched_score
from basismixer.performance_codec import tempo_by_average, tempo_by_derivative 

def standard_analysis(
        attr, stat, note_array, performance_array, duration, forward_step_lim, 
        step, NUMBER_OF_WINDOWS=1, number_of_stats=1, is_BM=False
        ):
    '''
    A generalized definition for match score analysis
    
    Parameters:
    -----------
    attr : function
        One of the above functions, i.e. expressiveness_of_segment or tempo_of_segment.
    stat : function
        A statistical measure, i.e. mean, std or variance. 
    note_array : structed array
        The note_array from the partitura analysis
    performance_array : structured array
        The note_array from the basis mixer analysis
    duration : float
        The duration of a piece in seconds
    forward_spet_lim : int
        The maximum spread of the note array indices we can search for every window
    step : float
        The step for the analysis window.
    NUMBER_OF_WINDOWS : int 
        How many windows per step (equal to the length of the attr output vector).
    number_of_stats : itn
    
    is_BM : bool
    
    Returns:
    --------
    X : array
        An array of size len(duration of analysis) N x NUMBER_OF_WINDOWS x NUMBER_OF_WINDOWS.
    '''
    # normalize duration
    duration = duration / step
    step_unit = 1
    dim = int(round((duration - (NUMBER_OF_WINDOWS * step_unit)) / step_unit) + 1)
    # Experimenting with array resolution
    if NUMBER_OF_WINDOWS == 1:
        X = np.zeros((dim, number_of_stats))
    else :
        X = np.zeros((dim, NUMBER_OF_WINDOWS, number_of_stats))
    index = 0
    for i in range(1, dim - 1, step_unit):
        fix_start = i * step
        if NUMBER_OF_WINDOWS == 1:
            x = list()
            ind_list = list()
            if len(note_array[index:]) > forward_step_lim:
                look_in = forward_step_lim + index
            else :
                look_in = len(note_array)
            for ind, note in enumerate(note_array[index:look_in]):
                note_start = note[0] # onset
                note_end = note[0] + note[1] #onset + duration
                fix_end = fix_start + step
                # check if note is in window
                if (fix_start <= note_start <= fix_end) or (note_start <= fix_start and note_end >= fix_start):
                    ind_list.append(ind)
                    if is_BM:
                        x.append(performance_array[ind]) # Expressive Parameters
                    else:    
                        x.append(note)
            if x != []:
                X[i - 1] = attr(x, stat)
        else:
            for j in range(1, NUMBER_OF_WINDOWS + 1):
                x = list()
                ind_list = list()
                if len(note_array[index:]) > forward_step_lim*j:
                    look_in = forward_step_lim * j + index
                else:
                    look_in = len(note_array)
                for ind, note in enumerate(note_array[index:look_in]):
                    note_start = note[0] #onset
                    note_end = note[0] + note[1] #onset + duration
                    fix_end = fix_start + (j * step)
                    # check if note is in window
                    if (fix_start <= note_start <= fix_end) or (note_start <= fix_start and note_end >= fix_start):
                        ind_list.append(ind)
                        if is_BM:
                            x.append(performance_array[ind]) # Expressive Parameters
                        else:    
                            x.append(note)
                if x != []:
                    X[i - 1][j - 1] = attr(x, stat)
        if ind_list != []:
            index += min(ind_list)
    return X

def tempo_of_segment(x, stat):
    """
    Tempo feature extraction per windows.
    
    Parameters:
    -----------
    x : list(tuples)
        A segment of the note_array
    stat : function
        a statistical function that outputs
    
    Returns:
    --------
    stat(y1), stat(y2) : tuple(float)
        The statistics of Vectors y1 and y2
    
    """   
    
    score_onsets, score_durations, _, performed_onsets, performed_durations, _ = list(zip(*x))   
    y1 = tempo_by_average(score_onsets, performed_onsets, score_durations, performed_durations)[0]
    y2 = tempo_by_derivative(score_onsets, performed_onsets, score_durations, performed_durations)[0]
    return [stat(y1), stat(y2)]

def expressiveness_of_segment(x, stat):
    """
    Tempo feature extraction per windows.
    
    Parameters:
    -----------
    x : list(tuples)
        A segment of the note_array
    stat : function
        a statistical function that outputs
    
    Returns:
    --------
    stat(beat_period), stat(velocity), stat(timing) : tuple(float)
        The statistics of the expressive vectors
    
    """   
    
    beat_period, velocity, timing, articulation_log = list(zip(*x))   
    return [stat(beat_period), stat(velocity), stat(timing), stat(articulation)]



def perform_statistical_analysis(match_fn, attr, stat, step, number_of_windows):
    """Perform the analysis.
    
    Parameters
    ----------
    file : str
        the file name + extension.
    attr : function
        One of the above functions, i.e. expressiveness_of_segment or tempo_of_segment.
    stat : function
        A statistical measure, i.e. mean, std or variance. 
    number_of_windows : 
        How many windows per step (equal to the length of the attr output vector).
        
    Returns
    -------
    X : np.array
        An array of size len(duration of analysis) N x number_of_windows x number_of_windows.
    performance_SSM : np.array
        The SSM of X using dot product and robust PCA.
    """
    ppart, alignment, spart = partitura.load_match(match_fn, create_part=True)
    note_array, _ = to_matched_score(spart, ppart, alignment)
    parameter_names = ['beat_period', 'velocity', 'timing', 'articulation_log']
    pc = get_performance_codec(parameter_names)
    performance_array, _ = pc.encode(spart, ppart, alignment)
    durations = [n[1] for n in note_array if n[1]!=0]
    min_duration = min(durations)
    max_duration = max(durations)
    max_polyphony = max([len(list(item[1])) for item in itertools.groupby(note_array, key=lambda x: x[0])])
    forward_step_lim = int(max_duration / min_duration + max_polyphony)
    note_array, performance_array = zip(*sorted(zip(note_array, performance_array), key=lambda note: note[0][0]))
    duration = note_array[-1][0] + max_duration - step
    
    if attr[1] == "tempo":
        is_BM = False
        dummy = note_array[-1]
    else:
        is_BM = True
        dummy = performance_array[-1]
    
    attr = attr[0]
    stat = stat[0]
    
    number_of_stats = len(attr([dummy], stat))
    X = standard_analysis(attr, stat, note_array, performance_array, duration, forward_step_lim, step, number_of_windows, number_of_stats, is_BM)
    return X




