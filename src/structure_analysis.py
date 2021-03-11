from tensorly.decomposition import robust_pca
from sklearn.metrics.pairwise import cosine_similarity
from chords import chord_to_intervalVector
from basismixer.performance_codec import PerformanceCodec, get_performance_codec, to_matched_score
from sklearn.utils.random import sample_without_replacement
import partitura
import itertools
import numpy as np


def intervalic_analysis(note_array, step=1):
    '''
    Does the intervalic analysis of a piece.
    
    Parameters:
    -----------
    note_array : array(tuples)
        The note_array from the partitura analysis
    step : float
        The step for the analysis window.
    
    Returns:
    --------
    X : array(int)
        An array of the piece analysis which outputs interval vectors.
        The size of the array is len(duration of analysis) x 6 x 6.
    '''    
    # standard forward lim
    durations = [n['duration'] for n in note_array if n['duration']!=0]
    min_duration = min(durations)
    max_duration = max(durations)
    max_polyphony = max([len(list(item[1])) for item in itertools.groupby(note_array, key=lambda x: x[0])])
    forward_step_lim = int(max_duration / min_duration + max_polyphony)
    duration = note_array[-1]['onset'] + max_duration - step

    NUMBER_OF_WINDOWS = 6
    SIZE_OF_INTVEC = 6

    # normalize duration
    duration = duration/step
    step_unit = 1
    dim = int(round((duration-(NUMBER_OF_WINDOWS*step_unit))/step_unit)+1)
    # Experimenting with array resolution
    X = np.zeros((dim, NUMBER_OF_WINDOWS, SIZE_OF_INTVEC))
#     X = np.zeros((dim, NUMBER_OF_WINDOWS, SIZE_OF_INTVEC), dtype=np.uint8)
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



def analysis_to_SSM(note_array, step=1, return_analysis_matrix=False):
    """
    Perform the intervalic analysis and output structure SSM.


    Parameters
    ----------
    note_array : structured array
        The note array ordered on onsets
    duration : int
        the duration of the piece / onset of last note of the array
    step : float (optional)
        the transition step for analysis windows, usually 1 beat. 
        Default value = 1 beat depending on the time signature.

    Returns:
    --------
    S : np.array
        The self_similarity matrix, i.e. cosine similarity of the PCA of X, 
        where X is the intervalic analysis of the note_array.
    X : np.array (optional)
        The intervalic analysis of the note_array.

    Examples
    --------
    import partitura
    from basismixer.performance_codec import to_matched_score
    import matplotlib.pyplot as plt

    match_fn = "Mozart_K331_1st-mov_p08.match" 
    ppart, alignment, spart = partitura.load_match(match_fn, create_part=True)
    note_array, _ = to_matched_score(spart, ppart, alignment)
    S = analysis_to_SSM(note_array)

    plt.imshow(S, cmap="gray")
    """

    X = intervalic_analysis(note_array, step)
    low_rank_part, sparse_part = robust_pca(X, reg_E=0.04, learning_rate=1.2, n_iter_max=20)
    S = cosine_similarity(low_rank_part.reshape(X.shape[0], X.shape[1]*X.shape[2]))
    if return_analysis_matrix:
        return S, X
    else:
        return S



def analysis_with_sampling(note_array, performance_array, step, window_size, samples_per_window):
    '''
    A definition for match score analysis 
    
    Parameters
    ----------
    note_array : array(tuples)
        The note_array from the partitura analysis
    performance_array : array(tuples)
        The performance_array from the basis mixer analysis
    step : float
        The step for the analysis window.
    window_size : int
        The window size to draw samples from, usually a bar measure size.
    samples_per_window : int
        the number of samples to draw per window
    
    Returns
    -------
    X : array
        An array of size len(duration of analysis) N x 6 x 6.
        
    '''
    # standard forward lim
    durations = [n['duration'] for n in note_array if n['duration']!=0]
    min_duration = min(durations)
    max_duration = max(durations)
    max_polyphony = max([len(list(item[1])) for item in itertools.groupby(note_array, key=lambda x: x[0])])
    forward_step_lim = int(max_duration / min_duration + max_polyphony)
    duration = note_array[-1]['onset'] + max_duration - step
    
    # normalize duration
    duration = duration / step
    step_unit = 1
    dim = int(round((duration - (window_size * step_unit)) / step_unit) + 1)
    # 4 is the attributes number
    X = np.zeros((dim - 1, 4, 4, samples_per_window))
    index = 0
    for window in range(1, dim - 1, step_unit):
        fix_start = window * step
        for window_index in range(4):
            ind_list = list()
            if len(note_array[index:]) > forward_step_lim*window_size:
                look_in = forward_step_lim * window_size + index
            else:
                look_in = len(note_array)
            for ind, note in enumerate(note_array[index:look_in]):
                note_start = note['onset'] #onset
                note_end = note['onset'] + note['duration'] #onset + duration
                fix_end = fix_start + (window_size * step)
                # check if note is in window
                if (fix_start <= note_start <= fix_end) or (note_start <= fix_start and note_end >= fix_start):
                    ind_list.append(ind)
            if ind_list != []:
                # sample from window
                sample_indices = sample_without_replacement(len(ind_list), samples_per_window)
                # compute expressive features from performance array
                performance = [performance_array[j + index] for j in sample_indices ]
                beat_period, velocity, timing, articulation_log = list(zip(*performance))
                # Add to output array
                X[window][window_index][0] = np.array(beat_period)
                X[window][window_index][1] = np.array(velocity)
                X[window][window_index][2] = np.array(timing)
                X[window][window_index][3] = np.array(articulation_log)
                # update index
                index += min(ind_list)
    return X

def performance_analysis(match_fn, step=1, window_size=8, samples_per_window=10):
    """
    Parameters
    ----------
    match_fn : string
        The path of a match file, i.e. MATCH/SCORE/DIRECTORY/Mozart_piece.match
    file_structure : dict
        A dictionary with the sections and step for the file
    window_size : int
        The window size for the analysis in number of steps, usually a bar.
    samples_per_window : int
        The number of notes to sample per window.
    
    Returns
    -------
    X : array
        The analysis of the file.
    
    
    Examples
    --------
        file = "Mozart_K331_1st-mov_p08.match"
        window_size = 6 
        samples_per_window = 10
        X = perform_analysis(file, file_structure, window_size, samples_per_window)
        
    """
    ppart, alignment, spart = partitura.load_match(match_fn, create_part=True)
    note_array, _ = to_matched_score(spart, ppart, alignment)
    parameter_names = ['beat_period', 'velocity', 'timing', 'articulation_log']
    pc = get_performance_codec(parameter_names)
    performance_array, snote_ids = pc.encode(spart, ppart, alignment)    
    note_array, performance_array, snote_ids = zip(*sorted(zip(note_array, performance_array, snote_ids), key=lambda note: note[0][0]))
    X = analysis_with_sampling(note_array, performance_array, step, window_size, samples_per_window)
    return X



