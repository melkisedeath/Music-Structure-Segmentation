import partitura
import numpy as np
import itertools
from statistics import mean
import sys, os
from chords import chord_to_intervalVector
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from basismixer.performance_codec import PerformanceCodec, get_performance_codec, to_matched_score
from basismixer.performance_codec import tempo_by_average, tempo_by_derivative 
from tensorly.decomposition import robust_pca
import scipy.misc

sys.path.insert(0, os.path.dirname(os.getcwd())+ '\\samples\\scripts\\')
import match_4x22


MATCHSCORE_DIR = os.path.dirname(os.path.dirname(os.getcwd())) + "\\alignments\\match_4\\"
ARTIFACT_DIR = os.path.dirname(os.path.dirname(os.getcwd())) + "\\alignments\\ssm\\"

def standard_analysis(
        attr, stat, note_array, performance_array, duration, forward_step_lim, 
        step, NUMBER_OF_WINDOWS=1, number_of_stats=1, is_BM=False
        ):
    '''
    A generalized definition for match score analysis
    
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
    X : array
        An array of size len(duration of analysis) N x 6 x 6.
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
        The statistics of Vectors y1 and
    
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
    stat(y1), stat(y2) : tuple(float)
        The statistics of Vectors y1 and
    
    """   
    
    beat_period, velocity, timing, articulation_log = list(zip(*x))   
    return [stat(beat_period), stat(velocity), stat(timing)]






def plot_structure(ssm, piece_structure, pick_up = 1):
    """
    Plot the structure based on some characteristics.


    """
    step = piece_structure["step"]
    measure = piece_structure["measure"] / step
    parts = piece_structure["structure"]
    overlap = step * 2 # set the overlap factor freely

    figure, ax = plt.subplots(1, figsize=(25, 25))
    for text, part in parts.items():
        x, y = part

        x = (x-1)*measure + step
        y = (y-1)*measure + overlap
        p = patches.Rectangle((x,x),y-x,y-x, edgecolor='r', facecolor="none")
        # ax.annotate(text, (x + (y-x)/2, x + (y-x)/2), color='w', weight='bold',
        #             fontsize=14, ha='center', va='center')
        ax.add_patch(p)

    ax.imshow(ssm, cmap='gray_r')
    plt.show()



# step = match_4x22.mozart_K331["step"]

# for file in os.listdir(MATCHSCORE_DIR):
#     if file.endswith("match") and match_4x22.mozart_K331["name"] in file :
#         match_fn = MATCHSCORE_DIR + file
#         ppart, alignment, spart = partitura.load_match(match_fn, create_part=True)
#         note_array, snote_ids = to_matched_score(spart, ppart, alignment)
#         durations = [n[1] for n in note_array if n[1]!=0]
#         min_duration = min(durations)
#         max_duration = max(durations)
#         max_polyphony = max([len(list(item[1])) for item in itertools.groupby(note_array, key=lambda x: x[0])])
#         forward_step_lim = int(max_duration/min_duration + max_polyphony)
#         note_array = sorted(note_array, key=lambda note: note[0])
#         duration = note_array[-1][0] + max_duration - step

#         tempo_analysis = standard_analysis(lambda x : tempo_of_segment(x),
#                                    note_array, note_array[-1][0], int(max_duration/min_duration + max_polyphony), 0.5)
#         tempo_derivative_analysis = standard_analysis(lambda x : tempo_derivative_of_segment(x),
#                                    note_array, note_array[-1][0], int(max_duration/min_duration + max_polyphony), 0.5)
#         X = np.array(list(map(list, zip(tempo_analysis, tempo_derivative_analysis))))
#         low_rank_part, sparse_part = robust_pca(X, reg_E=0.04, learning_rate=1.2, n_iter_max=20)
#         if len(low_rank_part.shape) > 2 :
#             performance_SSM = np.tensordot(low_rank_part, low_rank_part.T)
#         else : 
#             performance_SSM = np.dot(low_rank_part, low_rank_part.T)
#         print("Plotting structure ... ")
#         plot_structure(performance_SSM, match_4x22.mozart_K331)

        # np.savetxt(ARTIFACT_DIR + "Performance_" + os.path.splitext(file)[0] + ".csv", performance_SSM, delimiter=',')
        






file = "Mozart_K331_1st-mov_p08.match"

def perform_multi_analysis(file):
    # Declare the attributes and statistics
    attributes = [(lambda x, stat: tempo_of_segment(x, stat), "tempo"),
                  (lambda x, stat: expressiveness_of_segment(x, stat), "expressive")
                 ]
    statistic_methods = [(lambda y : np.mean(y), "mean"),
                         (lambda y : np.std(y), "std"),
                         (lambda y : np.var(y), "var")
                        ]

    step = match_4x22.mozart_K331["step"]
    match_fn = MATCHSCORE_DIR + file
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



    for (attr, attr_name), (stat, stat_name) in itertools.product(attributes, statistic_methods):
        # run the attr with dummy argument [0] to see how many variables it returns
        # and fix the number of windows
        
        if attr_name == "tempo":
            is_BM = False
            dummy = note_array[-1]
        else:
            is_BM = True
            dummy = performance_array[-1]
        number_of_stats = len(attr([dummy], stat))
        for number_of_windows in [1, number_of_stats]:
            X = standard_analysis(attr, stat, note_array, performance_array, duration, forward_step_lim, step, number_of_windows, number_of_stats, is_BM)
            low_rank_part, sparse_part = robust_pca(X, reg_E=0.04, learning_rate=1.2, n_iter_max=20)
            if len(low_rank_part.shape) > 2:
                performance_SSM = np.tensordot(low_rank_part, low_rank_part.T)
            else:
                performance_SSM = np.dot(low_rank_part, low_rank_part.T)
            
            text = attr_name + "_" + stat_name + "_" + str(number_of_windows) + "_"
            np.savetxt(ARTIFACT_DIR + "Test_01\\" + text + os.path.splitext(file)[0] + ".csv", performance_SSM, delimiter=',')




def plot_all_stats():

    step = match_4x22.mozart_K331["step"]
    measure = match_4x22.mozart_K331["measure"] / step
    parts = match_4x22.mozart_K331["structure"]
    overlap = step * 2 # set the overlap factor freely

    cmap = "gray"
    f, axarr = plt.subplots(4,3, figsize=(25, 25))

    l = []
    for filename in os.listdir(ARTIFACT_DIR + "Test_01\\"):
        if filename.endswith(".csv"):
            l.append(ARTIFACT_DIR + "Test_01\\" + filename)
    k=0
    for i in range(4):
        for j in range(3):
            X = np.loadtxt(l[k], delimiter=',')
            axarr[(i, j)].imshow(X, cmap=cmap)
            axarr[(i, j)].title.set_text(os.path.splitext(os.path.basename(l[k]))[0])

            for text, part in parts.items():
                x, y = part
                x = (x -1 ) * measure + step
                y = (y - 1) * measure + overlap
                p = patches.Rectangle((x,x),y-x,y-x, edgecolor='r', facecolor="none")
                axarr[(i, j)].add_patch(p)
            k += 1
    plt.tight_layout()
    plt.show()



