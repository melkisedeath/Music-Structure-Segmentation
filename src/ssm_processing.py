import numpy as np
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny

def filter_diag_sm(S, L=30):
    """Path smoothing of similarity matrix by forward filtering along main diagonal

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        S: Similarity matrix (SM)
        L: Length of filter

    Returns:
        S_L: Smoothed SM
    """
    N = S.shape[0]
    M = S.shape[1]
    S_L = np.zeros((N, M))
    S_extend_L = np.zeros((N + L, M + L))
    S_extend_L[0:N, 0:M] = S
    for pos in range(0, L):
        S_L = S_L + S_extend_L[pos:(N + pos), pos:(M + pos)]
    S_L = S_L / L
    return S_L


def filter_diag_mult_sm(S, L=1, tempo_rel_set=np.asarray([1]), direction=0):
    """Path smoothing of similarity matrix by filtering in forward or backward direction
    along various directions around main diagonal
    Note: Directions are simulated by resampling one axis using relative tempo values

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        S: Self-similarity matrix (SSM)
        L: Length of filter
        tempo_rel_set: Set of relative tempo values
        direction: Direction of smoothing (0: forward; 1: backward)

    Returns:
        S_L_final: Smoothed SM
    """
    N = S.shape[0]
    M = S.shape[1]
    num = len(tempo_rel_set)
    S_L_final = np.zeros((N, M))

    for s in range(0, num):
        M_ceil = int(np.ceil(M/tempo_rel_set[s]))
        resample = np.multiply(np.divide(np.arange(1, M_ceil+1), M_ceil), M)
        np.around(resample, 0, resample)
        resample = resample - 1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)
        S_resample = S[:, index_resample]

        S_L = np.zeros((N, M_ceil))
        S_extend_L = np.zeros((N + L, M_ceil + L))

        # Forward direction
        if direction == 0:
            S_extend_L[0:N, 0:M_ceil] = S_resample
            for pos in range(0, L):
                S_L = S_L + S_extend_L[pos:(N + pos), pos:(M_ceil + pos)]

        # Backward direction
        if direction == 1:
            S_extend_L[L:(N+L), L:(M_ceil+L)] = S_resample
            for pos in range(0, L):
                S_L = S_L + S_extend_L[(L-pos):(N + L - pos), (L-pos):(M_ceil + L - pos)]

        S_L = S_L / L
        resample = np.multiply(np.divide(np.arange(1, M+1), M), M_ceil)
        np.around(resample, 0, resample)
        resample = resample - 1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)

        S_resample_inv = S_L[:, index_resample]
        S_L_final = np.maximum(S_L_final, S_resample_inv)

    return S_L_final



def forw_back_smoothing(S, L=20, tempo_rel_set=np.asarray([1])):
    S_forward = filter_diag_mult_sm(S, L, tempo_rel_set, direction=0)
    S_backward = filter_diag_mult_sm(S, L, tempo_rel_set, direction=1)
    S_final = np.maximum(S_forward, S_backward)
    return S_final

def threshold_matrix(S, thresh, strategy='absolute', scale=False, penalty=0, binarize=False):
    """Treshold matrix in a relative fashion

    Notebook: C4/C4/C4S2_SSM-Thresholding.ipynb

    Args:
        S: Input matrix
        thresh: Treshold (meaning depends on strategy)
        strategy: Thresholding strategy ('absolute', 'relative', 'local')
        scale: If scale=True, then scaling of positive values to range [0,1]
        penalty: Set values below treshold to value specified
        binarize: Binarizes final matrix (positive: 1; otherwise: 0)
        Note: Binarization is applied last (overriding other settings)


    Returns:
        S_thresh: Thresholded matrix
    """
    if np.min(S) < 0:
        raise Exception('All entries of the input matrix must be nonnegative')

    S_thresh = np.copy(S)
    N, M = S.shape
    num_cells = N * M

    if strategy == 'absolute':
        thresh_abs = thresh
        S_thresh[S_thresh < thresh] = 0

    if strategy == 'relative':
        thresh_rel = thresh
        num_cells_below_thresh = int(np.round(S_thresh.size*(1-thresh_rel)))
        if num_cells_below_thresh < num_cells:
            values_sorted = np.sort(S_thresh.flatten('F'))
            thresh_abs = values_sorted[num_cells_below_thresh]
            S_thresh[S_thresh < thresh_abs] = 0
        else:
            S_thresh = np.zeros([N, M])

    if scale:
        cell_val_zero = np.where(S_thresh == 0)
        cell_val_pos = np.where(S_thresh > 0)
        if len(cell_val_pos[0]) == 0:
            min_value = 0
        else:
            min_value = np.min(S_thresh[cell_val_pos])
        max_value = np.max(S_thresh)
        # print('min_value = ', min_value, ', max_value = ', max_value)
        if max_value > min_value:
            S_thresh = np.divide((S_thresh - min_value), (max_value - min_value))
            if len(cell_val_zero[0]) > 0:
                S_thresh[cell_val_zero] = penalty
        else:
            print('Condition max_value > min_value is voliated: output zero matrix')

    if binarize:
        S_thresh[S_thresh > 0] = 1
        S_thresh[S_thresh < 0] = 0
    return S_thresh



def extract_lines(S, extract_edges = False):
    """ Extract the lines with probabilistic Hough Transform.

    Parameters
    ----------
    S : np.array
        The self similarity matrix computed with cosine similarity.
    
    Returns
    -------
    S_smooth : np.array
        The SSM smoothed and thresholded.
    lines : list()
        The position of the line start and end relative to S.

    Examples
    --------
    from structure_analysis import analysis_to_SSM  
    import matplotlib.pyplot as plt
    
    S = analysis_to_SSM(note_array)
    S_smooth, lines, edges = extract_lines(S, extract_edges = True)
    
    # Generating figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('SSM smoothed and Thresholded')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edge Detection')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
    """
    
    S_smooth = threshold_matrix(forw_back_smoothing(S_cos), thresh=0.05, strategy='relative')
    edges = canny(S_smooth, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
    lines = probabilistic_hough_line(edges, threshold=1, line_length=15,
                                 line_gap=5)
    if extract_edges :
        return S_smooth, lines, edges
    else :
        return S_smooth, lines