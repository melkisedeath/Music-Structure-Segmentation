from structure_analysis import performance_analysis
from performance_analysis_with_stats import perform_statistical_analysis
from partitura.io.importmatch import MatchFile
import numpy as np 
import os, sys
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import itertools



sampling_attr_dict = {
    "tempo" : 0,
    "velocity" : 1,
    "timing" : 2,
    "articulation" : 3
}

stat_attributes = {
    "tempo" : lambda x, stat: tempo_of_segment(x, stat),
    "all" : lambda x, stat: expressiveness_of_segment(x, stat),
             }

statistic_methods = {
    "mean" : lambda y : np.mean(y),
     "std" : lambda y : np.std(y),
     "var" : lambda y : np.var(y),
                    }


def performance_array_to_attr(Pn, t=10):
    or_shape = Pn.shape
    Pn = Pn.reshape(or_shape[0], np.prod(or_shape[1:]))
    f = lambda i : np.hstack((Pn[:, 0 + i*t : t + i*t], Pn[:, 4*t + i*t: 5*t+i*t], Pn[:,8*t+i*t: 9*t +i*t], Pn[:,12*t+i*t: 13*t+i*t]))
    P_tempo = f(0)
    P_vel = f(1)
    P_timing = f(2)
    P_artic = f(3)
    return P_tempo, P_vel, P_timing, P_artic


def std_curve_of_pivots(X, step, bar, dummy_section=2):
    """
    From the performance analysis output a curve of larger sections.

    Parameters
    ----------
    X : np.array
        A performance analysis.
    step : int
        the step of the performance analysis X.
    bar : int
        the length of a bar.
    dummy_section : int
        The length of a dummy sections in bars.


    Returns
    -------
    RX : np.array
        The performance analysis of RP.
    X1 : np.array
        The performance analysis of P1.
    X2 : np.array
        The performance analysis of P2.
    """
    dummy_section *= bar
    number_of_bars = int(len(X)*step/bar)
    std_array = np.zeros((number_of_bars))
    for i in range(number_of_bars):
        std_array[i] = np.std(X[i*bar : i*bar + dummy_section])
    return std_array


def std_curve_fitting_plot(X, step, bar, title_text, save_path):
    y = std_curve_of_pivots(X, step, bar).tolist()
    x = range(len(y))
    # coefs = np.polynomial.polynomial.polyfit(x, y, len(x)+1)
    # ffit = np.polynomial.polynomial.Polynomial(coefs)
    # z = np.linspace(0, 100, num=200)
    fig = go.Figure(px.line(x=x, y=y))
    title_text = 'Standard_Deviation_Curve_fitting - '+ os.path.basename(title_text)
    fig.update_layout(title_text=title_text)
    pio.write_html(fig, file=os.path.join(save_path, title_text+'.html') , auto_open=True)


def get_step_and_bar(fn):
    """
    Method to compute step and bar from time signature and estimated tempo.

    Parameters
    ----------
    fn : string
        The path to the match file

    Returns
    -------
    step : int
        The step for the analysis relevant to bar and tempo.
    bar : int
        The bar length relevant to the beat, i.e. 4/4 -> 4, 2/8 -> 2
    """
    mf = MatchFile(fn)
    # A list of tuples(t, b, (n, d)), indicating a time signature of n over d, starting at t in bar b
    ts = mf.time_signatures
    if len(ts) > 1 :
        print("There is method yet to treat multiple time_signatures")
        # output default step
        step = 1
        bar = None
    elif len(ts) == 0:
        raise ValueError("No time signatures found")
    else:
        time_signature = ts[0]
        numerator = time_signature[2][0]
        denominator = time_signature[2][1]
        bar = numerator
        step = 1
        # use the following to calculate step based on tempo
        if denominator == 2:
            pass
        elif denominator == 4:
            pass
        elif denominator == 8:
            pass
        else:
            raise ValueError("Unrecognised Time signature")
        return step, bar

def get_analysis(P, method="sampling", attr="tempo", stat="mean"):
    """ From a match scores output performance analysis.

    Parameters
    ----------
    P : path
        The match score path.
    method : str (optional)
        The analysis method, i.e. "sampling" or "statistical".
    attr : str (optional)
        The expressive attribute to extract, 
        i.e. "tempo", "velocity", "timing", "articulation", or "all".
    stat : str (optional if method=="statistical")
        The statistical method, i.e. "mean", "std", "var" .

    Returns
    -------
    RX : np.array
        The performance analysis of RP.
    X1 : np.array
        The performance analysis of P1.
    X2 : np.array
        The performance analysis of P2.
    """

    step, bar = get_step_and_bar(P)

    if method == "sampling":
        # find step, window_size and samples_per_window
        window_size = int(bar / step)
        # samples per window should be the min notes per window_size.
        # set default value for now
        samples_per_window = 10

        X = performance_analysis(P, step, window_size, samples_per_window)
        if attr in sampling_attr_dict.keys(): 
            X = performance_array_to_attr(X)[sampling_attr_dict[attr]]
        elif attr == "all":
            pass
        else:
            raise KeyError("Wrong attribute name inserted")
    elif method == "statistical":
        # find step and number of windows

        if attr == "all":
            number_of_windows = 4
        if attr == "tempo":
            number_of_windows = 2
        try:
            X = perform_statistical_analysis(P, stat_attributes[attr], statistic_methods[stat], step, number_of_windows)
        except KeyError:
            print("This attribute doesn't match the method")
            raise
    else:
        raise KeyError("Wrong method name inserted")

    return X


def save_std_plot(P, method="sampling", attr="tempo", stat="mean", path_folder=None):
    """
    Save Standard Deviation plot

    Parameters
    ----------
    P : path
        The match score path.
    """

    if path_folder!=None:
        P = os.path.join(path_folder, RP)
    else :
        print("No path folder was given")

    X = get_analysis(P)
    step, bar = get_step_and_bar(P)
    save_folder = os.path.join(os.getcwd(), "html")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    std_curve_fitting_plot(RX, step, bar, P + "_" + method + "_" + attr, save_folder)


def triplet_analysis(RP, P1, P2, method="sampling", attr="tempo", stat="mean"):
    """ From three match scores output performance analysi.

    Parameters
    ----------
    RP : path
        The reference match score path.
    P1 : path
        The match score path 1.
    P2 : path
        The match score path 2.
    method : str (optional)
        The analysis method, i.e. "sampling" or "statistical".
    attr : str (optional)
        The expressive attribute to extract, 
        i.e. "tempo", "velocity", "timing", "articulation", or "all".
    stat : str (optional if method=="statistical")
        The statistical method, i.e. "mean", "std", "var" .

    Returns
    -------
    RX : np.array
        The performance analysis of RP.
    X1 : np.array
        The performance analysis of P1.
    X2 : np.array
        The performance analysis of P2.
    """

    step, bar = get_step_and_bar(RP)

    if method == "sampling":
        # find step, window_size and samples_per_window
        window_size = int(bar / step)
        # samples per window should be the min notes per window_size.
        # set default value for now
        samples_per_window = 10

        RX = performance_analysis(RP, step, window_size, samples_per_window)
        X1 = performance_analysis(P1, step, window_size, samples_per_window)
        X2 = performance_analysis(P2, step, window_size, samples_per_window)
        if attr in sampling_attr_dict.keys(): 
            RX = performance_array_to_attr(RX)[sampling_attr_dict[attr]]
            X1 = performance_array_to_attr(X1)[sampling_attr_dict[attr]]
            X2 = performance_array_to_attr(X2)[sampling_attr_dict[attr]]
        elif attr == "all":
            pass
        else:
            raise KeyError("Wrong attribute name inserted")
    elif method == "statistical":
        # find step and number of windows

        if attr == "all":
            number_of_windows = 4
        if attr == "tempo":
            number_of_windows = 2
        try:
            RX = perform_statistical_analysis(RP, stat_attributes[attr], statistic_methods[stat], step, number_of_windows)
            X1 = perform_statistical_analysis(P1, stat_attributes[attr], statistic_methods[stat], step, number_of_windows)
            X2 = perform_statistical_analysis(P2, stat_attributes[attr], statistic_methods[stat], step, number_of_windows)
        except KeyError:
            print("This attribute doesn't match the method")
            raise
    else:
        raise KeyError("Wrong method name inserted")

    return RX, X1, X2


def triplet_distance(RP, P1, P2, path_folder=None):
    """
    The distance of the triplet.

    Parameters
    ----------
    RP : path
        The reference match score path.
    P1 : path
        The match score path 1.
    P2 : path
        The match score path 2.
    """

    if path_folder!=None:
        RP = os.path.join(path_folder, RP)
        P1 = os.path.join(path_folder, P1)
        P2 = os.path.join(path_folder, P2)
    else :
        print("No path folder was given")

    RX, X1, X2 = triplet_analysis(RP, P1, P2, method="sampling", attr="tempo", stat="mean")
    step, bar = get_step_and_bar(RP)
    save_folder = os.path.join(os.getcwd(), "html")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    std_curve_fitting_plot(RX, step, bar, RP + "_tempo", save_folder)
    std_curve_fitting_plot(X1, step, bar, P1 + "_tempo", save_folder)
    std_curve_fitting_plot(X2, step, bar, P2 + "_tempo", save_folder)
    return "The program has runned succesfully"



Directory = "C://Users//melki//Desktop//JKU//codes//alignments//match_4"

triplet_distance(
    "Mozart_K331_1st-mov_p01.match",
    "Mozart_K331_1st-mov_p02.match",
    "Mozart_K331_1st-mov_p03.match",
    Directory
    )




# for file in os.walk(Directory):
#     if file.endswith(".match"):
#          for (method, attr) in itertools.product(["sampling", "statistical"], sampling_attr_dict.keys()):
#             try :
#                 P = Directory + file
#                 save_std_plot(P, method=method, attr=attr, stat="mean", path_folder=Directory)
#             except :
#                 pass


from metric_learn import SCML

def training(directory_real, directory_random):
    dir_real = os.listdir(directory_real)
    dir_rand = os.listdir(directory_random)
    # it should be the pre-processed files.
    triplets = [ (p1, p2, r) 
        for (p1, p2), r in itertools.product(
            [(p1, p2) for (p1, p2) in itertools.permutations(dir_real, 2)], dir_rand
            )
        if p1.endswith(".match") and p2.endswith(".match") and r.endswith(".match")
        ]
    scml = SCML(random_state=42)
    scml.fit(triplets)
