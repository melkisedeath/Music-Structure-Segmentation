from Data_and_Dicts import dictOfTonnetz, dictOfTonnetze, dictOfTonnetzeRearranged
from structural_functions import getKeyByValue


def TonnetzToString(Tonnetz):
    """TonnetzToString: List -> String.

    Parameters
    ----------
    Tonnetz : list(int)
        A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..

    Returns
    -------
    TonnetzString : string
        A string that evaluates the Tonnetz intervals as strings.
        Example! Tonnetz list [3, 4, 5] evaluates to "T345".
    """
    TonnetzString = getKeyByValue(dictOfTonnetze, Tonnetz)
    return TonnetzString


# TODO just Take a Chord and Place the first Note.
def PlaceFirstNote(listOfChords, Tonnetz):
    """Take a Chord and Place the first Note.
    

    Parameters
    ----------
    listOfChordss : list(list(int))
        A list with the chords in pitch class notation
    Tonnetz : list(int)
        A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..


    Returns
    -------
    position : (int, int)
        A position of integer coordinates in the Cartesian plane.
    """
    position = (0, 0)
    try:
        firstNote = listOfChords[0][0]
        position = dictOfTonnetz[TonnetzToString(Tonnetz)][firstNote]
    except KeyError():
        print("This Tonnetz's Initial position is not defined")
    return position


def rerrangeTonnetz(Tonnetz):
    """Re-arrange the Tonnetz List to place points correctly.
    

    Parameters
    ----------
    Tonnetz : list(int)
        A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..


    Returns
    -------
    dictOfTonnetzeRearranged[TonnetzString] : list(int)
        A list with the Tonnetz intervals but re-arranged in a more convinient positioning manner.
            Example [1,3,8] -> [3, 8, 1]
        The concept is to create a 3 by 4 periodicity on the Cartesian plane.
    """
    TonnetzString = TonnetzToString(Tonnetz)
    return dictOfTonnetzeRearranged[TonnetzString]
