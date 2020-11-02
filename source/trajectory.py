from itertools import product

import ConvexHullMaxPairOfPoints as convHull
from Data_and_Dicts import dictOfTonnetz
from FirstNotePosition import rerrangeTonnetz, TonnetzToString
from TrajectoryClass import TrajectoryClass
""" Import convex Hull Comparison on set of cartesian points """
""" Import function that turn a Tonnetz list to a string """
""" Import dictionary with Tonnetz Base positions """
""" Import trajectory Object class """

INVALID_POS = (9999999, 9999999)
""" The value of the invalid Position, feel free to change """

# listofDist = lambda x : x + list(map(lambda y : 12 - y, x)  )


class PlacementError(RuntimeError):
    """Create an Error for Invalid Positioning."""

    def __init__(self, message="Could not place Chord with this strategy"):
        """Init function with ERROR message."""
        self.message = message


def isValidPos(pos):
    """Define a variable for Invalid Positions."""
    return pos != INVALID_POS


def applyFirstSuccessful(strategies):
    """Successively tries to apply different strategies.

    Successively tries to apply different strategies
    stopping at the first successful one.
    Strategies are functions which take no argument (typically lambdas
    wrapping a function with its arguments).
    """
    result = False
    for strategy in strategies:
        try:
            result = strategy()
        # Consume exceptions silently
        except PlacementError:
            pass
        if result:
            return result
    raise PlacementError()


# Probably rewrite that with lambda.
def axesMovementsDict(T_axes, point):
    """A dictionary to compute movement in Tonnetz Space."""
    x, y = point
    movementsDict = {
        0: (x, y),
        T_axes[0]: (x, y + 1),
        T_axes[1]: (x + 1, y),
        T_axes[2]: (x - 1, y - 1),
        12 - T_axes[0]: (x, y - 1),
        12 - T_axes[1]: (x - 1, y),
        12 - T_axes[2]: (x + 1, y + 1)
    }
    return movementsDict



# Probably rewrite that with lambda.
def axesMovementsDictDistancePlus(T_axes, point):
    """A dict to compute movement in Tonnetz for distances bigger than one."""
    x, y = point
    # the point represents the poisition of the previous note
    # and T_axes represent the distance.
    if T_axes == [3, 2, 7] or T_axes == [9, 2, 1]:
        movementsDict = {
            6: (x, y + 2),
            4: (x + 2, y),
            8: (x - 2, y),
            (T_axes[0] - T_axes[1]) % 12: (x - 1, y + 1),
            (T_axes[1] - T_axes[0]) % 12: (x + 1, y - 1)
        }
    else:
        movementsDict = {
            (T_axes[0] * 2) % 12: (x, y + 2),
            ((12 - T_axes[2]) * 2) % 12: (x + 2, y + 2),
            (T_axes[1] - T_axes[0]) % 12: (x + 1, y - 1),
            ((12 - T_axes[2]) + T_axes[1]) % 12: (x + 2, y + 1),
            ((12 - T_axes[2]) + T_axes[0]) % 12: (x + 1, y + 2),
        }
    return movementsDict


def intervalToPoint(interval, point, Tonnetz):
    """Turn an interval to a Point.

    This adapts an interval to a Position in the Cartesian Plane note
    that Intervals who don't belong onto axes values are Invalid Positions.

    Parameters
    ----------
    interval : int
        An interval, i.e. 3 is minor 3rd.
    point : tuple(int)
        a point.
    Tonnetz : list(int)
        A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..

    Returns
    -------
    point : tuple(int)
        The coordinates of a point relative to the interval.
    """
    movementDict = axesMovementsDict(Tonnetz, point)
    try:
        point = movementDict[interval]
    # here is the definition of Invalid Positions
    except KeyError:
        # If the point is not in the first dictionary create a second
        # with non connected fixed positions.
        try:
            movementDict2 = axesMovementsDictDistancePlus(Tonnetz, point)
            point = movementDict2[interval]
        except KeyError:
            point = INVALID_POS
    return point


def checkInvalidPosition(chord, point):
    """Check if a position is not valid."""
    if not isValidPos(point):
        print(chord, point)
        raise ValueError("Bad reference point")


def ChordConfiguration(chord, axes, Tonnetz):
    """Compute the coordinates of a chord.

    This function takes a chord (PC set) an origin point (usually the
    coordinates for one of the notes of the chord) and the Tonnetz System
    in which the trajectory is being build and returns (x, y) coordinates
    for all the notes of the chord. We take the Cartesian product of the chord
    and we iterate in order to find the coordinates.
    If the iteration exceed the length of the chord throw ERROR.

    Parameters
    ----------
    chord : list(int)
        A list of PC notes.
    axes : tuple(int)
    Tonnetz : list(int)
        A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..

    Returns
    -------
    coordDict : The coordinates for the chord parameter.
    """

    coordDict = {chord[0]: axes}
    n = 0
    while(len(chord) > len(coordDict)):
        for noteA, noteB in product(chord, chord):
            if(noteA in coordDict and noteB not in coordDict):
                newPoint = intervalToPoint(
                    (noteB - noteA) % 12, coordDict[noteA], Tonnetz)
                if isValidPos(newPoint):
                    coordDict[noteB] = newPoint
            if(n > len(chord)):
                print(
                    chord,
                    coordDict.items(),
                    axes,
                    n,
                    len(chord),
                    len(coordDict))
                raise RuntimeError("Infinite Loop")
        n += 1
    # If coordinates weren't found for all notes throw ERROR.
    if(any(note not in coordDict for note in chord)):
        print(chord, coordDict.items(), axes, Tonnetz)
        raise BaseException("Lost chord")
    return coordDict


# THIS HAS TO BE A GLOBAL VARIABLE
# See in begining of the document commented code!
def distanceOne(T_axes):
    """Return A list of accepted distances(On Tonnetz axes).



    """


    listofDist = [
        T_axes[0],
        T_axes[1],
        T_axes[2],
        (12 - T_axes[0]),
        (12 - T_axes[1]),
        (12 - T_axes[2])]
    return listofDist


def distanceInt(interval, T_axes):
    """Return a value that evaluates an interval.

    A single value in [0-2] estimation of note distance
    this is used to chose the origin point
    """
    listofDist = distanceOne(T_axes)
    if interval == 0:
        value = 0
    elif interval in listofDist:
        value = 1
    else:
        value = 2
    return value


def positionFromMin(chord, note, coordDict, Tonnetz):
    """Find the note that fits the description and its coordinates.

    Parameters
    ----------
    chord : list(int)
        A list of PC notes.
    note : int
        a note in PC notation
    coordDict : dict(tuple(int))
        The coordinates of the notes of chord in the Cartesian plane.
    Tonnetz : list(int)
        A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..

    Returns
    -------
    newPoint : tuple(int)
        the coordinates for note parameter.
        Takes a chord coordinates and an additional note and finds coordinates relative to this chord.
    """
    distanceValueList = [
        distanceInt((i - note) % 12, Tonnetz) for i in chord
    ]
    keyIndex = valueList.index(min(distanceValueList))
    noteA = chord[keyIndex]
    number = (note - noteA) % 12
    position = coordDict[noteA]
    newPoint = intervalToPoint(number, position, Tonnetz)
    return newPoint


def distance_matrix(chord1, chord2, Tonnetz):
    """The distance matrix for every couple of notes between two chords.

    Parameters
    ----------
    chord1 : list(int)
        A list of PC notes.
    chord2 : list(int)
        A list of PC notes.
    Tonnetz : list(int)
        A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..

    Returns
    -------
    l1 : list(int)
        the sum of rows of the distance matrix.
    l2 : list(int)
        the sum of collumns of the distance matrix.
    """
    matrix = [([(distanceInt((i - j) % 12, Tonnetz)) for i in chord1])
          for j in chord2]
    # sum of rows
    l1 = [sum([row[i] for row in matrix]) for i in range(len(chord1))]
    # sum of collumns
    l2 = list(map(sum, matrix))
    return l1, l2


def IndexesOfMinimum(chord1, chord2, Tonnetz):
    """Take two chords and find the indexes of the pair with min distance.

    Parameters
    ----------
    chord1 : list(int)
        A list of PC notes.
    chord2 : list(int)
        A list of PC notes.
    Tonnetz : list(int)
        A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..

    Returns
    -------
    minimumIndex1 : int
        the index of the prefered element of chord1
    minimumIndex2 : int
        the index of the prefered element of chord2
        
    By searching the distance matrix collumn and row sums we find the closest
    pair of notes between two chords (usually distance 0 or 1).
    """
    l1, l2 = distance_matrix(chord1, chord2, Tonnetz)
    min1 = min(l1)
    min2 = min(l2)
    minimumIndex1 = l1.index(min1)
    minimumIndex2 = l2.index(min2)
    distValue = distanceInt(
        (chord1[minimumIndex1] - chord2[minimumIndex2]) %
        12, Tonnetz)
    if distValue >= 1:
        listOfMinIndices1 = [i for i, n in enumerate(l1) if n > min1 - 2]
        listOfMinIndices2 = [i for i, n in enumerate(l2) if n > min2 - 2]
        minCheck = 2
        for i in listOfMinIndices1:
            for j in listOfMinIndices2:
                distVal = distanceInt((chord1[i] - chord2[j]) % 12, Tonnetz)
                if distVal < minCheck:
                    minimumIndex1 = i
                    minimumIndex2 = j
                    minCheck = distVal
    return minimumIndex1, minimumIndex2


def positionOfTheMinNote(chord1, chord2, coordDict1, Tonnetz):
    """Find the actual position of the pair of closest notes.

    Parameters
    ----------
    chord1 : list(int)
        A list of PC notes.
    chord2 : list(int)
        A list of PC notes.
    coordDict1 : dict(tuple(int))
        The coordinates of the notes of chord1 in the Cartesian plane.
    Tonnetz : list(int)
        A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..

    Returns
    -------
    newPoint : tuple(int)
        The position of the most relevant note of chord2. 
        We need that to constract all the other positions of the chord
        with this reference point.
    position : tuple(int)
        The position of the most relevant note of chord1 to chord2.
    """
    index1, index2 = IndexesOfMinimum(chord1, chord2, Tonnetz)
    noteA = chord1[index1]
    noteB = chord2[index2]
    chord2[0], chord2[index2] = chord2[index2], chord2[0]
    interval = (noteB - noteA) % 12
    position = coordDict1[noteA]
    newPoint = intervalToPoint(interval, position, Tonnetz)
    return newPoint, position


def concat3DictValues(Dict1, Dict2, Dict3):
    """Concat the positions of three consecutive chords.

    Parameters
    ----------
    Dict1 : dict(tuple(int))
        The coordinates of some chord.
    Dict2 : dict(tuple(int))
        The coordinates of some chord.
    Dict3 : dict(tuple(int))
        The coordinates of some chord.

    Returns
    -------
    lconcat : list(tuple(int))
        The concatenation of coordinates of all input chords.
    """
    l1 = list(Dict1.values())
    l2 = list(Dict2.values())
    l3 = list(Dict3.values())
    lconcat = l1 + l2 + l3
    return lconcat


def maximumDistanceOfConvexHull(graph1):
    """Compute maximum diameter of a Convex Hull.

    Parameters
    ----------
    graph1 : list(tuple(int))
        A list of coordinates in the Cartesian plane.

    Returns
    -------
    maxdistance : float
        The Eucledian distance of the two most distants points of the graph.
        From a set of (x, y) points find the most distant and
        compute their cartesian distance.    
    """
    point1, point2 = convHull.diameter(graph1)
    sumofsquares = (point1[0] - point2[0]) ^ 2 + (point1[1] - point2[1]) ^ 2
    maxdistance = format(sumofsquares**(0.5), '.2f')
    return maxdistance


def computeChordCoord(thisChord, someChordCoord, Tonnetz):
    """Compute a chord's coordinates in a dictionary format.

    Parameters
    ----------
    thisChord : list(int)
        A list of PC notes
    someChordCoord : dict(tuple(int))
        The coordinates of some chord.
    Tonnetz : list(int)
        A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..

    Returns
    -------
    thisChordCoord : dict(tuple(int))
        Compute a chord's coordinates in a dictionary format with :
        -  keys : the name of the PC-notes
        -  values : (x, y) coordinates
    edge : tuple(tuple(int))
        The connecting edge between two chords, based on minimum distance.
    """
    origin, otherRefOrigin = positionOfTheMinNote(
        list(someChordCoord.keys()), thisChord, someChordCoord, Tonnetz)
    if not isValidPos(origin):
        raise PlacementError()
    thisChordCoord = ChordConfiguration(thisChord, origin, Tonnetz)
    edge = [(origin, otherRefOrigin)]
    return thisChordCoord, edge


def TrajectoryConvexHullComparison(
        placement1,
        placement2,
        lastChordCoord,
        secondLastChordCoord):
    """Convex Hull Comparison of two different chord sequences.

    From two different chordCoordinates choose the most compact
    That is done by choosing the minimum greatest convexHull diameter.
    """
    concatPoints1 = concat3DictValues(
        placement1[0],
        lastChordCoord,
        secondLastChordCoord)
    concatPoints2 = concat3DictValues(
        placement2[0],
        lastChordCoord,
        secondLastChordCoord)
    graph1 = list(set(concatPoints1))
    graph2 = list(set(concatPoints2))
    distance1 = maximumDistanceOfConvexHull(graph1)
    distance2 = maximumDistanceOfConvexHull(graph2)
    if distance1 > distance2:
        return placement2
    else:
        return placement1


def TrajectoryCheckSecond(placement1, trajectory):
    """Check if a second chord configuration based in future is valid."""
    try:
        secondLastChordCoord = trajectory.getLastPosition(2)
        lastChordCoord = trajectory.getLastPosition()
        nextChord = trajectory.getNextChord()
        placement2 = placeChordWithVirtualRef(
            trajectory.getThisChord(),
            lastChordCoord,
            nextChord,
            trajectory.Tonnetz)
        return TrajectoryConvexHullComparison(
            placement1, placement2, lastChordCoord, secondLastChordCoord)
    except PlacementError:
        return placement1


def TrajectoryLookConnected(trajectory):
    """A test to check conditions."""
    thisChord = trajectory.getThisChord()
    thisChordPoints1, edge1 = computeChordCoord(
        thisChord, trajectory.getLastPosition(), trajectory.Tonnetz)
    if edge1[0][1] != edge1[0][0]:
        try:
            thisChordPoints2, edge2 = computeChordCoord(
                thisChord, trajectory.getLastPosition(2), trajectory.Tonnetz)
            if edge2[0][1] == edge2[0][0]:
                return TrajectoryCheckSecond(
                    (thisChordPoints2, edge2), trajectory)
        except PlacementError:
            pass
    return TrajectoryCheckSecond((thisChordPoints1, edge1), trajectory)


def TrajectoryCheckPosition(trajectory):
    """Apply tactics on trajectory calculations.

    Parameters
    ----------
    trajectory : object
        A trajectory object.

    Returns
    -------
    chordCoord : dict(tuple(int))
        The coordinates of a chord. This method tries a number 
        of different tactics for placing a chord and applies the first succesful.
    """
    return applyFirstSuccessful([
        lambda: TrajectoryLookConnected(trajectory),
        lambda: computeChordCoord(
                trajectory.getThisChord(),
                trajectory.getLastPosition(2),
                trajectory.Tonnetz),
        *(
            lambda: placeChordWithVirtualRef(
                    trajectory.getThisChord(),
                    trajectory.getLastPosition(),
                    trajectory.getNextChord(lookahead),
                    trajectory.Tonnetz)
            for lookahead in range(1, min(5, trajectory.chordsRemaining()))
        )
    ])


def TrajectoryWithFuture(trajectory):
    """Start Normal Calculations.
    
    Parameters
    ----------
    trajectory : object
        A trajectory object
    
    Returns
    -------
    chordCoord : dict(tuple(int))
        The coordinates of a chord.
    """
    if trajectory.index > 1 and trajectory.chordsRemaining() > 1:
        chordCoord = TrajectoryCheckPosition(trajectory) 
    elif trajectory.index == 0:
        raise PlacementError("Strategy not valid for this position")
    else:
        chordCoord  = computeChordCoord(
            trajectory.getThisChord(),
            trajectory.getLastPosition(),
            trajectory.Tonnetz)
    return chordCoord


def placeChordWithVirtualRef(thisPCS, placedChordCoord, tempPCS, Tonnetz):
    """Find a chords coordinates based on a reference chord.
    
    Parameters
    ----------
    thisPCS : list(int)
        A chord in Pitch Class notation
    placedChordCoord : dict(tuple(int))
        The coordinates of specified chord, usually the last chord with known coordinates.
    tempPCS : list(int)
        the next chord in Pitch Class notation
    Tonnetz : list(int)
        A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..

    Returns 
    -------
    chordCoord : dict(tuple(int))
        The coordinates of the current chord.
    """
    virtualChordCoord, _ = computeChordCoord(
        tempPCS, placedChordCoord, Tonnetz)
    chordCoord = computeChordCoord(thisPCS, virtualChordCoord, Tonnetz)
    return chordCoord

def get_edge_3D(candx, candy):
    ''' Compute the right 3D edge from lists of candidates.

    Parameters
    ----------
    candx : list(tuples(int, int, float))
        The candidate positions of the first chord
    candy : list(tuples(int, int, float))
        The candidate positions of the second chord
    '''
    minimum = 11
    for elx in candx:
        for ely in candy:
            if minimum > abs(elx[2] - ely[2]):
                minimum = abs(elx[2] - ely[2])
                connectingEdge = (elx, ely)
    return connectingEdge


def buildTrajectory(listOfChords, Tonnetz, origin=(0, 0)):
    """The Call function for trajectory Calculations.

    Parameters
    ----------
    listOfChords : list(list(int))
        The chords of the piece in Midi_pitches.
    Tonnetz : list(int)
        A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..
    origin : tuple(int)
        the origin point for the first note of the trajectory.
        Default value (0, 0) preferably for a C note.

    Returns
    -------
    trajectory : object
        a trajectory object (See the trajectory class for more information)
    """
    Tonnetz = rerrangeTonnetz(Tonnetz)
    # Adjust Tonnetz to correcly calculate coordinates
    c_pc = list(set(map(lambda x : x%12, listOfChords[0])))
    chordCoord_2D = ChordConfiguration( 
                                        c_pc,
                                        origin,
                                        Tonnetz)
    chordCoord_3D = {
        mp: (chordCoord_2D[mp % 12][0], chordCoord_2D[mp % 12][1], round(mp/12, 2)) 
            for mp in listOfChords[0]}
    trajectory = TrajectoryClass(
        chordCoord_3D,
        listOfChords,
        Tonnetz)
    for index, chord in enumerate(listOfChords):
        if index == 0:
            continue
        elif index == 1:
            thisChordCoord, connectingEdge = applyFirstSuccessful([
                lambda: computeChordCoord(
                    trajectory.getThisChord(),
                    trajectory.getLastPosition(),
                    trajectory.Tonnetz),
                lambda: placeChordWithVirtualRef(
                    trajectory.getThisChord(),
                    trajectory.getLastPosition(),
                    trajectory.getNextChord(),
                    trajectory.Tonnetz)
            ])
        else:
            thisChordCoord, connectingEdge = TrajectoryWithFuture(trajectory)
        
        thisChordCoord_3D = {
            mp: (thisChordCoord[mp % 12][0], thisChordCoord[mp % 12][1], round(mp/12, 2)) 
            for mp in listOfChords[index]
            }
        candx = [x for x in trajectory.chordPositions[-1].values() 
            if (x[0], x[1]) == connectingEdge[0][0]]
        candy = [y for y in thisChordCoord_3D.values() 
            if (y[0], y[1]) == connectingEdge[0][1]]
        try :
            connectingEdge = get_edge_3D(candx, candy)
        except UnboundLocalError :
            connectingEdge = []
        trajectory.addChord(thisChordCoord_3D, connectingEdge)
    return trajectory


# ------------------------TRAJECTORY EDGES----------------------------------


def TrajectoryNoteEdges(trajectory):
    """Compute the edges of every chord in the trajectory.


    Parameters
    ----------
    trajectory : object
        A trajectory object.

    Returns
    -------
    TotalEdges : list(tuples(tuple(int)))
        From trajectory note positions calculate the interconnecting edges.
    """
    TotalEdges = []
    dist = [-1, 0, 1]
    for dicts in trajectory.chordPositions:
        chordEdges = []
        cartl = list(product(dicts.values(), dicts.values()))
        for couple in cartl:
            (x1, y1), (x2, y2) = couple
            if (x1 - x2) in dist and (y1 - y2) in dist:
                if not (((x1 - x2) == 1 and (y1 - y2) == -1) or
                        ((x1 - x2) == -1 and (y1 - y2) == 1)):
                    chordEdges.append(couple)
        TotalEdges.append(chordEdges)
    return TotalEdges


def SetOfPoints(trajectory):
    """Remove duplicate points.
    
    Parameters
    ----------
    trajectory : object
        A trajectory object.

    Returns
    -------
    PointSet : list(tuples(int))
        From trajectory positions remove duplicates.
    AllPoint : list(tuples(int))
        All trajectory positions with duplicates.
    """
    AllPoints = []
    for dicts in trajectory.chordPositions:
        AllPoints = AllPoints + list(dicts.values())
    PointSet = list(set(AllPoints))
    return PointSet, AllPoints


def weightsOfTrajPoints(setOfPoints, multiSetOfPoints):
    """Calculate the multiplicity of Points and normalize.

    Parameters
    ----------
    SetOfPoints : list(tuples(int))
        From trajectory positions remove duplicates.
    multiSetOfPoints : list(tuples(int))
        All trajectory positions with duplicates.

    Returns
    -------
    dictOfPointWeight : dict(key : tuple(int), value : multiplicity)
    """
    dictOfPointWeight = dict()
    for point in setOfPoints:
        dictOfPointWeight[point] = multiSetOfPoints.count(point)
    # Ideas about Using Normalized weights :
    # Maximum = max(list(dictOfPointWeight.values()))
    # Minimum = min(list(dictOfPointWeight.values()))
    return dictOfPointWeight


def weightsOfTrajPoints_Normalized(setOfPoints, multiSetOfPoints):
    """Calculate the multiplicity of Points and normalize.

    Parameters
    ----------
    SetOfPoints : list(tuples(int))
        From trajectory positions remove duplicates.
    multiSetOfPoints : list(tuples(int))
        All trajectory positions with duplicates.

    Returns
    -------
    dictOfPointWeight : dict(key : tuple(int), value : normalized multiplicity)
    """
    pointWeight = [multiSetOfPoints.count(point) for point in setOfPoints]
    normalizer = 1 / float( sum(pointWeight) )

    # multiply each item by the normalizer
    normalized = dict(zip(setOfPoints, [x * normalizer for x in pointWeight]))
    # Ideas about Using Normalized weights :
    # Maximum = max(list(dictOfPointWeight.values()))
    # Minimum = min(list(dictOfPointWeight.values()))
    return normalized





'''
EXAMPLE CODE
-------------
Here is some example Code to test stuff.

'''
# listOfChords = [[48, 55, 60, 64, 67, 72], [60, 65, 69], [62, 65, 69, 72], [60, 64, 67], [60, 62, 67], [62, 67, 71, 77], [60, 64, 67, 72, 76, 84]]
# Tonnetz = [3, 4, 5]
# trajectory = buildTrajectory(listOfChords, Tonnetz)
# print(trajectory.connectingEdges)
# ps, pms = SetOfPoints(trajectory)
# print(ps)
# print(weightsOfTrajPoints_Normalized(ps, pms))

# print(trajectory.chordPositions)

# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# xs, ys, zs = zip(*ps)

# ax.scatter(xs, ys, zs, marker='o')

# ax.set_xlabel('3 axis')
# ax.set_ylabel('4 axis')
# ax.set_zlabel('Pitch')

# plt.show()