class TrajectoryClass:
    """A class containing Trajectory Information."""

    def __init__(self, initialChordPosition, listOfChords, Tonnetz):
        """Initialize function of Trajectory.

        Parameters
        ----------
        self : obj
            Takes the Trajectory Class object.
        initialChordPosition : list(dict(int))
            The positions of the first chord. For the first chord of a trajectory.
            We give an origin point and a placing algorithm.
        listOfChords : list(list(int))
            A list of Chords in Pitch class notations.
        Tonnetz : list(int)
            A list with the Tonnetz intervals, i.e. [3,4,5] or [1,2,9], etc..

        Returns
        -------
        self : object
        Initializes a Trajectory Object with arguments:
            chordPositions : list(dict, int)
            connectingEdges : list(tuples(tuples(int)))
            index : int
            listOfChords : list(list(int))
            Tonnetz : list(int)
        """
        self.chordPositions = [initialChordPosition]
        self.connectingEdges = []
        self.index = 1  # Redundant: should always be len(chordPositions)
        self.listOfChords = listOfChords
        self.Tonnetz = Tonnetz

    def addChord(self, chordPosition, connectingEdge):
        """Add a new chord in Trajectory.

        Parameters
        ----------
        self : obj
            Takes the Trajectory Class object.
        chordPosition : 

        connectingEdge : ((int, int), (int, int))
            The connecting Edge between the positions of two successive chords.

        Returns
        -------
        Updates the Trajectory Object
        """
        self.chordPositions.append(chordPosition)
        self.connectingEdges.append(connectingEdge)
        self.index += 1

    def getLastPosition(self, offset=1):
        """Get the last chord coordinates, or change offset.
            
        Parameters
        ----------
        self : obj
            Takes the Trajectory Class object.
        offset : int
            the index of inverse chord list order.

        Returns
        -------
        self.listOfChords[self.index] : dict((int, int))
            The positions of the last chord or previous depending on the offset.
        """
        if offset > self.index:
            raise IndexError()
        chordCoord = {k : (v[0], v[1]) for k, v in self.chordPositions[-offset].items()}
        return chordCoord

    def getThisChord(self):
        """Get the PC values of the currect chord.

        Parameters
        ----------
        self : obj
            Takes the Trajectory Class object.

        Returns
        -------
        self.listOfChords[self.index] : dict((int, int))
            The positions of the current chord.
            """
        c_pc = list(set(map(lambda x : x%12, self.listOfChords[self.index])))
        return c_pc

    def getNextChord(self, offset=1):
        """Get the PC values of the next chord, or change offest.

        Parameters
        ----------
        self : obj
            Takes the Trajectory Class object.
        offset : int
            the index of chord list order starting at the present chord.

        Returns
        -------
        self.listOfChords[self.index + offset] : dict((int, int))
            The positions of the next chord or future ones depending on the offset.
        """
        c_pc = list(set(map(lambda x : x%12, self.listOfChords[self.index + offset])))
        return c_pc

    # def addType(self, trajType):
    #     """Precise the type of the Trajectory, recursive, with future, etc.


    #     """
    #     self.type = trajType

    def chordsRemaining(self):
        """Return the number of remaining chords to place.

        Parameters
        ----------
        self : obj
            Takes the Trajectory Class object.

        Returns
        -------
        remainingChords : int
            The number of chords remaining to be assigned a position.
        """
        remainingChords = len(self.listOfChords) - len(self.chordPositions)
        return remainingChords




# ADD MIDI FILE PROPERTIES

    def addNumberOfInstruments(self, numberOfInstruments):
        """How many instruments in midi file.

        The number of instruments typically is provides by program changes.
        """
        self.numOfInstr = numberOfInstruments

    def addInstruments(self, listOfInstruments):
        """A list with all the instruments, no duplicates."""
        self.instruments = list(set(listOfInstruments))
        self.addNumberOfInstruments(len(set(listOfInstruments)))

    def addTempo(self, tempo):
        """Tempo Estimation."""
        self.tempo = tempo

    def addNumber_of_signature_changes(self, number):
        """Number of time signature changes."""
        self.number_of_signature_changes = number

    def addTime_signatures(self, signature_changes):
        """Add the time signatures of the piece.

        The default value if the time signature is not precised is 4/4
        """
        self.time_signatures = list(set(signature_changes))
        self.addNumber_of_signature_changes(len(signature_changes))
