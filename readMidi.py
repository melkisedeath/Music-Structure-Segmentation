def read_midi(file):
    """
    Read notes with onset and duration from MIDI file. Time is specified in beats.

    Parameters
    -----------
    file : strg
        Path to MIDI file

    Returns
    --------
    type : list of Multitype
        sorted list of MIDINote events, i.e. Onset_time, note_duration, MIDI_Notes
    """
    mid = mido.MidiFile(file)
    piece = []
    ticks_per_beat = mid.ticks_per_beat
    for track_id, t in enumerate(mid.tracks):
        time = 0
        track = []
        end_of_track = False
        active_notes = {}
        for msg in t:
            time += msg.time / ticks_per_beat
            if msg.type == 'end_of_track':
                # check for end of track
                end_of_track = True
            else:
                if end_of_track:
                    # raise if events occur after the end of track
                    raise ValueError("Received message after end of track")
                elif msg.type == 'note_on' or msg.type == 'note_off':
                    # only read note events
                    note = (msg.note, msg.channel)
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # note onset
                        if note in active_notes:
                            raise ValueError(f"{note} already active")
                        else:
                            active_notes[note] = (time, msg.velocity)
                    else:
                        # that is: msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)
                        # note offset
                        if note in active_notes:
                            onset_time = active_notes[note][0]
                            note_duration = time - active_notes[note][0]
                            # append to track
                            track.append(Event(time=onset_time,
                                               duration=note_duration,
                                               data=MIDINote(value=msg.note,
                                                             velocity=active_notes[note][1],
                                                             channel=msg.channel,
                                                             track=track_id)))
                            del active_notes[note]
                        else:
                            raise ValueError(f"{note} not active (time={time}, msg.type={msg.type}, "
                                             f"msg.velocity={msg.velocity})")
        piece += track
    return list(sorted(piece, key=lambda x: x.time))
