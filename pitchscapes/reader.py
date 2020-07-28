import numpy as np
from collections import namedtuple

import mido

from .scapes import PitchScape
from .util import sample_pitch_scape


Event = namedtuple("Event", "time duration data")
MIDINote = namedtuple("MIDINote", "value velocity channel track")


def read_midi(file):
    """
    Read notes with onset and duration from MIDI file. Time is specified in beats.
    :param file: path to MIDI file
    :return: sorted list of MIDINote events
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


def chordify(piece):
    """
    Create time bins at note events (on- or offset). For each bin create a set of notes that are on.
    :param piece: List of MIDINote Events
    :return: list of Events (with start time and duration) with note sets
    """
    # create dictionary with time on- and offsets and events starting at a certain onset
    event_dict = {}
    for e in piece:
        # add onset and offset time slot
        if not e.time in event_dict:
            event_dict[e.time] = set()
        if not e.time + e.duration in event_dict:
            event_dict[e.time + e.duration] = set()
        # add event to onset time slot
        event_dict[e.time].add(e)
    # turn dict into ordered list of time-events pairs
    event_list = list(sorted(event_dict.items(), key=lambda item: item[0]))
    # take care of events that extend over multiple time slots
    active_events = set()
    for time, events in event_list:
        # from the active events (that started earlier) only keep those that reach into current time slot
        active_events = set(event for event in active_events if event.time + event.duration > time)
        # add these events to the current time slot
        events |= active_events
        # remember events that start with this slot to possibly add them in later slots
        active_events |= events
    # the last element should not contain any events as it was only created because the last event(s) ended at that time
    if event_list[-1][1]:
        raise ValueError("The last time slot should be empty. This is a bug (maybe due to floating point arithmetics?)")
    # turn dict into an ordered list of events with correct durations and combined event data
    return [Event(time=time, duration=next_time - time, data=frozenset([e.data for e in events]))
            for (time, events), (next_time, next_events) in zip(event_list, event_list[1:])]


def pitch_class_counts_from_chordified(chordified):
    """
    Take a chordified piece and compute pitch class counts for each time slot.
    :param chordified: chordified piece
    :return: tuple of two numpy arrays: one of shape (N, 12) with with pitch-class counts for each time bin,
    and one of shape (N + 1,) with time boundaries
    """
    pitch_class_counts = np.zeros((len(chordified), 12))
    for time_idx, event in enumerate(chordified):
        for pitch in event.data:
            pitch_class_counts[time_idx][pitch.value % 12] += 1
    times = np.array([event.time for event in chordified] + [chordified[-1].time + chordified[-1].duration])
    return pitch_class_counts, times


def pitch_class_counts(file):
    """
    Read pitch-class counts from MIDI file by concatenating read_midi, chordify, and pitch_class_counts
    :param file: MIDI file
    :return: 2-tuple of numpy arrays with pitch-class counts and time boundaries
    """
    return pitch_class_counts_from_chordified(chordify(read_midi(file)))


def get_pitch_scape(file_path, normalise=False):
    pitch_counts, times = pitch_class_counts(file_path)
    return PitchScape(values=pitch_counts, times=times, normalise=normalise)


def sample_piece(n_samples,
                 file_path=None,
                 scape=None,
                 prior_counts=1.,
                 normalise=False):
    if (file_path is None) == (scape is None):
        raise ValueError("Have to provide exactly one of 'file_path' and 'scape' as arguments")
    if file_path is not None:
        scape = get_pitch_scape(file_path, normalise=normalise)
    return sample_pitch_scape(scape=scape, n_samples=n_samples, prior_counts=prior_counts)
