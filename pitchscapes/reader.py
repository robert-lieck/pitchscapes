from collections import namedtuple
from itertools import product
from warnings import warn

import numpy as np
import mido
import music21

from .scapes import PitchScape
from .util import sample_pitch_scape


Event = namedtuple("Event", "time duration data")


def read_midi_mido(file):
    """
    Read notes with onset and duration from MIDI file. Time is specified in beats.
    :param file: path to MIDI file
    :return: sorted list of pitch Events
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
                                               data=msg.note))
                            del active_notes[note]
                        else:
                            raise ValueError(f"{note} not active (time={time}, msg.type={msg.type}, "
                                             f"msg.velocity={msg.velocity})")
        piece += track
    return list(sorted(piece, key=lambda x: x.time))


def read(file):
    events = []
    try:
        piece = music21.converter.parse(file)
    except Exception:
        raise RuntimeError(f"Could not read the file '{file}'")
    for part_id, part in enumerate(piece.parts):
        for note in part.flat.notes:
            if isinstance(note, (music21.note.Note, music21.chord.Chord)):
                for pitch in note.pitches:
                    events.append(Event(data=int(pitch.ps),
                                        time=float(note.offset),
                                        duration=float(note.duration.quarterLength)))
            else:
                raise Warning(f"Encountered unknown MIDI stream object {note} (type: {type(note)}) "
                              f"while reading file '{file}'")
    return list(sorted(events, key=lambda e: e.time))


def chordify(piece):
    """
    Create time bins at note events (on- or offset). For each bin create a set of notes that are on.
    :param piece: List of pitch Events
    :return: list of Events (with start time and duration) with note sets
    """
    # create dictionary with time on- and offsets and events starting at a certain onset
    event_dict = {}
    for e in piece:
        # add onset and offset time slot
        if e.time not in event_dict:
            event_dict[e.time] = set()
        if e.time + e.duration not in event_dict:
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
        raise ValueError(f"The last time slot should be empty but it contains: '{event_list[-1][1]}'. "
                         f"This is a bug (maybe due to floating point arithmetics?)")
    # turn dict into an ordered list of events with correct durations and combined event data
    return [Event(time=time, duration=next_time - time, data=frozenset([e.data for e in events]))
            for (time, events), (next_time, next_events) in zip(event_list, event_list[1:])]


def piano_roll(file, min_pitch=None, max_pitch=None, return_range=False, return_durations=False, reader=read):
    # read piece and chordify
    chordified = chordify(reader(file))
    assert len(chordified) > 0, "this piece seems to be empty"
    # get all occuring pitches
    all_pitches = frozenset.union(*[event.data for event in chordified])
    # get actual minimum and maximum pitch
    actual_min_pitch = min(all_pitches)
    actual_max_pitch = max(all_pitches)
    # check against requested values
    if min_pitch is None:
        min_pitch = actual_min_pitch
    else:
        if actual_min_pitch < min_pitch:
            raise ValueError(f"actual minimum pitch ({actual_min_pitch}) is smaller than requested value ({min_pitch}")
    if max_pitch is None:
        max_pitch = actual_max_pitch
    else:
        if actual_max_pitch > max_pitch:
            raise ValueError(f"actual maximum pitch ({actual_max_pitch}) is greater than requested value ({max_pitch}")
    assert max_pitch >= min_pitch  # safety check
    # allocate numpy array of appropriate size
    roll = np.zeros((len(chordified), max_pitch - min_pitch + 1), dtype=np.bool)
    # set multiple-hot for all time slices
    for time_idx, event in enumerate(chordified):
        if len(event.data) == 0:
            continue
        roll[time_idx, np.array(list(event.data)) - min_pitch] = True
    # construct return tuple
    ret = (roll,)
    if return_range:
        ret = ret + (np.arange(min_pitch, max_pitch + 1),)
    if return_durations:
        ret = ret + (np.array([event.duration for event in chordified]),)
    # return
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


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
            pitch_class_counts[time_idx][pitch % 12] += 1
    times = np.array([event.time for event in chordified] + [chordified[-1].time + chordified[-1].duration])
    return pitch_class_counts, times


def pitch_class_counts(file, reader=read):
    """
    Read pitch-class counts from MIDI file by concatenating read_midi, chordify, and pitch_class_counts
    :param file: MIDI file
    :return: 2-tuple of numpy arrays with pitch-class counts and time boundaries
    """
    return pitch_class_counts_from_chordified(chordify(reader(file)))


def get_pitch_scape(file_path, **kwargs):
    """
    Load a pitch scape from file.
    :param file_path: file to load
    :param kwargs: kwargs passed to PitchScape
    :return: PitchScape object
    """
    pitch_counts, times = pitch_class_counts(file_path)
    return PitchScape(values=pitch_counts, times=times, **kwargs)


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


def sample_density(n_time_intervals,
                   file_path=None,
                   scape=None,
                   **kwargs):
    if (file_path is None) == (scape is None):
        raise ValueError("Have to provide exactly one of 'file_path' and 'scape' as arguments")
    if file_path is not None:
        scape = get_pitch_scape(file_path, **kwargs)
    times = np.linspace(scape.min_time, scape.max_time, n_time_intervals + 1)
    samples = None
    for idx, (start, end) in enumerate(zip(times[:-1], times[1:])):
        val = scape[start, end]
        if samples is None:
            samples = np.zeros((n_time_intervals,) + val.shape)
        samples[idx, ...] = val
    return samples


def sample_scape(n_time_intervals,
                 file_path=None,
                 scape=None,
                 **kwargs):
    """
    Sample points from a scape on a uniform grid. The total time [min_time, max_time] is divided into n_time_intervals
    equally sized intervals. A sample for every possible combination of start and end time is collected and returned,
    ordered first by start time and then by end time.
    :param n_time_intervals: the number of time intervals
    :param file_path: path to a file for reading a scape via get_pitch_scape (alternative to passing scape directly)
    :param scape: the scape to sample (alternative to providing file_path)
    :param kwargs: kwargs to pass to get_pitch_scape
    :return: array of samples (first dimension has size n_time_intervals * (n_time_intervals + 1) / 2)
    """
    if (file_path is None) == (scape is None):
        raise ValueError("Have to provide exactly one of 'file_path' and 'scape' as arguments")
    if scape is not None:
        if kwargs:
            warn("Keyword arguments are ignored if you provide 'scape'")
    else:
        scape = get_pitch_scape(file_path, **kwargs)
    times = np.linspace(scape.min_time, scape.max_time, n_time_intervals + 1)
    samples = []
    for start, end in product(times, times):
        if start >= end:
            continue
        samples.append(scape[start, end])
    return np.array(samples)
