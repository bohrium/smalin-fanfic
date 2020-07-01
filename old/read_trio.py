''' author: samtenka
    change: 2019-11-22
    create: 2019-08-31
    descrp: parse piano trio midi file
'''

from utils import CC
from mido import MidiFile
from collections import namedtuple

Note = namedtuple('Note', 'start_beat end_beat pitch volume') 

def pitch_name(pitch):
    #return 'C C# D D# E F F# G G# A A# B'.split()[pitch % 12]
    return 'C Db D Eb E F Gb G Ab A Bb B'.split()[pitch % 12]

quartet = {
    'violin': 'Staff',
    'cello': 'Staff-1',
    'piano lh': 'Staff-2',
    'piano rh': 'Staff-3'
}

def read_midi(filenm, ticks_per_beat=192 * (60.0/80), beats_in_anacrusis=1):
    track_idx_by_player = {p:None for p in quartet}
    notes_by_player = {p:[] for p in quartet}

    # 0. match file's tracks to quartet quartet 
    mi = MidiFile(filenm)
    for idx, track in enumerate(mi.tracks):
        for p in quartet: 
            if quartet[p] == track.name:
                track_idx_by_player[p] = idx
                break
        else: continue
        print(CC + '@C track {}:`@M {}@C ` (player @M {}@C ) has @Y {} @C messages'.format(
             idx, track.name, p, len(track)
        ))
    for p in quartet:
        assert track_idx_by_player[p] is not None
    print()

    # 1. read each player's notes
    for p in quartet:
        print(CC + 'parsing @M {} @C '.format(p))
    
        ticks_elapsed = ticks_per_beat * (-beats_in_anacrusis)
    
        note_starts = {}
        for msg in mi.tracks[track_idx_by_player[p]]:
            assert msg.type != 'note_off'
            if msg.type != 'note_on': continue
            #assert msg.channel == track_idx_by_player[p]
    
            ticks_elapsed += msg.time
            if msg.velocity==0 and msg.note in note_starts:
                notes_by_player[p].append(Note(
                    float(note_starts[msg.note][0])/ticks_per_beat,
                    float(ticks_elapsed)/ticks_per_beat,
                    int(msg.note),
                    note_starts[msg.note][1]/127.0,
                ))
                del note_starts[msg.note]
                continue
            if msg.note not in note_starts:
                note_starts[msg.note] = (ticks_elapsed, msg.velocity)

    return notes_by_player

if __name__=='__main__':
    notes_by_player = read_midi('mendel.mid')
    for note, _ in zip(notes_by_player['cello'], range(30)):
        print(CC + 'time @B {:6.1f}@C  - @B {:6.1f}@C \t pitch @R {} @C == @R {:2s} @C \t volume @G {:.2f} @C '.format(
            note.start_beat,
            note.end_beat,
            note.pitch,
            pitch_name(note.pitch),
            note.volume,
        ))

