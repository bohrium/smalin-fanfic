''' author: samtenka
    change: 2019-08-31
    create: 2019-08-31
    descrp: parse string quartet midi file
'''

from utils import CC
from mido import MidiFile
from collections import namedtuple

Note = namedtuple('Note', 'start_beat end_beat pitch') 

def pitch_name(pitch):
    return 'C C# D D# E F F# G G# A A# B'.split()[pitch % 12]

quartet = {
    'violin a': 'violino i.',
    'violin b': 'violino ii.',
    'viola': 'viola.',
    'cello': 'violoncello.'
}

def read_midi(filenm, ticks_per_beat=192, beats_in_anacrusis=2):
    track_idx_by_player = {p:None for p in quartet}
    notes_by_player = {p:[] for p in quartet}
    
    # 0. match file's tracks to quartet quartet 
    mi = MidiFile(filenm)
    for idx, track in enumerate(mi.tracks):
        for p in quartet: 
            if quartet[p] == track.name.lower():
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
            assert msg.channel == track_idx_by_player[p]
    
            ticks_elapsed += msg.time
            if msg.velocity==0:
                assert msg.note in note_starts
                notes_by_player[p].append(Note(
                    float(note_starts[msg.note])/ticks_per_beat,
                    float(ticks_elapsed)/ticks_per_beat,
                    int(msg.note)
                ))
                del note_starts[msg.note]
                continue
            assert msg.note not in note_starts
            note_starts[msg.note] = ticks_elapsed

    return notes_by_player

if __name__=='__main__':
    notes_by_player = read_midi('beethoven.op132.mvt3.mid')
    for note, _ in zip(notes_by_player['violin a'], range(30)):
        print(CC + 'time @B {:6.1f}@C  - @B {:6.1f}@C \t pitch @R {} @C == @R {} @C '.format(
            note.start_beat,
            note.end_beat,
            note.pitch,
            pitch_name(note.pitch),
        ))

