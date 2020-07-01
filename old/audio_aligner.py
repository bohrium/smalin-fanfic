''' author: samtenka
    change: 2019-08-31
    create: 2019-08-31
    descrp: align midi and audio timescales by small perturbations 
'''

import numpy as np
import cv2
import tqdm
import sys
import os 
from audio_reader import get_pressures 
from read_quartet import read_midi 

notes_by_player = read_midi('beethoven.op132.mvt3.mid')

AUD_NM = 'beethoven.op132.mvt3.mp3'
APPROX_BEAT_RATE = 0.68
ANACRUSIS = 2
START_TIME = 11.0 + ANACRUSIS/APPROX_BEAT_RATE

pressures, audio_frame_rate, audio_duration = get_pressures(AUD_NM)
def get_segment(start, end):
    return pressures[int(audio_frame_rate*start) : int(audio_frame_rate*end), :]
def get_power(start, end):
    return np.mean(np.square(get_segment(start, end)))
def is_edge(time):
    p_wide = get_power(max(0, time-0.25), min(audio_duration, time+0.25))
    p_thin = get_power(max(0, time-0.05), min(audio_duration, time+0.05))
    return ((1e-4 + 1.2 * (p_thin**0.5)) < p_wide**0.5)

dt = 1.0/24
times = np.arange(9.0, 13.0, dt)
print(''.join('|' if (t-int(t))<=dt else ' ' for t in times[2:-2]))

edge_flags_wave = np.zeros(times.shape, dtype=np.float32)
for i, t in enumerate(times):
    edge_flags_wave[i] = 1.0 if is_edge(t) else 0.0
edge_flags_wave = (edge_flags_wave[1:] + edge_flags_wave[:-1])/2 
edge_flags_wave = (edge_flags_wave[1:] + edge_flags_wave[:-1])/2 
print(''.join('{:d}'.format(int(4*e)) if e else ' ' for e in edge_flags_wave))

A, B, d = -2.7, -0.7, dt*APPROX_BEAT_RATE 
beats = np.arange(A, B, d) 
edge_flags_midi = np.zeros(beats.shape, dtype=np.float32)
for p in notes_by_player:
    for n in notes_by_player[p]:
        for edge in (n.start_beat, n.end_beat):
            if not (A < edge < B): continue
            edge_flags_midi [int((edge-A)/d)] = 1.0
edge_flags_midi = (edge_flags_midi[1:] + edge_flags_midi[:-1])/2 
edge_flags_midi = (edge_flags_midi[1:] + edge_flags_midi[:-1])/2 
print(''.join('{:d}'.format(int(4*e)) if e else ' ' for e in edge_flags_midi))
