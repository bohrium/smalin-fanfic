''' author: samtenka
    change: 2019-11-22
    create: 2019-08-31
    descrp: create audio/video pairing by following posts from
                @enriqueav      medium.com/@enriqueav/881b18e41397
                @VideoCurious   answers.opencv.org/question/35590
                @Rajath         stackoverflow.com/questions/24804928
'''

import numpy as np
import cv2
import tqdm
import sys
import os 
from audio_reader import get_pressures 

from read_trio import read_midi 
notes_by_player = read_midi('mendel.mid')

WIDTH =  1280
HEIGHT = 720
FRAME_RATE = 24.0
DURATION = 80.0
VID_NM = 'temp.mp4'
AUD_NM = 'mendel.mp3'
OUT_NM = 'trio.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(VID_NM, fourcc, FRAME_RATE, (WIDTH, HEIGHT)) 

                    #B   #G   #R
red     = np.array([  0,   0, 255])
orange  = np.array([  0,  64, 255])
yellow  = np.array([  0, 128, 192])
green   = np.array([ 64, 192,  64])
blue    = np.array([192,  64,  64])
purple  = np.array([192,  64, 128])

colors_by_player = {
    'violin': yellow,
    'cello': red,
    'piano lh': green,
    'piano rh': purple,
}
octave_offset_by_player = {
    'violin': 1,
    'cello': 1,
    'piano lh': -1,
    'piano rh': -2,
}

PIXELS_PER_BEAT = 20
BEAT_RATE = 3*90.0/60
PIXELS_PER_SEMITONE = 6
BEATS_IN_ANACRUSIS = 2

START_TIME = 2.333 - BEATS_IN_ANACRUSIS/BEAT_RATE 

def height_from_pitch(pitch):
    return HEIGHT - PIXELS_PER_SEMITONE * pitch 
def width_from_beat(frame_nb, beat): 
    frame_nb -= START_TIME * FRAME_RATE 
    linear = PIXELS_PER_BEAT * (beat - (BEAT_RATE/FRAME_RATE) * frame_nb)
    stretched = linear + 200*np.tanh(linear/100.0) 
    return int(WIDTH//2 + stretched)

def brightness_from_width(w):
     return (1.0 - ((w-WIDTH//2)/float(WIDTH//2))**2.0)**0.5




pressures, audio_frame_rate, duration = get_pressures(AUD_NM)
def get_segment(start, end):
    return pressures[int(audio_frame_rate*start) : int(audio_frame_rate*end), :]
def get_power(start, end):
    return np.mean(np.square(get_segment(start, end)))

first_active_note_idx_by_player = {p:0 for p in notes_by_player}
print('GENERATING VIDEO...')
for frame_nb in tqdm.tqdm(range(int(FRAME_RATE * DURATION))):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    time = frame_nb/FRAME_RATE 

    #p_wide = get_power(max(0, time-0.25), min(duration, time+0.25))
    #p_thin = get_power(max(0, time-0.05), min(duration, time+0.05))
    #is_edge = 1e-4 + 1.2 * (p_thin**0.5) < p_wide**0.5

    #if is_edge:
    #    frame[200:300, 200:300, 0] = 255
    #frame[100:200, 100:200, 1] = 0

    # draw moving box: 
    for p in notes_by_player:
        for note in notes_by_player[p][first_active_note_idx_by_player[p]:]:
            w_start = width_from_beat(frame_nb, note.start_beat)
            w_end   = width_from_beat(frame_nb, note.end_beat)
            if not (0 <= w_start):
                first_active_note_idx_by_player[p] += 1
                continue
            if not (w_end < WIDTH):
                break

            h = height_from_pitch(note.pitch + 12 * octave_offset_by_player[p])
            brightness = 0.7 * min(brightness_from_width(w_start), brightness_from_width(w_end))
            if w_start < WIDTH//2 < w_end:
                brightness = 1.3
                #if abs( 0.5 - (w_end - WIDTH//2)/float(w_end-w_start) ) > 0.45 : 
                #    frame[100:200, 100:200, 1] = 255
            hh = PIXELS_PER_SEMITONE
            frame[h-hh:h+hh, w_start:w_end , :] = (
                np.minimum(255, colors_by_player[p] * brightness)
            ).astype(np.uint8)


    # draw vertical line: 
    frame[:, (WIDTH//2 - 1):(WIDTH//2 + 1) , :] = green 

    video.write(frame)

video.release()

print('PAIRING with AUDIO...')
os.system('ffmpeg -i {} -i {} -shortest -strict -2 {}'.format(
    VID_NM, AUD_NM, OUT_NM 
))
