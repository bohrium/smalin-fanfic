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
DURATION = 65.0
VID_NM = 'temp.mp4'
AUD_NM = 'mendel.mp3'
OUT_NM = 'trio.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(VID_NM, fourcc, FRAME_RATE, (WIDTH, HEIGHT)) 

T_MINUS = -10.0
dt = 0.01 
#   real times, anim times
distortions = np.array((
    ( T_MINUS , T_MINUS ),
    (     0.0 ,  0.0    ),
    (    11.0 , 10.8    ), # give time    (do if anim is otherwise early)
    (    20.0 , 20.2    ),
    (    20.0 , 20.2    ),
    (    22.0 , 22.6    ), # take time    (do if anim is otherwise late)
    (    24.0 , 24.0    ),
    (    27.0 , 27.5    ),
    (    30.0 , 30.2    ),
    (    34.0 , 34.2    ),
    (    37.0 , 36.6    ),
    (    41.0 , 40.9    ),
    (    43.0 , 43.4    ),
    (    47.0 , 47.3    ),
    (    50.0 , 49.8    ),
    (    53.0 , 53.2    ),
    (    55.0 , 55.2    ),
    (    56.5 , 56.4    ),
    (    57.2 , 57.3    ),
    (    58.0 , 58.0    ),
    (    60.0 , 60.0    ),
    (    65.0 , 65.0    ),
    (    65.7 ,165.0    ),
    (    66.3 ,225.0    ),
    (    67.0 ,323.0    ),
    (    87.0 ,343.0    ),
))
linear_interp = np.interp(np.arange(T_MINUS, DURATION, dt), distortions[:,0], distortions[:,1]) 

def anim_time(t):
    quot = int((t-T_MINUS)/dt)
    return linear_interp[quot] + (t-T_MINUS - quot*dt)

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
thickness_by_player = {
    'violin': 0.7,
    'cello': 1.6,
    'piano lh': 1.3,
    'piano rh': 1.0,
}
curve_intensity_by_player = {
    'violin': 2.0,
    'cello': 2.0,
    'piano lh': 0.0,
    'piano rh': 0.0,
}
narrowness_by_player = {
    'violin': 1.0,
    'cello': 1.0,
    'piano lh':10.0,
    'piano rh':10.0,
}

PIXELS_PER_BEAT = 20
BEAT_RATE = 3*90.0/60
PIXELS_PER_SEMITONE = 6
CURVE_SCALE = 100.0
BEATS_IN_ANACRUSIS = 2

#START_TIME = 2.333 - BEATS_IN_ANACRUSIS/BEAT_RATE 
START_TIME =  2.333 - BEATS_IN_ANACRUSIS/BEAT_RATE 

def height_from_pitch(pitch):
    return HEIGHT - PIXELS_PER_SEMITONE * pitch 
def width_from_beat(frame_nb, beat, curve_intensity=2.0): 
    real_time = anim_time(frame_nb/float(FRAME_RATE) - START_TIME)
    linear = PIXELS_PER_BEAT * (beat - BEAT_RATE * real_time)
    stretched = linear + curve_intensity * CURVE_SCALE * np.tanh(linear/CURVE_SCALE)
    return int(WIDTH//2 + stretched)

    #frame_nb -= START_TIME * FRAME_RATE 
    #linear = PIXELS_PER_BEAT * (beat - (BEAT_RATE/FRAME_RATE) * frame_nb)
    #stretched = linear + curve_intensity * CURVE_SCALE * np.tanh(linear/CURVE_SCALE)
    #return int(WIDTH//2 + stretched)

def brightness_from_width(w, narrowness):
     return (1.0 - ((w-WIDTH//2)/float(WIDTH//2))**2.0)**(narrowness*0.5)




pressures, audio_frame_rate, duration = get_pressures(AUD_NM)
def get_segment(start, end):
    return pressures[int(audio_frame_rate*start) : int(audio_frame_rate*end), :]
def get_power(start, end):
    return np.mean(np.square(get_segment(start, end)))

first_active_note_idx_by_player = {p:0 for p in notes_by_player}
print('GENERATING VIDEO...')
for frame_nb in tqdm.tqdm(range(int(FRAME_RATE * DURATION))):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # draw moving box: 
    for p in ('piano rh', 'piano lh', 'cello', 'violin'):
        for note in notes_by_player[p][first_active_note_idx_by_player[p]:]:
            w_start = width_from_beat(frame_nb, note.start_beat, curve_intensity_by_player[p])
            w_end   = width_from_beat(frame_nb, note.end_beat, curve_intensity_by_player[p])
            if not (0 <= w_start):
                first_active_note_idx_by_player[p] += 1
                continue
            if not (w_end < WIDTH):
                break

            h = height_from_pitch(note.pitch + 12 * octave_offset_by_player[p])
            nn = narrowness_by_player[p]
            brightness = 0.7 * min(brightness_from_width(w_start, nn), brightness_from_width(w_end, nn))
            if w_start < WIDTH//2 < w_end:
                brightness = 1.3
            hh = int(thickness_by_player[p] * PIXELS_PER_SEMITONE)
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
