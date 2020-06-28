''' author: samtenka
    change: 2020-06-28
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

from read_aria import read_midi 
notes_by_player = read_midi('music/bach.007.04.mid')

WIDTH =  1280
HEIGHT = 720
FRAME_RATE = 24.0
DURATION =  80.0
VID_NM = 'temp.mp4'
AUD_NM = 'music/bach.007.04.mp3'
OUT_NM = 'aria.mp4'

#

PIXELS_PER_BEAT = 20
BEAT_RATE = 126.0/60
PIXELS_PER_SEMITONE = 6
CURVE_SCALE = 100.0
BEATS_IN_ANACRUSIS = 0
START_TIME = 1.25

#

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(VID_NM, fourcc, FRAME_RATE, (WIDTH, HEIGHT)) 

T_MINUS = -START_TIME
dt = 0.01 
#   real times, anim times
distortions = np.array((
    (T_MINUS,T_MINUS),
    (     0.0 ,   0.0   ),
    (    20.0 ,  20.0   ),
    (    40.0 ,  40.0   ),
    (    80.0 ,  80.0   ),
    (   160.0 , 160.0   ),
    (   240.0 , 240.0   ),
    (   320.0 , 320.0   ),
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
    'v1': yellow,
    'v2': orange,
    'cl': red,
    'tn': green,
    'pl': blue, 
    'pr': purple,
}
octave_offset_by_player = {
    'v1': 1,
    'v2': 1,
    'cl': 0,
    'tn': 0,
    'pl': -2,
    'pr': -3,
}
thickness_by_player = {
    'v1': 0.7,
    'v2': 0.7,
    'cl': 0.7,
    'tn': 1.6,
    'pl': 1.0,
    'pr': 1.0,
}
curve_intensity_by_player = {
    'v1': 1.0,
    'v2': 1.0,
    'cl': 1.0,
    'tn': 3.0,
    'pl': 0.0,
    'pr': 0.0,
}
narrowness_by_player = {
    'v1':10.0,
    'v2':10.0,
    'cl':10.0,
    'tn': 1.0,
    'pl':10.0,
    'pr':10.0,
}

def height_from_pitch(pitch):
    return HEIGHT - PIXELS_PER_SEMITONE * pitch 
def width_from_beat(frame_nb, beat, curve_intensity=2.0): 
    real_time = anim_time(frame_nb/float(FRAME_RATE) - START_TIME)
    linear = PIXELS_PER_BEAT * (beat - BEAT_RATE * real_time)
    stretched = linear + curve_intensity * CURVE_SCALE * np.tanh(linear/CURVE_SCALE)
    return int(WIDTH//2 + stretched)

def beat_from_width(frame_nb, width): 
    real_time = anim_time(frame_nb/float(FRAME_RATE) - START_TIME)
    stretched = width-WIDTH//2
    return stretched/PIXELS_PER_BEAT + BEAT_RATE * real_time  

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
    time = frame_nb/float(FRAME_RATE) - START_TIME

    frame = cv2.putText(img=frame, text='real time {:.2f}'.format(time),
            org=(10,40), fontFace=0, fontScale=1,
            color=(64,64,64), thickness=1)

    frame = cv2.putText(img=frame, text='anim time {:.2f}'.format(anim_time(time)),
            org=(10,80), fontFace=0, fontScale=1,
            color=(64,64,64), thickness=1)

    frame = cv2.putText(img=frame, text='beat {:.2f}'.format(beat_from_width(frame_nb, WIDTH//2)),
            org=(10,120), fontFace=0, fontScale=1,
            color=(64,64,64), thickness=1)

    # draw moving box: 
    for p in ('pr', 'pl', 'cl', 'v1', 'v2', 'tn'):
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
