''' author: samtenka
    change: 2019-08-31
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

from read_quartet import read_midi 
notes_by_player = read_midi('beethoven.op132.mvt3.mid')

WIDTH =  1280
HEIGHT = 720
FRAME_RATE = 24.0
DURATION = 300.0
VID_NM = 'temp.mp4'
AUD_NM = 'beethoven.op132.mvt3.mp3'
OUT_NM = 'quartet.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(VID_NM, fourcc, FRAME_RATE, (WIDTH, HEIGHT)) 

green   = np.array([ 64, 192,  64])
yellow  = np.array([ 64, 192, 192])
orange  = np.array([ 64, 128, 192])
red     = np.array([ 64,  64, 192])
purple  = np.array([128,  64, 192])

colors_by_player = {
    'violin a': yellow,
    'violin b': orange,
    'viola': red,
    'cello': purple,
}

PIXELS_PER_BEAT = 90
BEAT_RATE = 0.7
PIXELS_PER_SEMITONE = 6

START_TIME = 11.0 + 2.0/BEAT_RATE 

def height_from_pitch(pitch):
    return HEIGHT - PIXELS_PER_SEMITONE * pitch 
def width_from_beat(frame_nb, beat): 
    frame_nb -= START_TIME * FRAME_RATE 
    linear = PIXELS_PER_BEAT * (beat - (BEAT_RATE/FRAME_RATE) * frame_nb)
    stretched = linear + 200*np.tanh(linear/100.0) 
    return int(WIDTH//2 + stretched)

def brightness_from_width(w):
     return (1.0 - ((w-WIDTH//2)/float(WIDTH//2))**2.0)**0.5

first_active_note_idx_by_player = {p:0 for p in notes_by_player}

print('GENERATING VIDEO...')
for frame_nb in tqdm.tqdm(range(int(FRAME_RATE * DURATION))):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

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

            h = height_from_pitch(note.pitch)
            brightness = 0.7 * min(brightness_from_width(w_start), brightness_from_width(w_end))
            if w_start < WIDTH//2 < w_end:
                brightness = 1.3
            frame[h:h+(2*PIXELS_PER_SEMITONE) , w_start:w_end , :] = (
                colors_by_player[p] * brightness
            ).astype(np.uint8)

    # draw vertical line: 
    frame[:, (WIDTH//2 - 1):(WIDTH//2 + 1) , :] = green 

    video.write(frame)

video.release()

print('PAIRING with AUDIO...')
os.system('ffmpeg -i {} -i {} -shortest -strict -2 {}'.format(
    VID_NM, AUD_NM, OUT_NM 
))
