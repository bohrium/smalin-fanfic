''' author: samtenka
    change: 2020-06-29
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

from skimage.draw import circle

from read_aria import read_midi 
notes_by_player = read_midi('music/bach.007.04.mid')

WIDTH =  1280
HEIGHT = 720
FRAME_RATE = 24.0
DURATION = 10#215.0
VID_NM = 'temp.mp4'
AUD_NM = 'music/bach.007.04.mp3'
OUT_NM = 'aria.mp4'

#

PIXELS_PER_BEAT = 120#40
BEAT_RATE = 125.2/60
PIXELS_PER_SEMITONE = 6
CURVE_SCALE = 100.0
BEATS_IN_ANACRUSIS = 0
START_TIME = 1.22

#

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(VID_NM, fourcc, FRAME_RATE, (WIDTH, HEIGHT)) 

T_MINUS = -START_TIME
dt = 0.01 
#   real times, anim times
distortions = np.array((
    (T_MINUS,T_MINUS),
    (     0.0 ,   0.0   ),
    (    30.0 ,  30.3   ),
    (    56.0 ,  56.1   ),
    (    80.0 ,  80.3   ),
    (    95.0 ,  95.1   ),
    (    99.0 ,  99.2   ),
    (   113.0 , 113.1   ),
    (   135.4 , 135.0   ),
    (   155.4 , 155.0   ),
    (   175.3 , 175.0   ),
    (   210.0 , 210.1   ),
    (   225.0 , 225.0   ),
))
linear_interp = np.interp(np.arange(T_MINUS, DURATION, dt), distortions[:,0], distortions[:,1]) 

#   seem beats, phys beats
B_MINUS = -1.0
db = 0.01
beat_distortions = {
    'tn':np.array(((B_MINUS  , B_MINUS ), (100.0, 100.0))),
    'cl':np.array((
        (B_MINUS,B_MINUS),
        (   0.0 ,   0.0 ),
        (   2.0 ,   1.9 ),
        (   3.0 ,   2.95),
        (   4.0 ,   4.05),
        (   5.0 ,   5.00),
        (   6.0 ,   6.0 ),
        (   7.67,   7.70),
        (  10.0 ,  10.0 ),
        (  11.0 ,  10.9 ),
        (  14.0 ,  14.0 ),
        (  14.67,  14.75),
        (  15.0 ,  15.0 ),
        (  17.0 ,  16.9 ),
        (  28.0 ,  28.0 ),
        (  29.0 ,  28.9 ),
        (  37.67,  37.50),
        ( 100.0 , 100.0 ),
    )),
    'v1':np.array((
        (B_MINUS  , B_MINUS ),
        (     0.0 ,   0.0   -0.03),
        (     0.67,   0.60  -0.03),
        (     1.0 ,   1.0   -0.03),
        (     1.67,   1.63  -0.03),
        (     2.0 ,   2.0   -0.03),
        (     2.33,   2.27  -0.03),
        (     3.33,   3.27  -0.03),
        (     6.0 ,   5.95  -0.03),
        (     8.0 ,   7.95  -0.03),
        (     9.0 ,   8.9   -0.03),
        (    10.0 ,   9.9   -0.03),
        (    12.0 ,  11.9   -0.03),
        (    15.0 ,  14.95  -0.03),
        (    17.0 ,  17.05  -0.03),
        (    17.33,  17.40  -0.03),
        (    17.67,  17.60  -0.03),
        (    19.0 ,  18.95  -0.03),
        (    33.0 ,  32.95  -0.03),
        (    33.67,  33.57  -0.03),
        (    34.33,  34.40  -0.03),
        (    35.33,  35.27  -0.03),
        (    37.33,  37.40  -0.03),
        (   100.0 , 100.0   -0.03),
    )),
}
beat_distortions['pr'] = beat_distortions['cl']
beat_distortions['pl'] = beat_distortions['cl']
beat_distortions['v2'] = beat_distortions['v1']
beat_linear_interp = {
    k:np.interp(np.arange(B_MINUS, 5*DURATION*BEAT_RATE, db), v[:,0], v[:,1]) 
    for k,v in beat_distortions.items()
}

def anim_time(t):
    quot = int((t-T_MINUS)/dt)
    return linear_interp[quot] + (t-T_MINUS - quot*dt)

def phys_beat(b,p):
    quot = int(float(b-B_MINUS)/db)
    return float(beat_linear_interp[p][quot] + (b-B_MINUS - quot*db)*(beat_linear_interp[p][quot+1]-beat_linear_interp[p][quot]))

                    #B   #G   #R
red     = np.array([  0,   0, 255])
red_    = np.array([  0,  32, 255])
orange  = np.array([  0,  64, 255])
orange_ = np.array([  0,  96, 224])
yellow  = np.array([  0, 128, 192])
yellow_ = np.array([ 32, 160, 128])
green   = np.array([ 64, 192,  64])
green_  = np.array([128, 128,  64])
blue    = np.array([192,  64,  64])
blue_   = np.array([192,  64,  96])
purple  = np.array([192,  64, 128])
purple_ = np.array([108,  48, 212])

colors = [red     ,
          red_    ,
          orange  ,
          orange_ ,
          yellow  ,
          yellow_ ,
          green   ,
          green_  ,
          blue    ,
          blue_   ,
          purple  ,
          purple_ , 
]

colors_by_player = {
    'v1': yellow,
    'v2': orange,
    'tn': green,
    'pr': blue,
    'pl': purple, 
    'cl': red,
}
octave_offset_by_player = {
    'v1': 1,
    'v2': 1,
    'tn': 0,
    'pr':-4, 
    'pl':-1, 
    'cl':-1,
}
thickness_by_player = {
    'v1': 0.7,
    'v2': 0.3,
    'tn': 1.6,
    'pr': 1.0,
    'pl': 1.0,
    'cl': 0.7,
}
curve_intensity_by_player = {
    'v1': 0.0,# 1.0,
    'v2': 0.0,# 1.0,
    'tn': 0.0,# 3.0,
    'pr': 0.0,# 0.0,
    'pl': 0.0,#-0.5,
    'cl': 0.0,# 1.0,
}
narrowness_by_player = {
    'v1':1.0,# 5.0,
    'v2':1.0,# 5.0,
    'tn':1.0,# 1.0,
    'pr':1.0,#25.0,
    'pl':1.0,# 5.0,
    'cl':1.0,#25.0,
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

    for pos, string in {
         40: 'real time {:.2f}'.format(time),
         80: 'anim time {:.2f}'.format(anim_time(time)),
        120: 'beat {:.2f}'.format(beat_from_width(frame_nb, WIDTH//2)),
    }.items(): 
        frame = cv2.putText(text=string,
                            img=frame, org=(10,pos), thickness=2,  
                            fontFace=0, fontScale=1, color=(64,64,64))

    # draw moving box: 
    for p in ('pr', 'pl', 'cl', 'v1', 'v2', 'tn'):
        relevant_notes = notes_by_player[p][first_active_note_idx_by_player[p]:]
        for ddb in range(-10, +10):
            b = int(beat_from_width(frame_nb, WIDTH//2))
            w = width_from_beat(frame_nb, b+ddb, curve_intensity_by_player[p])
            if 0<=w<WIDTH:
                dw = 3 if (b+ddb)%3 == 0 else 0
                frame[:, w-dw:w+dw , :] = green 

        for ii in range(len(relevant_notes)):
            note = relevant_notes[ii]
            next_note = relevant_notes[ii+1] if ii+1<len(relevant_notes) else None

            w_start = width_from_beat(frame_nb, phys_beat(note.start_beat, p), curve_intensity_by_player[p])
            w_end   = width_from_beat(frame_nb, phys_beat(note.end_beat  , p), curve_intensity_by_player[p])

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

            ## instrumental:
            #cc = lambda b: np.minimum(255, colors_by_player[p] * brightness)
            # harmonic:
            cc = lambda b: np.minimum(255, colors[(note.pitch*5)%12] * brightness * b)
            ## brightness:
            #cc = lambda b: np.minimum(255, colors[(note.pitch)%12] * brightness * b)

            if p in ('tn',):
                # lilting:
                #try:
                for g,b in [(1.0,0.8), (0.8,1.0)]:
                    for t in list(np.arange(0.0, 1.01, 0.04))+list(np.sqrt(np.arange(0.5, 1.01, 0.04))):
                        fac = 1.0-max(0.0, min(1.0,
                            0.5*(note.end_beat - note.start_beat)*
                            PIXELS_PER_BEAT/(0.1+abs(w_end-WIDTH//2))
                            if w_start < WIDTH//2 else 0.0
                        ))

                        w_ran = (w_end - w_start)

                        fac_b = max(0.0, min(1.0,
                            0.5*(note.end_beat - note.start_beat)*
                            PIXELS_PER_BEAT/(0.1+abs(w_start+t*w_ran-WIDTH//2))
                        ))

                        tt = frame_nb/30.0
                        dd = 0.5 * (2.0-fac_b) * 6.0 #* (8.5 + np.sin(2*3.14159 * tt) - np.sin(2*3.14159 * 3*tt))

                        if next_note is not None and next_note.start_beat < 0.25 + note.end_beat: 
                            h2 = height_from_pitch(next_note.pitch + 12 * octave_offset_by_player[p])
                        else:
                            h2 = h

                        h_start = h
                        h_ran = h2-h

                        #rrr, ccc = circle(int(h_start+(0.1*fac*t + (1-fac)*t*t*t)*h_ran), int(w_start+t*w_ran), int(dd))
                        #frame[rrr, ccc, :] = cc(1.0).astype(np.uint8)

                        frame[
                            int(h_start+(0.1*fac*t + (1-fac)*t*t*t)*h_ran-g*dd):int(h_start+(0.1*fac*t + (1-fac)*t*t*t)*h_ran+g*dd),
                            int(w_start+dd+t*(w_ran-2*dd)-g*dd):int(w_start+dd+t*(w_ran-2*dd)+g*dd),
                            :
                        ] = cc(b).astype(np.uint8)
                #except:
                #    pass


            if p in ('v1', 'v2'):
                # appearing rectangle:
                if True:#w_start < WIDTH//2:
                    for g,b in [(1.0,0.8), (0.8,1.0)]:
                        w_mid = (w_end + w_start)/2.0
                        w_dif = (w_end - w_start)/2.0

                        frame[int(h-g*hh):int(h+g*hh), int(w_mid-g*w_dif):int(w_mid+g*w_dif), :] = (
                            cc(b).astype(np.uint8)
                        )

            if p in ():
                # galloping rectangle:
                for g,b in [(1.0,0.8), (0.8,1.0)]:
                    w_mid = (w_end + w_start)/2.0
                    w_dif = (w_end - w_start)/2.0

                    g *= max(0.4, min(1.0, 1.5*(note.end_beat - note.start_beat)*PIXELS_PER_BEAT/(
                        0.1+abs(w_start-WIDTH//2)
                    )) if w_start < WIDTH//2 else 0.4)

                    frame[int(h-g*hh):int(h+g*hh), int(w_mid-g*w_dif):int(w_mid+g*w_dif), :] = (
                        cc(b).astype(np.uint8)
                    )

            if p in ('pr','pl', 'cl'):
                # hollow rectangle:
                for g,b in [(1.0,1.0), (0.85,0.75), (0.7,0.5), (0.55, 0.25), (0.4,0.0)]:
                    w_mid = (w_end + w_start)/2.0
                    w_dif = (w_end - w_start)/2.0
                    frame[int(h-g*hh):int(h+g*hh), int(w_mid-g*w_dif):int(w_mid+g*w_dif), :] = (
                        cc(b).astype(np.uint8)
                    )

            if p in ():
                # fuzzy rectangle:
                for g,b in [(1.0,0.1), (0.9,0.2), (0.8,0.4), (0.7,0.6), (0.6,0.8), (0.5,0.9), (0.4,1.0)]: 
                    w_mid = (w_end + w_start)/2.0
                    w_dif = (w_end - w_start)/2.0
                    frame[int(h-g*hh):int(h+g*hh), int(w_mid-g*w_dif):int(w_mid+g*w_dif), :] = np.maximum(
                        frame[int(h-g*hh):int(h+g*hh), int(w_mid-g*w_dif):int(w_mid+g*w_dif), :],
                        cc(b).astype(np.uint8)
                    )

    # draw vertical line: 
    #frame[:, (WIDTH//2 - 1):(WIDTH//2 + 1) , :] = purple

    video.write(frame)

video.release()

print('PAIRING with AUDIO...')
os.system('ffmpeg -i {} -i {} -shortest -strict -2 {}'.format(
    VID_NM, AUD_NM, OUT_NM 
))
