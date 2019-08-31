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

WIDTH =  1280
HEIGHT = 720
FRAME_RATE = 29.0
DURATION = 20.0
VID_NM = 'noise.mp4'
AUD_NM = 'beethoven.op132.mvt3.mp3'
OUT_NM = 'out.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(VID_NM, fourcc, FRAME_RATE, (WIDTH, HEIGHT)) 
yellow = np.array([64, 192, 192], dtype=np.uint8)
magenta = np.array([192, 64, 192], dtype=np.uint8)

print('GENERATING VIDEO...')
for i in tqdm.tqdm(range(int(FRAME_RATE * DURATION))):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # draw moving box: 
    frame[200:230 , int(-1-1.7*i)%(WIDTH-100):int(-1-1.7*i)%(WIDTH-100)+90, :] += yellow 

    # draw vertical line: 
    frame[:, int(WIDTH/2):(int(WIDTH/2)+3) , :] += magenta 

    video.write(frame)

video.release()

print('PAIRING with AUDIO...')
os.system('ffmpeg -i {} -i {} -shortest -strict -2 {}'.format(
    VID_NM, AUD_NM, OUT_NM 
))
