''' author: samtenka
    change: 2019-08-31
    create: 2019-08-31
    descrp: read numpy array of pressures from mp3 
'''

from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
import os
from utils import CC


def get_pressures(AUD_NM):
    WAV_NM = (AUD_NM+'*').replace('.mp3*', '.wav')
    
    if not os.path.isfile(WAV_NM):
        os.system('ffmpeg -i {} {}'.format(
            AUD_NM, WAV_NM
        ))
    
    frame_rate, pressures = wavfile.read(WAV_NM)
    
    assert pressures.shape[1]==2, 'expected stereo waveform!'
    pressures = pressures.astype(np.float32) / 2.0**15 
    assert (
        ( 0.5 <= np.max(pressures) <=  1.0) and
        (-1.0 <= np.min(pressures) <= -0.5)
    )
    nb_frames, _ = pressures.shape
    duration = float(nb_frames)/frame_rate
    print(CC + 'audio `@M {}@C ` endures @G {:.2f} @C seconds'.format(
        AUD_NM, duration 
    ))
    return pressures, frame_rate, duration 

if __name__=='__main__':
    AUD_NM = 'beethoven.op132.mvt3.mp3' 
    pressures, frame_rate, duration = get_pressures(AUD_NM)

    def get_segment(start, end):
        return pressures[int(frame_rate*start) : int(frame_rate*end), :]
    
    def get_power(start, end):
        return np.mean(np.square(get_segment(start, end)))
    
    for t in np.arange(10.0, 30.0, 0.1):
        p_wide = get_power(t-0.25, t+0.25)
        p_thin = get_power(t-0.05, t+0.05)
        print(CC + 'at time @G {:8.2f}@C , power is @R {:8.2e}@C :\t@R {}@Y {} @G {} @C '.format(
            t, p_wide,
            '*' * int(min(100, 1000 * (p_wide**0.5))),
            '*' * (int(min(100, 1000 * 1.2 * (p_thin**0.5))) - int(min(100, 1000 * (p_wide**0.5)))),
            '!'*4 if 1.2 * (p_thin**0.5) < 1e-4 + p_wide**0.5 else ''
        ))


