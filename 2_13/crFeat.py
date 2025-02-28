import os
import pims
import tifffile
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings
import glob

def checkFrames(frames):
    for i,frame in enumerate(frames):
        if i%2 == 0:
            alstFr = frame
            if(i != 0):
                if(blstFr.shape != frame.shape and blstFr.dtype != frame.dytpe):
                    print(frame.shape, frame.dtype)
        if i%2 == 1:
            blstFr = frame
            if(alstFr.shape != frame.shape and alstFr.dtype != frame.dytpe):
                print(frame.shape, frame.dtype)
    
files = glob.glob('ABrMotion/BrMot/*.tif')
files.sort()
print(files)

#frames = pims.as_grey(pims.open('ABrMotion/BrMot/*.tif'))
frames = pims.as_grey(pims.ImageSequence(files))
#checkFrames(frames)
#frames1 = pims.TiffStack("CMotion2_13.tif")
#frames2 = pims.TiffStack("CMotion2_13_1.tif")
#f = tp.locate(frames[0], diameter=9, minmass=100, separation=14)
#"tp.annotate(f, frames[0])
#tp.quiet(suppress=True)
features = tp.batch(frames, diameter=9, minmass=100, separation=14, invert=False,processes=1)
features.to_pickle('frameData.pkl')

