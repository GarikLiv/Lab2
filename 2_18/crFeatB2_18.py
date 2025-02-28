import os
import pims
import tifffile
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings
import glob


files = glob.glob('218Mot/BMot/*.tif')
files.sort()
frames = pims.as_grey(pims.ImageSequence(files))
features = tp.batch(frames, diameter=9, minmass=100, separation=14, invert=False,processes=1)
features.to_pickle('frameDataB2_18.pkl')
