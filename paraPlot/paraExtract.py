import h5py
from astropy.time import Time
import numpy as np

class ObsId(object):
    def __init__(self, name, datadir,): 
        self.name = name
        self.datadir = datadir
