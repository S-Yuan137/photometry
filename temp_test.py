import sy_class
import matplotlib.pyplot as plt
import sys
import numpy as np

path = f"C:/Users/Shibo/Desktop/COMAP-sem2/maps/after_astro_cal/fg4_Feeds1-11-14-15-17-18_Band0.fits"
map1 = sy_class.AstroMap(path)
print(sy_class.jackknife(map1, [240,240],20))