import numpy as npy
from Helper import *

class SetParams:
    def __init__(self, title, frame, sdWeight, var=0, d=30):
        self.title = title
        self.value = getRandomValue(frame,sdWeight,var,d)
        self.diff = getNumpyZerosLike(self.value)
        self.m = getNumpyZerosLike(self.value)