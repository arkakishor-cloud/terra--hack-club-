import numpy as np
import matplotlib.pyplot as plt

def noise (width, height, scale=12, seed=12):
    rng= np.random.default_rng(seed)

    #small world
    coarse= rng.random((scale,scale))

    #coordinates
    x=np.linspace(0, scale-1,width)
    y=np.linspace(0,scale-1,height)
    