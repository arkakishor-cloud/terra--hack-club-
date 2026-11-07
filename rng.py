import numpy as np
import matplotlib.pyplot as plt

def noise (width, height, scale=12, seed=12):
    rng= np.random.default_rng(seed)

    #small world
    coarse= rng.random((scale,scale))

    #coordinates
    x=np.linspace(0, scale-1,width)
    y=np.linspace(0,scale-1,height)

    xx , yy = np.meshgrid(x,y)

    #neighbouring coordinates
    x0=np.floor(xx).astype(int)
    y0=np.floor(yy).astype(int)

    x1=np.minimum(x0+1, scale-1)
    y1=np.minimum(y0+1, scale-1)

    #interpolation weights
    dx=xx-x0
    dy=yy-y0

    a=coarse[y0,x0]
    b=coarse[y0,x1]
    c=coarse[y1,x0]
    d=coarse[y1,x1]

    #bilinear interpolation
    top=a*(1-dx)+b*dx
    