import numpy as np
import matplotlib.pyplot as plt

def gen_noise (width, height, scale=12, seed=100):
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
    bottom=c*(1-dy)+d*dy
    noise = top*(1-dy)+bottom*dy
    return noise

def gen_fbm(width, height, octaves=5,persistence=0.5,lacunarity=2,base_scale=4,seed=42):
    total=np.zeros((height,width))
    amplitude=1.0
    max_amplitude=0.0

    for octave in range(octaves):
        scale=int(base_scale*(lacunarity**octave))
        noise= gen_noise(width,height,scale=scale,seed=seed+octave)
        total+=noise*amplitude
        max_amplitude+=amplitude
        amplitude*=persistence
    return total/max_amplitude


#generation
terrain=gen_fbm(2500,
                2500,
                octaves=10,
                persistence=0.8,
                lacunarity=2,
                base_scale=4)

#display
plt.figure(figsize=(8,8))
plt.imshow(terrain,cmap='terrain')
plt.colorbar(label="Elevation")
plt.title("Atlas")
plt.axis("off")
plt.show()