
import numpy as np
import skimage
from skimage import io
import matplotlib.pyplot as plt


RED_PX=[0,1]
GREEN_PX=[0,1]
# BLUE_PX=[0,61,105,106,105,103,104,108,112,123,133,143,155,167,179,192,204,216,229,242,255]
BLUE_PX=[0.,0.239,0.4117,0.4156,0.4117,0.4039,0.4078,0.4235,0.4392,0.4823,0.5215,0.5608,0.6078,0.6549,0.7019,0.7529,0.8,0.8471,0.8980,0.9490,1.]

def cuvres_adjust(channel,values):
    #faltten
    original_size=channel.shape
    flat_channel=channel.flatten()
    channel_adjusted=np.interp(
        flat_channel,
        np.linspace(0,1,len(values)),
        values)
    # put back to image shape
    return channel_adjusted.reshape(original_size)


def get_adjusted_image(img):
    r, g, b = np.split(img,indices_or_sections=3,axis=2)
    red_adjusted=cuvres_adjust(r,RED_PX)
    green_adjusted=cuvres_adjust(g,GREEN_PX)
    blue_adjusted=cuvres_adjust(b,BLUE_PX)
    adjusted_img = np.concatenate((red_adjusted, green_adjusted, blue_adjusted),axis=2)
    return adjusted_img
