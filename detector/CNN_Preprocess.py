import numpy as np


# make the image to have the same format as at training 
def cnnPreprocess(img):
        img = img.astype('float32')
        img /= 255
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        return img


