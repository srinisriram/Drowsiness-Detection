import numpy as np
import tensorflow as tf
import cv2
from EyePair import *
from PIL import Image
from keras import models

def find_eye_pair_from_image(image):
    found_eye_pair = False
    picture = ReturnEyePairFunc(image)
    if(np.sum(picture == 0)): # not return (default black picture)
            return found_eye_pair, picture
    found_eye_pair = True
    return found_eye_pair, picture

model = models.load_model('test.h5')

def predict_eye_pair_from_image(picture):
    im = Image.fromarray(picture, 'RGB')
    im = im.resize((64,64))
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    if result == 0:
        prediction = "Closed"
    elif result == 1:
        prediction = "Open"
    print ("Your eyes are", prediction)


def main():
    i=0
    capture = cv2.VideoCapture(0)
    while True:
        ret, image = capture.read()
        cv2.imshow('frame', image)
        k = cv2.waitKey(30) & 0xff
        if k == 27: #ESC
            break
        found_eye_pair, picture = find_eye_pair_from_image(image)
        if found_eye_pair:
            print("Detected an Eye Pair")
            predict_eye_pair_from_image(picture)
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
