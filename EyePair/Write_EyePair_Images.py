import numpy as np
import cv2
from EyePair import *


def find_eye_pair_from_image(image):
	found_eye_pair = False
	picture = ReturnEyePairFunc(image)
	if(np.sum(picture == 0)): # not return (default black picture)
			return found_eye_pair
	i=0
	while i < 10:
		a = cv2.imwrite('TwestPics'+str(i)+'.jpg', picture)
		i+=1
	found_eye_pair = True
	return found_eye_pair

def main():
	i=0
	capture = cv2.VideoCapture(0)
	while True:
		ret, image = capture.read()
		cv2.imshow('frame', image)
		k = cv2.waitKey(30) & 0xff
		if k == 27: #ESC
			break
		if find_eye_pair_from_image(image):
			print("Successfully found eye pair from image.Exiting...")
			break
	capture.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
