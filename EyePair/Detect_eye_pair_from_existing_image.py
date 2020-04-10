import glob
import shutil

import numpy as np
import cv2
from EyePair import *
import shutil



while True:
	a = cv2.imread('Fwame4.jpg')

	picture = ReturnEyePairFunc(a)
	cv2.imshow("picture",picture)
	cv2.imwrite('Eyesleep.jpg', picture)
	if(np.sum(picture == 0)): # not return (default black picture)
                print("Not Eye-Pair")




	k = cv2.waitKey(30) & 0xff

	if k == 27: #ESC
		break


	cv2.destroyAllWindows()
