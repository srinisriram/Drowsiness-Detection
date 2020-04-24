import cv2
import dlib
import numpy as np
from keras.models import load_model
from scipy.spatial import distance as dist
from imutils import face_utils
from Crop_Eyes import cropEyes
from CNN_Preprocess import cnnPreprocess
from Variables import drowsy_counter, state, wake_up


def main():

	global drowsy_counter
	global state
	global wake_up

	# open the camera,load the cnn model 
	camera = cv2.VideoCapture(0)
	model = load_model('blinkModel.h5')

	while True:
		ret, frame = camera.read()

		cv2.putText(frame, "State: {}".format(state), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "Drowsiness Alert: {}".format(wake_up), (300, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.imshow('blinks counter', frame)

		eyes = cropEyes(frame)

		if eyes is None:
			continue
		else:
			left_eye,right_eye = eyes

		# average the predictions of the two eyes 
		prediction = (model.predict(cnnPreprocess(left_eye)) + model.predict(cnnPreprocess(right_eye)))/2.0


		# if the eyes are open reset the counter for close eyes
		if prediction > 0.5 :
			state = 'open'
			drowsy_counter = 0
			wake_up = 'False'

		else:
			state = 'close'
			drowsy_counter +=1
		if drowsy_counter > 50:
			wake_up = "Closed Eyes"
		# show the frame
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord('q'):
			break
	# do a little clean up
	cv2.destroyAllWindows()
	del(camera)


if __name__ == '__main__':
	main()
