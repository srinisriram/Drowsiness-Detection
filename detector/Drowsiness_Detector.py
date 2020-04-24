import cv2
import dlib
import numpy as np
from keras.models import load_model
from scipy.spatial import distance as dist
from imutils import face_utils
from Crop_Eyes import cropEyes
from CNN_Preprocess import cnnPreprocess
from Variables import drowsy_counter, state, wake_up, max_closed_eyes

#Load the CNN
model = load_model('models/blinkModel.h5')

def Extract_and_Predict_Eyes(frame):
	"""
	This function will:
		1. Extract the Eyes (If there are any)
		2. Predict whether the Eyes are Opened or Closed by using the CNN classifier.

	Params:
		frame: The input image
	Returns:
		A variable "state" which can have these three values:
			1. open(Eyes are opened)
			2. close(Eyes are closed)
			3. False(No Eyes were detected)
	"""
	#Crop the eyes from the frame
	eyes = cropEyes(frame)

	#If there are no eyes the function will return as False. Otherwise we define the variables left_eye and right_eye to be the eyes.
	if eyes is None:
		return False
	else:
		left_eye,right_eye = eyes

	#Run the cnnPreprocess function on the two eyes to make sure the eyes can be classified by the CNN.
	cnnLeft = cnnPreprocess(left_eye)
	cnnRight = cnnPreprocess(right_eye) 

	#Average the predictions that the CNN gives the two eyes (Each prediction is between 0 and 1).
	prediction = (model.predict(cnnLeft) + model.predict(cnnRight))/2.0

	#If the prediction is greater than 0.5, then the eyes are open, so we define the variable 'state'= 'open'. Otherwise the eyes are closed, so 'state' = 'close'.
	if prediction > 0.5 :
		state = 'open'
	else:
		state = 'close'

	#Return the value of state to the main function.
	return state


def main():
	"""
	This is the Drowsiness Detector. By using the previously defined functions, we will use the data gathered by these functions to determine if the person is drowsy or not.

	Params:
		None
	Returns:
		None
	"""

	#Get the variables defined in Variables.py
	global drowsy_counter
	global state
	global wake_up

	#Open the camera stream. 
	camera = cv2.VideoCapture(0)

	#Set a while True loop that will continue to read frames from the camera and detect if a person is drowsy or not.
	while True:
		ret, frame = camera.read()

		#Put text on the camera frame, one being the state of the eyes right now, and one being the Drowsiness Alert.
		cv2.putText(frame, "State: {}".format(state), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		cv2.putText(frame, "Drowsiness Alert: {}".format(wake_up), (300, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

		#Open the Camera Stream.
		cv2.imshow('blinks counter', frame)

		#Unpack the variable 'state' by using the Extract_and_Predict function on the Camera frame.
		state = Extract_and_Predict_Eyes(frame)

		#If 'state' = 'open', then we reset the drowsy_counter to 0, and set the Drowsiness Alert to False. If 'state'='close', then we add 1 to the drowsy_counter.
		if state == 'open':
			drowsy_counter = 0
			wake_up = 'False'
		if state == 'close':
			drowsy_counter +=1

		#If the drowsy_counter exceeds the threshold for consecutive closed eye frames(defined in Variables.py), then we set the Drowsiness Alert to be 'Closed Eyes'
		if drowsy_counter > max_closed_eyes:
			wake_up = "Closed Eyes"


		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord('q'):
			break
	# do a little clean up
	cv2.destroyAllWindows()
	del(camera)


if __name__ == '__main__':
	main()
