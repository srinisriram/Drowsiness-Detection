import cv2
import dlib
import numpy as np
from keras.models import load_model
from scipy.spatial import distance as dist
from imutils import face_utils
from Crop_Eyes import cropEyes
from CNN_Preprocess import cnnPreprocess
from Variables import drowsy_counter, state, wake_up, max_closed_eyes

#Load CNN
model = load_model('models/blinkModel.h5')

def Extract_And_Predict_Eyes(frame):
	"""
	This function will take a frame, extract the left and right eye (if they are there), and predict whether the eyes are opened or closed using the CNN Classifier.

	Params:
		frame: The input image where we will extract and predict the eyes.

	Returns:
		state: A variable that will have either of these 3 values:
			1. Open (Eyes are open)
			2. Close (Eyes are closed)
			3. False (No Eyes were detected)
	"""

	#Crop the eyes from the frame using the cropEyes function
        eyes = cropEyes(frame)

	#If no Eyes are detected the function will return as false. Otherwise we will set the variables left_eye and right_eye as the two eyes.
        if eyes is None:
		return False 
        else:
                found_eye = True
                left_eye,right_eye = eyes

	#Using the CNN Classifier and the cnnPreprocess function we will predict the two eyes. After that we will average the two predictions (predictions are from 0-1).
        prediction = (model.predict(cnnPreprocess(left_eye)) + model.predict(cnnPreprocess(right_eye)))/2.0

        #If the prediction value is greater than 0.5, the eyes are open. Otherwise the eyes are closed.
        if prediction > 0.5 :
                state = 'open'

        else:
                state = 'close'
	#Return the state of the eyes to the main function.
        return state

def main():
	"""
	This is the the Drowsiness Detector Function. By reading live frames from the camera, we will predict if the person is drowsy or not.

	Params:
		None
	Returns:
		None

	"""
	#Set up the variables defined in Variables.py
	global drowsy_counter
	global wake_up
	global max_closed_eyes
	global state

	# Start the camera stream.
	camera = cv2.VideoCapture(0)

	#Create a while True function that will always read frames from the camera and perform the Drowsiness Detection.
	while True:
		ret, frame = camera.read()

		#Put the text on the video frame that tells the state of the eyes right now, and if there is a Drowsiness Alert.
		cv2.putText(frame, "State: {}".format(state), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		cv2.putText(frame, "Drowsiness Alert: {}".format(wake_up), (300, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

		#Show the camera stream with the text.
		cv2.imshow('blinks counter', frame)

		#Using previously defined function, we will predict if the Eyes are opened or closed by unpacking the previous variable "state".
		state = Extract_And_Predict_Eyes(frame)

		#If the eyes are open set the drowsy_counter to 0 and set the Drowsiness Alert to False.
		if state == 'open':
			drowsy_counter = 0
			wake_up = 'False'

		#If the eyes are closed add 1 to the drowsy_counter
		if state == 'close':
			drowsy_counter+=1
		#If the drowsy_counter exceeds the threshold for closed eyes frames (defined in Variables.py), set the Drowsiness Alert to "Closed Eyes".
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
