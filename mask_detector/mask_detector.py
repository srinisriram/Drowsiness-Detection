# import the necessary packages
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from detect import detect


#Load all the models, and start the camera stream
faceModel = cv2.dnn.readNet("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")
maskModel = load_model("models/mask_detector.h5")
stream = cv2.VideoCapture(0)

while True:
	#Read frame from the stream
	ret, frame = stream.read()

	#Run the detect function on the frame
	(locations, predictions) = detect(frame, faceModel, maskModel)

	#Go through each face detection.
	for (box, pred) in zip(locations, predictions):
		#Extract the prediction and bounding box coords
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		#Determine the class label and make actions accordingly
		if mask > withoutMask:
			label = 'Mask'
			color = (0,255,0)
			#Add email sending, motor control, etc.
		else:
			label = 'No Mask'
			color = (0,0,255)
			#Add email sending, motor control, etc.

		# add probability in label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		#Place label and Bounding Box
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# break from loop if key pressed is q
	if key == ord("q"):
		break

#Cleanup
stream.release()
cv2.destroyAllWindows()
