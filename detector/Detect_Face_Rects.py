import cv2

#Load OpenCV's face cascade.
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')


def detect(img, cascade = face_cascade , minimumFeatureSize=(20, 20)):
    """
    This function extracts the face coordinates from an image using OpenCV's Haarcascade.

    Params:
	img: The image where we will extract the face coordinates.
	cascade: The haarcascade we will be using to detect the face(haarcascade_frontalface_alt.xml).
	minimumFeatureSize: A variable used when we detect for faces(minimum size of the face).

    Returns:
	rects: A set of rectangle coordinates defining where the face is.
	None: If there are no faces detected will return an empty array.
    """
    #Check if we can load the haarcascade.
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))
    #Define the variable rects to be the coordinates for the face that is detected.
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)

    # If there are no faces detected return empty coordinates.
    if len(rects) == 0:
        return []

    #Convert last coord from (width,height) to (maxX, maxY)
    rects[:, 2:] += rects[:, :2]

    #Return face coordinates to main function.
    return rects

