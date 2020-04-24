import cv2
import dlib
from Detect_Face_Rects import detect
from imutils import face_utils
import numpy as np

#Open dlib's facial_landmarks predictor.
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")



def cropEyes(frame):
    """
    This function will take an input frame and extract the left and right eye from the image (if there are any).

    Params:
        frame: The input frame which we will extract the eyes.

    Returns:
        left_eye_image: The image of the left eye.
        right_eye_image: The image of the right eye.
        None: If there are no eyes the function will return None.
    """
    #Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect the face from the grayscale image
    te = detect(gray, minimumFeatureSize=(80, 80))

    #If the face detector returns nothing then return None to the main function.
    #If there are more than one faces grab the bigger one (index 0).
    #If there is one face then set the face variable to be the face.
    if len(te) == 0:
       return None
    elif len(te) > 1:
       face = te[0]
    elif len(te) == 1:
        [face] = te

    #Make a rectangle using the face coordinates.
    face_rect = dlib.rectangle(left = int(face[0]), top = int(face[1]),
                                                            right = int(face[2]), bottom = int(face[3]))

    #Determine the facial landmarks for the face region using dlib shape predictor.
    shape = predictor(gray, face_rect)
    shape = face_utils.shape_to_np(shape)

    # Grab the indexes of the right eye and the left eye.
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # extract the left and right eye coordinates
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    # keep the upper and the lower limit of the eye
    # and compute the height
    l_uppery = min(leftEye[1:3,1])
    l_lowy = max(leftEye[4:,1])
    l_dify = abs(l_uppery - l_lowy)

    # compute the width of the eye

    lw = (leftEye[3][0] - leftEye[0][0])

    # we want the image for the cnn to be (26,34)
    # so we add the half of the difference at x and y
    # axis from the width at height respectively left-right
    # and up-down
    minxl = (leftEye[0][0] - ((34-lw)/2))
    maxxl = (leftEye[3][0] + ((34-lw)/2))
    minyl = (l_uppery - ((26-l_dify)/2))
    maxyl = (l_lowy + ((26-l_dify)/2))

    # crop the eye rectangle from the frame
    left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
    left_eye_rect = left_eye_rect.astype(int)
    left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]

    # same as left eye at right eye
    r_uppery = min(rightEye[1:3,1])
    r_lowy = max(rightEye[4:,1])
    r_dify = abs(r_uppery - r_lowy)
    rw = (rightEye[3][0] - rightEye[0][0])
    minxr = (rightEye[0][0]-((34-rw)/2))
    maxxr = (rightEye[3][0] + ((34-rw)/2))
    minyr = (r_uppery - ((26-r_dify)/2))
    maxyr = (r_lowy + ((26-r_dify)/2))
    right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
    right_eye_rect = right_eye_rect.astype(int)
    right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

    # if it doesn't detect left or right eye return None
    if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
        return None
    # resize for the CNN
    left_eye_image = cv2.resize(left_eye_image, (34, 26))
    right_eye_image = cv2.resize(right_eye_image, (34, 26))
    right_eye_image = cv2.flip(right_eye_image, 1)
    # return left and right eye
    return left_eye_image, right_eye_image
