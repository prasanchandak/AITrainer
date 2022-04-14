# Importing the libraries
from base64 import encode
from turtle import right
import cv2
import numpy as np
import mediapipe as mp
from firebase import firebase

##
import face_recognition
import os
from datetime import datetime
##
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path) # My list is a list of names of files in the directory path.
print(myList)
url = 'https://ai-trainer-d413b-default-rtdb.firebaseio.com/'
firebase = firebase.FirebaseApplication(url)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}') # This is the image read from the path
    images.append(curImg) # This function creates an array for the image array
    classNames.append(os.path.splitext(cl)[0]) # Adds the name of the image to the classNames array
print(classNames) # Prints the classNames array
# The above code reads and displays the names of the images in ImagesAttendance folder.

def findEncodings(images): # Images array is given in the function argument
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0] # Generating encodings
        # Given an image, return the 128-dimension face encoding for each face in the image.
        encodeList.append(encode) # Appending to the encodeList array
    return encodeList 

# This function finds the encodings in the images. These encodings can be then saved to a file using a numpy array

def markAttendance(name):
    with open('Attendance2.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        print(myDataList)
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# This function marks attendance in the Attendance2.csv file



encodeListKnown =  findEncodings(images) # Gives the encodings of the input array of images
# This encodeListKnown can be then loaded from a numpy file.
print(encodeListKnown)
print('Encoding Complete')


##


##
def calc_angle(a, b, c):  # 3D points
    ''' Arguments:
        a,b,c -- Values (x,y,z, visibility) of the three points a, b and c which will be used to calculate the
                vectors ab and bc where 'b' will be 'elbow', 'a' will be shoulder and 'c' will be wrist.

        Returns:
        theta : Angle in degress between the lines joined by coordinates (a,b) and (b,c)
    '''
    a = np.array([a.x, a.y])  # , a.z])    # Reduce 3D point to 2D
    b = np.array([b.x, b.y])  # , b.z])    # Reduce 3D point to 2D
    c = np.array([c.x, c.y])  # , c.z])    # Reduce 3D point to 2D

    ab = np.subtract(a, b)
    bc = np.subtract(b, c)

    theta = np.arccos(np.dot(ab, bc) / np.multiply(np.linalg.norm(ab), np.linalg.norm(
        bc)))  # A.B = |A||B|cos(x) where x is the angle b/w A and B
    theta = 180 - 180 * theta / 3.14  # Convert radians to degrees
    return np.round(theta, 2)

tracker_name = 'KMnO4'
name='KMnO4'
x1 =0
y1 = 0
x2 = 0
y2 = 0

def infer():
    global tracker_name
    global name
    global x1, x2, y1, y2
    mp_drawing = mp.solutions.drawing_utils  # Connecting Keypoints Visuals
    mp_pose = mp.solutions.pose  # Keypoint detection model
    left_flag = None  # Flag which stores hand position(Either UP or DOWN)
    left_count = 0  # Storage for count of bicep curls
    right_flag = None
    right_count = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)  # Lnadmark detection model instance
    while cap.isOpened():
        _, frame = cap.read()

        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25) 
        #cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        # face_locations() which gives an array listing the co-ordinates of each face.
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        # Generates an encoding on the current image given the coordinates


        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): # Loops the current face one by one and their location
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            # Compares the current encodings to the listed encodings
            '''

            https://face-recognition.readthedocs.io/en/latest/face_recognition.html

            face_recognition.api.compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6)[source]
            Compare a list of face encodings against a candidate encoding to see if they match.

            Parameters:	
            known_face_encodings – A list of known face encodings
            face_encoding_to_check – A single face encoding to compare against the list
            tolerance – How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
            Returns:	
            A list of True/False values indicating which known_face_encodings match the face encoding to check
            '''

            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # Gives the face distance of the encoded face.
            # Given a list of face encodings, compare them to a known face encoding and get a euclidean distance for each comparison face. 
            # The distance tells you how similar the faces are.
            # Returns:	A numpy ndarray with the distance for each face in the same order as the ‘faces’ array.
            # The lesser the face distance, the better the match

            print(faceDis)
            matchIndex = np.argmin(faceDis)
            # Returns the indices of the minimum values along an axis.
            # Array of indices into the array. It has the same shape as a.shape with the dimension along axis removed. 
            # If keepdims is set to True, then the size of axis will be 1 with the resulting array having same shape as a.shape.

            print("matches:\n", matches, "\n matchIndex: \n", matchIndex)

            if matches[matchIndex]:
                # matches is an array which has a list of true/false valuesindicating which known_face_encodings match the given face.
                name = classNames[matchIndex].upper() # If the 
                tracker_name = name
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                #cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

      #  cv2.imshow('Webcam', img)

        
        # BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR frame to RGB
        image.flags.writeable = False

        # Make Detections
        results = pose.process(image)  # Get landmarks of the object in frame from the model

        # Back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR

        try:
            # Extract Landmarks
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate angle
            left_angle = calc_angle(left_shoulder, left_elbow, left_wrist)  # Get angle
            right_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

            # Visualize angle
            cv2.putText(image, \
                        str(left_angle), \
                        tuple(np.multiply([left_elbow.x, left_elbow.y], [640, 480]).astype(int)), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, \
                        str(right_angle), \
                        tuple(np.multiply([right_elbow.x, right_elbow.y], [640, 480]).astype(int)), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Counter
            if left_angle > 160:
                left_flag = 'down'
            if left_angle < 50 and left_flag == 'down':
                left_count += 1
                left_flag = 'up'

            if right_angle > 160:
                right_flag = 'down'
            if right_angle < 50 and right_flag == 'down':
                right_count += 1
                right_flag = 'up'

        except:
            pass

        # Setup Status Box
        cv2.rectangle(image, (0, 0), (1024, 73), (10, 10, 10), -1)
        cv2.putText(image, 'Left =' + str(left_count) + '    Right=' + str(right_count),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        print(tracker_name)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, tracker_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        #Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe feed', image)
        
        k = cv2.waitKey(30) & 0xff  # Esc for quiting the app

        if k == 27:
            result = firebase.put("/Name", tracker_name, "biceps")
            result1 = firebase.put("/LeftRepsCount", str(left_count), "left")
            result2 = firebase.put("/RightRepsCount", str(right_count), "right")
            break
        elif k == ord('r'):  # Reset the counter on pressing 'r' on the Keyboard
            left_count = 0
            right_count = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    infer()