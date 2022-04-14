# import AITrainer
import subprocess

exec(open("AITrainer.py").read())
# This will execute another file here.

#subprocess.call("AITrainer.py", shell=True)

# # Importing the libraries
# import cv2
# import numpy as np
# import mediapipe as mp
# from firebase import firebase

# ##
# import face_recognition
# import os
# from datetime import datetime
# ##
# path = 'ImagesAttendance'
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)
# url = 'https://posturetracking-default-rtdb.firebaseio.com/'
# #firebase = firebase.FirebaseApplication(url)
# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)

# encodeListKnown =  AITrainer.findEncodings(images)


if __name__=='main':
    # AITrainer.tracker_name = 'KMnO4'
    # AITrainer.name='KMnO4'
    # AITrainer.x1 =0
    # AITrainer.y1 = 0
    # AITrainer.x2 = 0
    # AITrainer.y2 = 0
    # AITrainer.infer()

    exec(open("AITrainer.py").read())