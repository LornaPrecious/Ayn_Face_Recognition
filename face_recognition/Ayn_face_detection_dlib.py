import cv2
import time
import dlib
import imutils
from imutils import face_utils
#from helpers.helpers import convert_and_trim_bb
import face_recognition_models as frm
import os
import numpy as np

#*******************************************************************************************************
#BLINKING IMPORTS
from scipy.spatial import distance #Used to calculate the eclidean distance

#Ear - eye aspect ratio
def calc_Ear(eye):
    A = distance.euclidean(eye[1], eye[5]) #point 2, 6 (on image in the document)
    B = distance.euclidean(eye[2], eye[4]) #point 3, 5
    C = distance.euclidean(eye[0], eye[3]) #point 1, 4 - the horizontal distance - crosses the eye horizontally
    eye_aspect_ratio = (A + B)/(2.0 * C)
    return eye_aspect_ratio
#**************************************************************************************************************************


def detect():
    #MTCNN and dlib for face detection, python deepface module for face recognition, Open CV for video and image processing and display/ matplotlib:

    models_directory = os.path.join(os.path.dirname(__file__), "Lib", "site-packages", "face-recognition-models", "models")
    model_filename = frm.pose_predictor_model_location()
    model_path = os.path.join(models_directory, model_filename)

    #Output/ result folders for extracted face and eye images
    output_faces_directory = "./images/faces"
    output_eyes_directory = "./images/eyes"

    os.makedirs(output_faces_directory, exist_ok=True)
    os.makedirs(output_eyes_directory, exist_ok=True)

    #Access our device’s camera to read a live stream of video data - passing parameter 0 tells OpenCV to use default camera on the device
    video_capture = cv2.VideoCapture(0)
    time.sleep(1)

    frame_count = 0

    #DETECTOR INITIALIZATION
    detector_dlib = dlib.get_frontal_face_detector() #returns the pre-trained HOG + Linear SVM face detector included in the dlib library.
    predictor = dlib.shape_predictor(model_path)


    while True:   #EVERYTHING BELOW THIS OCCURS ON A SINGLE FRAME AT A TIME 
        # Grab a single frame of video
        #ret, check, result?
        check, frame_image = video_capture.read()

        # Bail out when the video file ends
        if not check:
            video_capture.release()
            break

        frame_count += 1

        if frame_count % 15 == 0:   
        
            frame_gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
            frame = cv2.equalizeHist(frame_gray)    
                    

                #DLIB FACE DETECTION  
        # DOESN'T DETECT FACES ON SOME ANGLED POSES
            faces_dlib = detector_dlib(frame)

            for face_dlib in faces_dlib: #For every face detected::
                    
                x, y, w, h = face_dlib.left(), face_dlib.top(), face_dlib.width(), face_dlib.height()

                    #OpenCV bounding box - rectangle around detected face - color red

                cv2.rectangle(frame_image, (x, y), (x + w, y + h), (0, 0, 255), 2)


                
                    # Extract the detected face from the original frame
                detected_face = frame_image[y:y+h, x:x+w]
                #detected_face = cv2.resize(detected_face, (300, 300)) #without a detected face, bring an error, needs a try, except

                """
                    Detected_face = frame_image[y:y+h, x:x+w] is using NumPy array slicing to extract a region of interest (ROI) from the original frame_image. Let me break down the components:

                    y:y+h: This represents the vertical range of the array (pixel rows) to be included in the extracted region. It starts from y (the top coordinate of the detected face) and goes up to y+h (the bottom coordinate of the detected face).

                    x:x+w: This represents the horizontal range of the array (pixel columns) to be included in the extracted region. It starts from x (the left coordinate of the detected face) and goes up to x+w (the right coordinate of the detected face).  
                    """
                #SAVING DETECTED FACES
                    
                face_filename = f"face_{frame_count}.png"
                face_filepath = os.path.join(output_faces_directory, face_filename)
                cv2.imwrite(face_filepath, detected_face)


                
                    #DLIB FACE LANDMARKS
                # Make the prediction and transfom it to numpy array
                shape_landmark = predictor(frame, face_dlib)
                shape = face_utils.shape_to_np(shape_landmark)   

                    # Draw on our image, all the found cordinate points (x,y) 
                for (x, y) in shape:
                    cv2.circle(frame_image, (x, y), 2, (0, 255, 0), -1)  


                    #EXTRACTING FACE FEATURES USING INDEXES
                        
                    """
                        We can fetch these indices manually :

                        shape[48:67] #(index starts from 0)
                        
                        or

                        The face_utils.FACIAL_LANDMARKS_IDXS.items() is a dictionary constant with the value shown below:

                        FACIAL_LANDMARKS_68_IDXS = OrderedDict([(“mouth”, (48, 68)), (“inner_mouth”, (60, 68)), (“right_eyebrow”, (17, 22)),(“left_eyebrow”, (22, 27)), (“right_eye”, (36, 42)), 
                        
                        (“left_eye”, (42, 48)),(“nose”, (27, 36)), (“jaw”, (0, 17))])
                
                """
            
                #DISPLAYING EYES BY GETTING THE LANDMARKS AND 'EXTRACTING THE BOUNDING RECT AROUND THE ROI
                for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    if(name == "left_eye" or name=="right_eye"):
                        #DISPLAYING THE EYES
                            # extract the ROI(region of interest) of the face region as a separate image
                        (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[i:j]]))
                        roi = frame_image[y1:y1 + h1, x1:x1 + w1]
                        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
                            # save the eye images
                        eye_filename = f"eye_ {name}{frame_count}.png"
                        eye_filepath = os.path.join(output_eyes_directory, eye_filename)
                        cv2.imshow("Eyes", roi)
                        cv2.imwrite(eye_filepath, roi)


    #************************************************************************************************
            #blinking
                        
                leftEye = []
                rightEye = []

            #getting points/cordinates from 68 shape detector for the eyes
                #left eye
                for n in range(36,42): #get the x and y cordinate for each eye fandmark cordinate/point
                    x = shape_landmark.part(n).x
                    y = shape_landmark.part(n).y
                    leftEye.append((x,y)) #Adding the x, y cordinates in the left_eye list/array
                    next_point = n+1 #(eg. cordinate 37 from 36)
                    if n == 41: #If u are at the last cordinate - for left eye
                        next_point = 36
                    x2 = shape_landmark.part(next_point).x #Getting cordinates for next point 
                    y2 = shape_landmark.part(next_point).y
                    cv2.line(frame_image,(x,y),(x2,y2),(0,255,0),1) # draw a line between cordinate n and cordinate next_point (green line, thickness i)

                #right eye
                for n in range(42,48):
                    x = shape_landmark.part(n).x
                    y = shape_landmark.part(n).y
                    rightEye.append((x,y))
                    next_point = n+1
                    if n == 47:
                        next_point = 42
                    x2 = shape_landmark.part(next_point).x
                    y2 = shape_landmark.part(next_point).y
                    cv2.line(frame_image,(x,y),(x2,y2),(0,255,0),1)


                    #ear = eye aspect ratio

                #Passing the left and right eye array/list to the calc_Ear function
                left_ear = calc_Ear(leftEye) 
                right_ear = calc_Ear(rightEye)


                EAR = (left_ear+right_ear)/2  #Average of both EARs
                EAR = round(EAR,2)    #Rounding off to 2 decimal places
                if EAR <0.26: #EAR below 0.26 means closed eyes
                    cv2.putText(frame,"DROWSY",(20,100),
                        cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
                    cv2.putText(frame,"Are you Sleepy?",(20,400),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
                    print("Drowsy")
                print(EAR)
    #**********************************************************************************************************************************************  
            #show the plot 
            cv2.imshow('Face Detection', frame_image)
            cv2.waitKey(1)



        """
        1.waitKey(0) will display the window infinitely until any keypress (it is suitable for image display).

        2.waitKey(1) will display a frame for 1 ms, after which display will be automatically closed. Since the OS has a minimum time between switching threads, the function will not wait exactly 1 ms, it will wait at least 1 ms, depending on what else is running on your computer at that time.

        So, if you use waitKey(0) you see a still image until you actually press something while for waitKey(1) the function will show a frame for at least 1 ms only.
         
        """

         # *********************************************************************
        #   THINGS TO IMPLEMENT

"""
Get users unique id to attach to corresponding face and eyes
Should extract eyes and face on the first two/ 5 frames to avoid too much 'data collection'/ extract in motion not at rest
Resize images if needed -> image = imutils.resize(image, width=500)
Ensure varrying light and head poses are accommodated
Make the code more efficient through classes and functions 

Implement partially alert/ asleep in blinking - code perhaps by researching more on euclidian distance and EAR
Find a universal EAR that defines drowsy etc.

Create a 'dataset/ faces' directory/database where all faces will be stored with their respective user ids for training, face identification/recognition and verification
"""
      
