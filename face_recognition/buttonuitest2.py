# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ayn.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
import os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

from pyfirmata2 import Arduino, SERVO
import pyfirmata2
import time
import pyrebase
import requests
import requests_oauthlib
import numpy as np
from azureml.core import Workspace, Model ,Webservice
from azureml.core import Webservice as azwebservice
import cv2
import dlib
import imutils
from imutils import face_utils
#from helpers.helpers import convert_and_trim_bb
import face_recognition_models as frm
import numpy as np
import scipy

from scipy.spatial import distance #Used to calculate the eclidean distance

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(756, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 751, 591))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(23)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.BtnRegister = QtWidgets.QPushButton(self.tab_2)
        self.BtnRegister.setGeometry(QtCore.QRect(160, 350, 371, 81))
        self.BtnRegister.setObjectName("BtnRegister")
        self.textEmail = QtWidgets.QTextEdit(self.tab_2)
        self.textEmail.setGeometry(QtCore.QRect(340, 40, 311, 31))
        self.textEmail.setObjectName("textEmail")
        self.label_Email = QtWidgets.QLabel(self.tab_2)
        self.label_Email.setGeometry(QtCore.QRect(30, 40, 121, 31))
        self.label_Email.setObjectName("label_Email")
        self.label_Password = QtWidgets.QLabel(self.tab_2)
        self.label_Password.setGeometry(QtCore.QRect(30, 90, 121, 31))
        self.label_Password.setObjectName("label_Password")
        self.label_PhoneNum = QtWidgets.QLabel(self.tab_2)
        self.label_PhoneNum.setGeometry(QtCore.QRect(30, 190, 191, 31))
        self.label_PhoneNum.setObjectName("label_PhoneNum")
        self.label_Econtact = QtWidgets.QLabel(self.tab_2)
        self.label_Econtact.setGeometry(QtCore.QRect(30, 240, 261, 41))
        self.label_Econtact.setObjectName("label_Econtact")
        self.textPassword = QtWidgets.QTextEdit(self.tab_2)
        self.textPassword.setGeometry(QtCore.QRect(340, 90, 311, 31))
        self.textPassword.setObjectName("textPassword")
        self.textPhone = QtWidgets.QTextEdit(self.tab_2)
        self.textPhone.setGeometry(QtCore.QRect(340, 190, 311, 31))
        self.textPhone.setObjectName("textPhone")
        self.textEcontact = QtWidgets.QTextEdit(self.tab_2)
        self.textEcontact.setGeometry(QtCore.QRect(340, 250, 311, 31))
        self.textEcontact.setObjectName("textEcontact")
        self.label_confirm_Password = QtWidgets.QLabel(self.tab_2)
        self.label_confirm_Password.setGeometry(QtCore.QRect(30, 140, 231, 31))
        self.label_confirm_Password.setObjectName("label_confirm_Password")
        self.textconfirmPassword = QtWidgets.QTextEdit(self.tab_2)
        self.textconfirmPassword.setGeometry(QtCore.QRect(340, 140, 311, 31))
        self.textconfirmPassword.setObjectName("textconfirmPassword")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(200, 300, 291, 91))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(23)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.photo = QtWidgets.QLabel(self.tab)
        self.photo.setGeometry(QtCore.QRect(270, 100, 161, 131))
        self.photo.setText("")
        self.photo.setPixmap(QtGui.QPixmap("../../Pictures/2c421z58lmg81.png"))
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")
        self.engine_slowdown = QtWidgets.QPushButton(self.tab)
        self.engine_slowdown.setGeometry(QtCore.QRect(10, 440, 331, 61))
        self.engine_slowdown.setObjectName("engine_slowdown")
        self.engine_reset = QtWidgets.QPushButton(self.tab)
        self.engine_reset.setGeometry(QtCore.QRect(390, 440, 311, 61))
        self.engine_reset.setObjectName("engine_reset")
        self.tabWidget.addTab(self.tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 756, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

                
        self.engine_slowdown.clicked.connect(self.slowdown)
        self.engine_reset.clicked.connect(self.resetcar)
        self.pushButton.clicked.connect(self.startprogram)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BtnRegister.setText(_translate("MainWindow", "Register"))
        self.label_Email.setText(_translate("MainWindow", "Email"))
        self.label_Password.setText(_translate("MainWindow", "Password"))
        self.label_PhoneNum.setText(_translate("MainWindow", "Phone Number"))
        self.label_Econtact.setText(_translate("MainWindow", "Emergency Contact "))
        self.label_confirm_Password.setText(_translate("MainWindow", "Confirm Password"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Register"))
        self.pushButton.setText(_translate("MainWindow", "START"))
        self.engine_slowdown.setText(_translate("MainWindow", "Trigger Engine Slowdown"))
        self.engine_reset.setText(_translate("MainWindow", "Trigger Engine Reset"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "StartApplication"))

    def slowdown(self):
        self.photo.setPixmap(QtGui.QPixmap("hazard.jpg"))
        self.runarduino()
        
        
    def resetcar(self):
        msg = QMessageBox()
        msg.setWindowTitle("Confio")
        self.photo.setPixmap(QtGui.QPixmap(""))
        self.runreset()
      
    def startprogram(self):
        print("")

    def runreset(self):
        PORT =  pyfirmata2.Arduino.AUTODETECT
        board = pyfirmata2.Arduino("COM5")
        # Set the pin number to which the servo is connected
        servo_pin = 7

        # Initialize the servo
        servo = board.get_pin(f'd:{servo_pin}:s')
        # Function to rotate the servo to a specific angle
        def rotate_servo(angle):
            # Map the angle to the servo's range (0 to 180)
            mapped_angle = int(angle)
            mapped_angle = max(0, min(mapped_angle, 180))

            # Rotate the servo to the specified angle
            servo.write(mapped_angle)

        # Specify the time interval and target angles
        time_interval = 2  # in seconds
        #target_angles = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,
        #                  100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180]  # example angles to rotate to
        target_angles = [0]  # example angles to rotate to
        try:
            while True:
                # Rotate the servo to each target angle in the list
                for angle in target_angles:
                    rotate_servo(angle)
                    time.sleep(time_interval)

        except KeyboardInterrupt:
            # Handle keyboard interrupt (Ctrl+C)
            print("\nProgram interrupted. Resetting the servo to the initial position.")
            rotate_servo(0)

        finally:
            # Close the connection to the Arduino
            board.exit()

    def runarduino(self):
        # Set the port where your Arduino is connected
    # Make sure to change this to the correct port for your setup (e.g., 'COM3' on Windows or '/dev/ttyACM0' on Linux)

    # PORT =  pyfirmata2.Arduino.AUTODETECT
        board = pyfirmata2.Arduino("COM5")
        # Set the pin number to which the servo is connected
        servo_pin = 7

        # Initialize the servo
        servo = board.get_pin(f'd:{servo_pin}:s')
        # Function to rotate the servo to a specific angle
        def rotate_servo(angle):
            # Map the angle to the servo's range (0 to 180)
            mapped_angle = int(angle)
            mapped_angle = max(0, min(mapped_angle, 180))

            # Rotate the servo to the specified angle
            servo.write(mapped_angle)

        # Specify the time interval and target angles
        time_interval = 2  # in seconds
        #target_angles = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,
        #                  100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180]  # example angles to rotate to
        target_angles = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]  # example angles to rotate to


        try:
            while True:
                # Rotate the servo to each target angle in the list
                for angle in target_angles:
                    rotate_servo(angle)
                    time.sleep(time_interval)

        except KeyboardInterrupt:
            # Handle keyboard interrupt (Ctrl+C)
            print("\nProgram interrupted. Resetting the servo to the initial position.")
            rotate_servo(0)

        finally:
            # Close the connection to the Arduino
            board.exit()

    def createaccount(self,email,password):
        firebaseConfig= {
        'apiKey': "AIzaSyAJEjo0Bc9jIxakXOFuM7-DenLJd5UFaSE",
        'authDomain': "ayn-test-f481a.firebaseapp.com",
        'projectId': "ayn-test-f481a",
        'storageBucket': "ayn-test-f481a.appspot.com",
        'messagingSenderId': "1004965116162",
        'appId': "1:1004965116162:web:9a2319783f4dc35a712a1b",
        'measurementId': "G-SV87B6G0FN",
        'databaseURL': ""
        }
        firebase = pyrebase.initialize_app(firebaseConfig)

        email = self.textEmail.text()
        if self.textPassword.text() == self.textconfirmPassword.text():
            password = self.textPassword.text()
            auth = firebase.auth()
            auth.create_user_with_email_and_password(email,password)

    def mltest(self):
        # Replace with your Azure ML credentials and model details
        subscription_id = "YOUR_SUBSCRIPTION_ID"
        resource_group = "YOUR_RESOURCE_GROUP"
        #workspace_name = "YOUR_WORKSPACE_NAME"
        model_name = "your_car_detection_model"
        service_name = "your_car_detection_service"

        def process_video(video_path):
            """Predicts whether cars are present in a video using an Azure ML model.

            Args:
                video_path (str): Path to the video file.
            """

            try:
                # Connect to Azure ML workspace
                #ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
                ws = Workspace(subscription_id=subscription_id, resource_group=resource_group)
                model = Model(ws, model_name)
                service = azwebservice(ws, service_name)
                # Open video capture
                cap = cv2.VideoCapture(video_path)

                # Process frames continuously
                while True:
                    # Capture frame
                    ret, frame = cap.read()

                    if not ret:
                        break  # End of video

                    # Preprocess frame (replace with your model's requirements)
                    preprocessed_frame = ...

                    # Send prediction request to Azure ML
                    predictions = service.run(data=[preprocessed_frame])
                    # Extract predicted probability or confidence score for "car" class
                    car_probability = predictions[0]["Probability"]["car"]

                    # Handle prediction result (replace with your desired actions)
                    if car_probability >= 0.5:
                        print("Car detected with confidence:", car_probability)
                        
                        # You can trigger actions like displaying detection, recording video segment, etc.
                    else:
                        print("No car detected (confidence:", car_probability, ")")
                        

                    # Continue processing frames
                    cv2.imshow("Video", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Release resources
                cap.release()
                cv2.destroyAllWindows()

            except Exception as e:
                print(f"Error during prediction: {e}")

        # Example usage
        video_path = "path/to/your/video.mp4"
        process_video(video_path)


            




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
