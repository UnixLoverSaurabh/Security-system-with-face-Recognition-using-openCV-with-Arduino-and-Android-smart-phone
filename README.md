# Security-system-with-face-Recognition-using-openCV-with-Arduino-and-Android-smart-phone

#### Hardware Requirements:
    * Arduino UNO (You can use other boards )
    * Android smart phone (image capture)
    * Breadboard (For prototyping)
    * Ultrasonic sensor (Distance detection)

#### Software Requirements:
    * Python 3.5 (Should be installed, Linux OS usually have it pre-installed)
    * OpenCV
    * pyserial (Can be installed with pip)
    * numpy
    * Haarcascade
    * Xampp 
    * [Arduino IDE](https://www.arduino.cc/en/main/software)
    
   
Communicate between Arduino and Python is acheived using **pyserial** module.
High quality image is automatically clicked with android smart phone and ear phone wire.
The clicked image is further sent to system for processing via data cable.
Opencv's Haarcascade classifier is used to identifying or verifying a person from a digital image.

#### Working
  Image is captured as soon ultrasonic sensor detect 
  Work by comparing selected facial features from given image with faces within a database.
  It can uniquely identify a person by analysing patterns based on the person's facial textures and shape.
  It also record and store information of entries made by persons.

A arduino included which automatically recognises if someone on door with
ultrasonic sensor and using machine learning it classifies between one the
owner or unknown and manages database for visitors.
