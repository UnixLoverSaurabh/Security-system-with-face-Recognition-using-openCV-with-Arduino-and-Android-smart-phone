import serial
from vpython import *
import cv2
import os
import numpy as np
import shutil
import time

arduinoSerialData = serial.Serial('/dev/ttyACM0', 9600)
subjects = ["", "Saurabh", "Smart saurabh"]
#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    #face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    #display an image window to show the image 
    cv2.imshow("Training on image...", cv2.resize(img, (400, 500)))
    # waitKey(interval) pauses the code flow for the given interval (milliseconds), so that we can view the image window for that time.
    cv2.waitKey(100)


    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]



#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        
        if not dir_name.startswith("s"):
            continue;

        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        
        #build path of directory containing images for current subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            
            #detect face
            face, rect = detect_face(image)
            
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels


#let's first prepare our training data
#data will be in two lists of same size
#one list will contain all the faces and other list will contain respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


# ========== Train Face Recognizer ===========
# Create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()


# It's time to train the face recognizer. We will do that by calling the `train(faces-vector, labels-vector)` method of face recognizer.
face_recognizer.train(faces, np.array(labels))
# Instead of passing vectorlabels directly to face recognizer, we are first converting it to numpy array? 
# The reason is that OpenCV expects labels vector to be a numpyarray.



# ========== Prediction =================
# This is where we actually get to see if our algorithm is actually recognizing our trained subjects's faces or not.
# We will take two test images of our celeberities, detect faces from each of them and then pass those faces to our trained face recognizer to see if it recognizes them. 

# Function to draw rectangle on image 
#according to given (x, y) coordinates and given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.rectangle(img, topLeftPoint, bottomRightPoint, rgbColor, lineWidth)
    
#function to draw text on give image starting from passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    # cv2.putText(img, text, startPoint, font, fontSize, rgbColor, lineWidth)

# This function recognizes the person in image passed
# and draws a rectangle around detected face with name of the subject
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    #  "confidence", is actually the opposite - the distance to the closest item in the database.
    # 100 is better than 200, and 0 would be a "perfect match"
    print("Confidence is " + str(confidence))
    #get name of respective label returned by face recognizer
    if confidence > 50:
    	label_text = "unknown"
    else:
    	label_text = subjects[label]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img, label_text


path = '/run/user/1000/gvfs/mtp:host=%5Busb%3A001%2C012%5D/Internal storage/DCIM/Camera/'
#path = '/home/saurabh/Desktop/sunil/cam/'
while (1==1):  #Create a loop that continues to read and display the data
	if (arduinoSerialData.inWaiting()>0):  #Check to see if a data point is available on the serial port
		myData = int(arduinoSerialData.readline()) #Read the distance measure as a string and convert to int
		if myData == 1:
			print(myData) #Print the measurement to confirm things are working
			count = 0
			#load test images
			for j in os.listdir(path):
				count += 1
				print(j)
			if count > 0:
				time.sleep(5)
				shutil.copy2(path + j, '/home/saurabh/Desktop/sunil/test-data')

				print("Predicting images...")
				test_img = cv2.imread(path + j)
				#perform a prediction
				predicted_img, label_text = predict(test_img)
				print("Prediction complete")
				cv2.destroyAllWindows()

				#display the predicted image
				cv2.imshow(label_text, cv2.resize(predicted_img, (600, 800)))
				cv2.waitKey(0)
				cv2.destroyAllWindows()
				os.remove(path + j)
