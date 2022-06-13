import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras

# loading models
haar = cv2.CascadeClassifier('./static/model/haarcascade_frontalface_default.xml')
classification_model = tf.keras.models.load_model('static/model/vgg16_model_1.h5')
print('Model loaded sucessfully ...')

# Settings
font = cv2.FONT_HERSHEY_SIMPLEX
class_names=['noStress','stress']


# Create a function to import an image and resize it to be able to be used with our model
def prep_image(img, img_shape=224):
  """
  Reads an image from memory, turns it into a tensor
  and reshapes it to (img_shape, img_shape, colour_channel).
  """
  # Convert image to RGB color_mode
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  # Resize the image (to the same size our model was trained on)  
  img = tf.image.resize(img, size = [img_shape, img_shape])
  # Adjust the image for batch format
  img = tf.expand_dims(img, axis=0)
  # Normalize image
  img = img/255.  
  return img


def pred_and_plot(model, image, class_names):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = prep_image(image)
    # Make a prediction
    pred = model.predict(img)
    # Get the predicted class
    pred_class = class_names[int(tf.round(pred)[0][0])]
    return pred_class

def face_detect(img, color='bgr'):
    if color == 'bgr':
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    faces = haar.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        # Crop image
        predict_img = img[y:y+h,x:x+h] # crop image
        # Make a prediction
        prediction = pred_and_plot(classification_model, predict_img,class_names)
        # Draw face rectangle
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),3)        
        # Write Prediction text
        text = f"Prediction: {prediction}"
        cv2.putText(img,text,(x,y),font,1,(0,255,0),2)
    return img

def classify_image(path, filename, color='bgr'):
    # step-1: read image in cv2
    img = cv2.imread(path) 
    # step-2: detect faces
    img = face_detect(img)
    # Save prediction image
    cv2.imwrite('./static/predict/{}'.format(filename),img)

    
def classify_video(path,status, color='bgr'):
    # step-1: read video in cv2
    print('Classifing Video .....')
    if status == 'stream':
        cap = cv2.VideoCapture(0) # enable video capture from stream
    else:
        cap = cv2.VideoCapture(path) # enable video capture from file

    while True:
        success,frame = cap.read() # read the video file
        if not success:
            break
        else:
            # step-2: detect faces
            frame = face_detect(frame)            
            # encode frames in JPG format (file extension, buffer)
            ret,buffer=cv2.imencode('.jpeg',frame)
            # convert buffer back to bytes
            frame = buffer.tobytes()        
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')            
    print('Closing Video .....')
    cap.release()