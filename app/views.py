from flask import render_template,request,Response
from flask import redirect,url_for
import os
from PIL import Image
from app.utils import classify_image,classify_video
#import cv2

# Settings
UPLOAD_FOLDER = 'static/uploads'
image_types = ['bmp','jpeg','jpg','jpe','jp2','png','tiff','tif','hdr','pic']
video_types=['aiff','asf','avi','bfi','caf','flv','gif','gxf','hls','iff','mp4','mov','qt','3gp','mkv','mk3d','mka','mks','mpeg','mpg','mxf','nut','ogg','oma','rl2','txd','wtv']

def base():
    return render_template("base.html")

def index():
    return render_template("index.html")

def stressapp():
    return render_template("stressapp.html")

def image():
    return render_template("image.html")

def video(filename,status):
    if request.method == "GET":
        print('Loading video ........ ')
        path = './static/uploads/'+filename
        print('Loading path .....', path)
        print('Loading status .....', status)
        return Response(classify_video(path,status,'bgr'), mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template('stress.html',fileupload=False,img_name="imageNotFound.png")


def getwidth(path):
    img = Image.open(path)
    size = img.size #width and height
    aspect = size[0]/size[1]
    w = 300 * aspect
    return int(w)

def stress():
    if request.method == "POST":
        status = request.form['btnradio']
        if status == 'stream':
            return render_template('stress.html',fileupload=True,type='video',filename='null',status=status)
        else:
            f = request.files['image']
            filename=  f.filename
            file_type = filename.lower().split('.')[1]
            path = os.path.join(UPLOAD_FOLDER,filename)
            f.save(path)
            print('Uploading file ......')

            # preditions (pass to pipeline models)
            if file_type in image_types:
                w = getwidth(path)
                classify_image(path, filename,'bgr')
                return render_template('stress.html',fileupload=True,type='image', img_name=filename,w=w)
            
            if file_type in video_types:
                return render_template('stress.html',fileupload=True,type='video',filename=filename,status=status)
        
    return render_template('stress.html',fileupload=False,img_name="imageNotFound.png")
    
