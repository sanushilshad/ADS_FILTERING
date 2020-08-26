from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import os
import tkinter as tk
from itertools import cycle
from PIL import ImageTk, Image
from itertools import cycle
from PIL import ImageTk
from PIL import Image
import tensorflow as tf
from sa import clothes_detector
import sa
from tkinter import messagebox
"""import matplotlib.pyplot as plt"""
from tensorflow import keras
""" print(tf.__version__)"""
pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
variables=['','','']



def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))


def main():


    
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file
    margin = args.margin
    image_dir = args.image_dir

    if not weight_file:
        weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()

    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)
        
        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # draw results

            
                
            for i, d in enumerate(detected):
                 
                    #label = "{}, {}".format(int(predicted_ages[i]),
                                            #"M" if predicted_genders[i][0] < 0.5 else "F")
                    label = "{}".format("Detected")
                    draw_label(img, (d.left(), d.top()), label)
                    
                    variables[0]=predicted_ages[i]
                    variables[1]="M" if predicted_genders[i][0] < 0.5 else "F"
                    
                    
                    
            if (variables[0]!='' or variables[1]!=''):
                cv2.imshow("result", img)
                key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)
                
                cv2.imwrite('image'+'.jpg',img)
                cv2.waitKey(1)
                variables[2]=sa.clothes_detector('image.jpg')
                print(variables[0],":",variables[1],":",variables[2])
                cv2.destroyAllWindows()
                break
            
                
        cv2.imshow("result", img)
        key = cv2.waitKey(1) if image_dir else cv2.waitKey(30)

        if key == 27:  # ESC
            break

        

main()
entries=""    
print(variables[0],variables[1],variables[2])
a=os.getcwd()
if (variables[1]=='M'):
    if(int(variables[0])<10):
        if(variables[2]=="T-shirt/top"):
            print(a+"\Advertisements\Male\Kids\T-shirt_top")
            entries= os.listdir(a+"\Advertisements\Male\Kids\T-shirt_top")
            e1=a+"\Advertisements\Male\Kids\T-shirt_top/"
        elif(variables[2]=='Coat'):
            print(a+"\Advertisements\Male\Kids\Coat")
            entries= os.listdir(a+"\Advertisements\Male\Kids\Coat")
            e1=a+"\Advertisements\Male\Kids\Coat/"

        elif(variables[2]=='Shirt'):
            print(a+"\Advertisements\Male\Kids\Shirts")
            entries= os.listdir(a+"\Advertisements\Male\Kids\Shirts")
            e1=a+"\Advertisements\Male\Kids\Shirts/"
        

    elif(int(variables[0])>10 and int(variables[0])<18):
        if(variables[2]=="T-shirt/top"):
            print(a+"\Advertisements\Mal\Teenagers\T-shirt_top")
            entries= os.listdir(a+"\Advertisements\Male\Teenagers\T-shirt_top")
            e1=a+"\Advertisements\Male\Teenagers\T-shirt_top/"
        elif(variables[2]=='Coat'):
            print(a+"\Advertisements\Mal\Teenagers\Coat")
            entries= os.listdir(a+"\Advertisements\Male\Teenagers\Coat")
            e1=a+"\Advertisements\Male\Teenagers\Coat/"

        elif(variables[2]=='Shirt'):
            print(a+"\Advertisements\Mal\Teenagers\Shirts")
            entries= os.listdir(a+"\Advertisements\Male\Teenagers\Shirts")
            e1=a+"\Advertisements\Male\Teenagers\Shirts/"
        


    elif(int(variables[0])>18 and int(variables[0])<55):
        if(variables[2]=="T-shirt/top"):
            print(a+"\Advertisements\Male\Adults\T-shirt_top")
            entries= os.listdir(a+"\Advertisements\Male\Adults\T-shirt_top")
            e1=a+"\Advertisements\Male\Adults\T-shirt_top/"

        elif(variables[2]=='Coat'):
            print(a+"\Advertisements\Male\Adults\Coat")
            entries= os.listdir(a+"\Advertisements\Male\Adults\Coat")
            e1=a+"\Advertisements\Male\Adults\Coat/"

        elif(variables[2]=='Shirt'):
            print(a+"\Advertisements\Male\Adults\Shirts")
            entries= os.listdir(a+"\Advertisements\Male\Adults\Shirts")
            e1=a+"\Advertisements\Male\Adults\Shirts/"

    elif(int(variables[0])>55):
        if(variables[2]=="T-shirt/top"):
            print(a+"\Advertisements\Male\Old people\T-shirt_top")
            entries= os.listdir(a+"\Advertisements\Male\Old people\T-shirt_top")
            e1=a+"\Advertisements\Male\Old people\T-shirt_top/"

        elif(variables[2]=='Coat'):
            print(a+"\Advertisements\Male\Old people\Coat")
            entries= os.listdir(a+"\Advertisements\Male\Old people\Coat")
            e1=a+"\Advertisements\Male\Old people\Coat/"

        elif(variables[2]=='Shirt'):
            print(a+"\Advertisements\Male\Old people\Shirts")
            entries= os.listdir(a+"\Advertisements\Male\Old people\Shirts")
            e1=a+"\Advertisements\Male\Old people\Shirts/"

elif (variables[1]=='F'):
    if(int(variables[0])<10):
        if(variables[2]=="T-shirt/top"):
            print(a+"\Advertisements\Female\Kids\T-shirt_top")
            entries= os.listdir(a+"\Advertisements\Female\Kids\T-shirt_top")
            e1=a+"\Advertisements\Female\Kids\T-shirt_top/"

        elif(variables[2]=='Shirt'):
            print(a+"\Advertisements\Female\Kids\Shirt")
            entries= os.listdir(a+"\Advertisements\Female\Kids\Shirt")
            e1=a+"\Advertisements\Female\Kids\Shirt/"


        elif(variables[2]=='Dress'):
            print(a+"\Advertisements\Female\Kids\Dress")
            entries= os.listdir(a+"\Advertisements\Female\Kids\Dress")
            e1=a+"\Advertisements\Female\Kids\Dress/"



    elif(int(variables[0])>10 and int(variables[0])<18):

        if(variables[2]=="T-shirt/top"):
            print(a+"\Advertisements\Female\Teenagers\T-shirt_top")
            entries= os.listdir(a+"\Advertisements\Female\Teenagers\T-shirt_top")
            e1=a+"\Advertisements\Female\Teenagers\Kids\T-shirt_top/"

        elif(variables[2]=='Shirt'):
            print(a+"\Advertisements\Female\Teenagers\Shirt")
            entries= os.listdir(a+"\Advertisements\Female\Teenagers\Shirt")
            e1=a+"\Advertisements\Female\Teenagers\Kids\Shirt/"



        elif(variables[2]=='Dress'):
            print(a+"\Advertisements\Female\Teenagers\Dress")
            entries= os.listdir(a+"\Advertisements\Female\Teenagers\Dress")
            e1=a+"\Advertisements\Female\Teenagers\Kids\Dress/"



    elif(int(variables[0])>18 and int(variables[0])<55):
        if(variables[2]=="T-shirt/top"):
            print(a+"\Advertisements\Female\Adults\T-shirt_top")
            entries= os.listdir(a+"\Advertisements\Female\Adults\T-shirt_top")
            e1=a+"\Advertisements\Female\Adults\T-shirt_top/"

        elif(variables[2]=='Shirt'):
            print(a+"\Advertisements\Female\Adults\Shirt")
            entries= os.listdir(a+"\Advertisements\Female\Adults\Shirt")
            e1=a+"\Advertisements\Female\Adults\Shirt/"



        elif(variables[2]=='Dress'):
            print(a+"\Advertisements\Female\Adults\Dress")
            entries= os.listdir(a+"\Advertisements\Female\Adults\Dress")
            e1=a+"\Advertisements\Female\Adults\Dress/"


        
    elif(int(variables[0])>55):

        if(variables[2]=="T-shirt/top"):
            print(a+"\Advertisements\Female\Old people\T-shirt_top")
            entries= os.listdir(a+"\Advertisements\Female\Old people\T-shirt_top")
            e1=a+"\Advertisements\Female\Old people\T-shirt_top/"

        elif(variables[2]=='Shirt'):
            print(a+"\Advertisements\Female\Old people\Shirt")
            entries= os.listdir(a+"\Advertisements\Female\Old people\Shirt")
            e1=a+"\Advertisements\Female\Old people\Shirt/"



        elif(variables[2]=='Dress'):
            print(a+"\Advertisements\Female\Old people\Dress")
            entries= os.listdir(a+"\Advertisements\Female\Old people\Dress")
            e1=a+"\Advertisements\Female\Old people\Dress/"
        

print("Entries")

photos = cycle(ImageTk.PhotoImage(Image.open(e1+image),master=root) for image in entries)

def slideShow():
  img = next(photos)
  
  displayCanvas.config(image=img)
  root.after(1000, slideShow) # 0.05 seconds

root = tk.Tk()

width = 500
height = 400
root.geometry('%dx%d' % (640, 480))
displayCanvas = tk.Label(root)
displayCanvas.pack()
root.after(10, lambda: slideShow())
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)


root.mainloop()

"""
class App(tk.Tk):
    '''Tk window/label adjusts to size of image'''
    def __init__(self,  x, y, delay):
        # the root will be self
        tk.Tk.__init__(self,)
        # set x, y position only
        self.geometry('400x300')
        self.delay = delay
       
        # allows repeat cycling through the pictures
        # store as (img_object, img_name) tuple
        self.pictures = cycle((ImageTk.PhotoImage(e1+image, master=root), image)
                              for image in entries)
        self.picture_display = tk.Label(self)
        self.picture_display.pack()

        
    def show_slides(self):
        '''cycle through the images and show them'''
        # next works with Python26 or higher
        img_object, img_name = next(self.pictures)
       
        self.picture_display.config(image=img_object)
        # shows the image filename, but could be expanded
        # to show an associated description of the image
        self.title(img_name)
        self.after(self.delay, self.show_slides)
    def run(self):
        self.mainloop()
# set milliseconds time between slides
delay = 3500

x = 10
y = 10
app = App( x, y, delay)
app.show_slides()
app.run()
"""