#Importing the required modules
import tkinter as tk
from tkinter import *
import easygui
import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from PIL import Image,ImageTk
from scipy.interpolate import UnivariateSpline

#Making the GUI main window 
top = tk.Tk()
top.geometry('400x400')
top.title('Cartoonify Your Image!')
top.configure(background='blue')
label = Label(top, background='#CDCDCD', font=('calibri', 20, 'bold'))

#Setting the background of the main window
load=Image.open("C:\\Users\\niles\\Desktop\\Cartoonifying an Image\\background.jpg")
render=ImageTk.PhotoImage(load)
bgt=Label(top,image=render)
bgt.place(x=0,y=0)

#Function to display image
def display(Image,title):
    plt.imshow(Image,cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.show()

#Function to upload image from device for cartoonifying
def upload():
    ImagePath = easygui.fileopenbox()
    img=cv2.imread(ImagePath)
    if img is None:
        print("Could not find any image, choose appropriate file.")
    cartoonify(img, ImagePath)

#Function to open camera and take image for applying effects
def camera():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    while img_counter==0:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("test", img)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()
    ImagePath = "C:\\Users\\niles\\Desktop\\Cartoonifying an Image\\background.jpg"
    cartoonify(img, ImagePath)

#Function to open camera and take image for applying effects
def cam2():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    while img_counter==0:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("test", img)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()
    ImagePath = "C:\\Users\\niles\\Desktop\\Cartoonifying an Image\\background.jpg"
    tryEffects(img)

#Function to upload image from device for more effects
def up2():
    ImagePath = easygui.fileopenbox()
    img=cv2.imread(ImagePath)
    if img is None:
        print("Could not find any image, choose appropriate file.")
    tryEffects(img)

#Funtion for saving an image with the provided path
def save(Image, ImagePath):
    newName="Cartoonified_Image"
    path1 = os.path.dirname(ImagePath)
    extension=os.path.splitext(ImagePath)[1]
    path = os.path.join(path1, newName+extension)
    cv2.imwrite(path, cv2.cvtColor(Image, cv2.COLOR_RGB2BGR))
    I= "Image saved by name " + newName +" at "+ path
    tk.messagebox.showinfo(title=None, message=I)

#Funtion for converting image to pencil scketch (both colored and grey)
def pencil_sketch(img1):
    sk_gray, sk_color = cv2.pencilSketch(img1, sigma_s=60, sigma_r=0.07, shade_factor=0.1) 
    sketches=[sk_gray, sk_color]
    fig, axes = plt.subplots(1,2, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sketches[i], cmap='gray')
    plt.show()

#Funtion for Apllying Sepia Filter to image (Adding a little red-brownish color)
def sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    img_sepia = cv2.filter2D(img, -1, kernel)
    display(img_sepia,"Sepia Filtered Image")

#Funtion for converting image to Grey image
def greyscale(img):
    greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display(greyImg,"Gray Image")

#Funtion for Brigtening image
def bright(img):
    img_bright = cv2.convertScaleAbs(img, beta=50)
    display(img_bright,"Brightened Image")

#Funtion for Sharpen image
def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)
    display(img_sharpen,"Sharpen Image")

#Funtion for converting image to HDR image
def HDR(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    display(hdr,"HDR Image")

#Funtion for Inverting image
def invert(img):
    inv = cv2.bitwise_not(img)
    display(inv,"Inverted Image")

def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))

#Funtion for converting image to Warm image
def Winter(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv2.merge((blue_channel, green_channel, red_channel ))
    display(sum,"Warm Image")

#Funtion for converting image to Cold image
def Summer(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))
    display(win,"Cold Image")

#Making the Effect list GUI window
def tryEffects(Orig):

    #Getting the orignal image (maintaining the color space)
    Orig = cv2.cvtColor(Orig, cv2.COLOR_BGR2RGB)

    #Making the GUI window 
    td = tk.Tk()
    td.geometry('400x400')
    td.title('Effects')
    td.configure(background='grey')
    label3 = Label(td, background='#CDCDCD', font=('calibri', 20, 'bold'))

    #Button for Pencil Sketch
    PencilSketch = Button(td, text="Pencil Sketch", command=lambda: pencil_sketch(Orig), padx=10, pady=5)
    PencilSketch.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    PencilSketch.pack(side=TOP, pady=1)

    #Button for Sepia Filter
    Sepia = Button(td, text="Sepia", command=lambda: sepia(Orig), padx=10, pady=5)
    Sepia.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    Sepia.pack(side=TOP, pady=2)

    #Button for Brighten Image
    BrightI = Button(td, text="Brighten Image", command=lambda: bright(Orig), padx=10, pady=5)
    BrightI.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    BrightI.pack(side=TOP, pady=3)

    #Button for Sharpen Image
    SharpenI = Button(td, text="Sharpen Image", command=lambda: sharpen(Orig), padx=10, pady=5)
    SharpenI.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    SharpenI.pack(side=TOP, pady=4)

    #Button for Grey Image
    GrayI = Button(td, text="GreyScaled Image", command=lambda: greyscale(Orig), padx=10, pady=5)
    GrayI.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    GrayI.pack(side=TOP, pady=5)

    #Button for HDR Image
    hdrI = Button(td, text="HDR Image", command=lambda: HDR(Orig), padx=10, pady=5)
    hdrI.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    hdrI.pack(side=TOP, pady=6)

    #Button for Inverted Image
    InvertI = Button(td, text="Inverted Image", command=lambda: invert(Orig), padx=10, pady=5)
    InvertI.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    InvertI.pack(side=TOP, pady=7)

    #Button for Warm Image
    Warm = Button(td, text="Warm Image", command=lambda: Summer(Orig), padx=10, pady=5)
    Warm.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    Warm.pack(side=TOP, pady=8)

    #Button for Cold Image
    Cold = Button(td, text="Cold Image", command=lambda: Winter(Orig), padx=10, pady=5)
    Cold.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    Cold.pack(side=TOP, pady=9)
    
    td.mainloop()

#Making the second GUI window (For more effects)
def effects():

    #Making the Second window 
    down = tk.Tk()
    down.geometry('400x400')
    down.title('Try More Effects!')
    down.configure(background='Green')
    label = Label(down, background='#CDCDCD', font=('calibri', 20, 'bold'))

    #Making the Camera button in the second window
    camera = Button(down, text="Open Camera", command=cam2, padx=10, pady=5)
    camera.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    camera.pack(side=TOP, pady=50)

    #Making the upload button in the second window
    upload = Button(down, text="Choose Image", command=up2, padx=10, pady=5)
    upload.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
    upload.pack(side=TOP, pady=50)

    down.mainloop()

#Making the Camera button in the GUI main window
camera = Button(top, text="Open Camera", command=camera, padx=10, pady=5)
camera.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
camera.pack(side=TOP, pady=30)

#Making the Upload button in the GUI main window
upload = Button(top, text="Choose Image", command=upload, padx=10, pady=5)
upload.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
upload.pack(side=TOP, pady=30)

#Making the Try More Effects button in the GUI main window
tryEffectsButton = Button(top, text="Try More Effects", command=effects, padx=10, pady=5)
tryEffectsButton.configure(background="#374256", foreground="wheat", font=('calibri', 10, 'bold'))
tryEffectsButton.pack(side=TOP, pady=30)

#Funtion for cartoonifying an Image
def cartoonify(Orignal,ImagePath):

    #To maintain the color space of the orignal image
    Orignal=cv2.cvtColor(Orignal,cv2.COLOR_BGR2RGB)
    #Displaying Orignal Image
    display(Orignal,"Original Image")    

    #Converting the color space from RGB to Grayscale
    Grayed=cv2.cvtColor(Orignal,cv2.COLOR_RGB2GRAY)
    #Displaying Gray Image
    display(Grayed,"Gray Image")

    #Sharpening Image using filter2D function, by putting a filter using numpy array
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    Sharpen = cv2.filter2D(Grayed, -1, filter)
    #Displaying Sharpen Image
    display(Sharpen,"Sharpen Image")

    #Applying median blur to image
    Blurred=cv2.medianBlur(Grayed,5)
    #Displaying Blurred Image
    display(Blurred,"Median Blurred Image")

    #Creating edge mask
    line_size = 7
    blur_value = 7
    LightEdged = cv2.adaptiveThreshold(Blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    #Displaying Edge Masked Image
    display(LightEdged,"Light Edge Masked Image")

    #Creating edge mask (Dark)
    DarkEdged=cv2.Canny(Blurred, 100, 200)
    #Displaying Edge Masked Image
    display(DarkEdged,"Dark Edge Masked Image")

    #Applying bilateral filter to remove noise as required
    NoiseFree=cv2.bilateralFilter(Orignal, 15, 75, 75)
    #Displaying Noise Free Image
    display(NoiseFree,"Bilateral Blurred Image")

    #Eroding and Dilating
    kernel=np.ones((1,1),np.uint8)
    Eroded=cv2.erode(NoiseFree,kernel,iterations=3)
    Dilated=cv2.dilate(Eroded,kernel,iterations=3)
    #Displaying Eroded and Dilated Image
    display(Dilated,"Eroded & Dilated Image")

    #Implementing K-Means Clustering (For number of colors in the image)
    k = number_of_colors = 5
    temp=np.float32(Dilated).reshape(-1,3)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    compactness,label,center=cv2.kmeans(temp,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    final_img=center[label.flatten()]
    final_img=final_img.reshape(Orignal.shape)

    #Final Cartoon Image
    Final=cv2.bitwise_and(final_img,final_img,mask= LightEdged)
    display(Final,"FINAL")

    #For all transition Plot
    images=[Orignal, Grayed, Sharpen, Blurred, LightEdged, DarkEdged, NoiseFree, Eroded, Dilated, Final]
    fig, axes = plt.subplots(5,2, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
    plt.show()

    #Save Button
    saveButton=Button(top,text="Save Cartoonified image",command=lambda: save(Final, ImagePath),padx=30,pady=5)
    saveButton.configure(background='#374256', foreground='wheat',font=('calibri',10,'bold'))
    saveButton.pack(side=TOP,pady=30)
 
#Main function to build the GUI window
top.mainloop()







