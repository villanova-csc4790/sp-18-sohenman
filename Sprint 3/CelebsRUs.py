import os
import sys
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk

class m4in:
    window =""
    knownPeople ={}
    userPic =""
    celebPic =""
    dir =os.path.dirname(os.path.abspath(__file__))
    db =os.path.join(dir, "database")
    casc =cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
    recognizer =cv.face.LBPHFaceRecognizer_create()

    def __init__(self, w):
        self
        window =w

    def learn(self):
        #global knownPeople
        pictures =[]  #region of face to compare
        names =[]  #people names from the database
        personId =0
        count =0
        kpTemp ={} #TODO: If broken remove this and change back to knownPeople
        # iterate through the database looking for images
        for root, dirs, files in os.walk(self.db):
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    p =os.path.join(root, file)
                    # get name
                    person =os.path.basename(os.path.dirname(p))
                    print(person)
                    # Creates kv for database
                    if person not in kpTemp:
                        kpTemp[person] =personId
                        personId +=1
                    id =kpTemp[person]

                    pImage =Image.open(p).convert('L')  # makes it greyscale
                    scaleImage =pImage.resize((400, 400), Image.ANTIALIAS)
                    picArray =np.array(scaleImage, 'uint8')  # converts image into numbers for comparsion

                    #detects face within the image
                    faces =self.casc.detectMultiScale(picArray, scaleFactor=1.1, minNeighbors=5)
                    if (len(faces) !=1):
                        count +=1
                        faces =self.casc.detectMultiScale(picArray, scaleFactor=1.2, minNeighbors=5)
                        if (len(faces) !=0):
                            count =count -1
                    print(len(faces))
                    for (x, y, w, h) in faces:
                        face =picArray[y:y +h, x:x +w]
                        pictures.append(face)
                        names.append(id)
        print("Pictures failed : ", count)
        print(kpTemp)
        self.recognizer.train(pictures, np.array(names))
        self.knownPeople = {v: k for k, v in kpTemp.items()}
        print("Training done.")

    # TODO: Add logo pop up, Add removal of message, activate compare button
    def upload(self):
        #global userPic
        self.userPic =askopenfilename()
        print(self.userPic)

        me = tk.Label(self.window, text=self.userPic)
        me.place(relx=0.27, rely=0.42, anchor=tk.CENTER)

    def showCeleb(self):
        celeb = tk.Label(self.window, text=self.celebPic)
        celeb.place(relx=0.73, rely=0.42, anchor=tk.CENTER)

    def compare(self):
        #global celebPic
        print(self.userPic)
        print (self.knownPeople)
        image = cv.imread(self.userPic)
        print (image)
        grey =cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #TODO: Try scaling the pics for better results?
        face =self.casc.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5)

        if (len(face) ==1):

            for (x, y, w, h) in face:
                # print(x, y, w, h)
                theFace =grey[y:y +h, x:x +w]  # region of interest

                id, confidence =self.recognizer.predict(theFace)

                self.celebPic =self.knownPeople[id]
                print(self.knownPeople[id])
                print(confidence)

            self.showCeleb()
        else:
            print("need better picture")

        #TODO: Add else to tell user to upload better image

    def end(event=None):
        #window.destroy
        sys.exit()

class runner:
    window =tk.Tk()
    m =m4in(window)
    window.title("Celebs R Us")
    window.geometry("900x550")
    window.resizable(0, 0)

    uploadBtn =tk.Button(window, text="Upload Image", width="10", command =m.upload)
    uploadBtn.place(relx=0.3, rely=0.85, anchor=tk.CENTER)

    compareBtn = tk.Button(window, text="Compare", width="10", command=m.compare)
    compareBtn.place(relx=0.5, rely=0.85, anchor=tk.CENTER)

    ranBtn =tk.Button(window, text="Learn", width="10", command=m.learn)
    ranBtn.place(relx=0.7, rely=0.85, anchor=tk.CENTER)

    window.mainloop()