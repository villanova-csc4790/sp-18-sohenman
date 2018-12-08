#Senior Project 2018: last edit =12-7-18 -Mav

import os
import sys
import pickle
import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox as mb
from tkinter.filedialog import askopenfilename

class Medtodi:
    window =""
    uploadBtn =""
    compareBtn =""
    learnBtn =""
    logoPic =""
    logo =""
    celebNT =""
    user =""
    celeb =""
    arrowPic =""
    arrow =""
    tip =""
    knownPeople ={}
    userPic =""
    celebPic =""
    dir =os.path.dirname(os.path.abspath(__file__))
    db =os.path.join(dir, "supportingFiles/database")
    dbYml ="supportingFiles/dbFiles/db.yml"
    dbMav ="supportingFiles/dbFiles/dbNames.mav"
    casc =cv.CascadeClassifier('supportingFiles/cascades/data/haarcascade_frontalface_alt.xml')
    recognizer =cv.face.LBPHFaceRecognizer_create()

    def __init__(self, w, uB, cB, lB, lP, l, cNT, u, c, aP, a, t):
        self.window =w
        self.uploadBtn =uB
        self.compareBtn =cB
        self.learnBtn =lB
        self.logoPic =lP
        self.logo =l
        self.celebNT =cNT
        self.user =u
        self.celeb =c
        self.arrowPic =aP
        self.arrow =a
        self.tip =t
        
        # checks to make sure db is accessible
        if (os.path.isfile(self.dbYml) and os.path.isfile(self.dbMav)):
            print("here")
            self.recognizer.read(self.dbYml)
            with open(self.dbMav, 'rb') as file:
                self.knownPeople =pickle.load(file)
        else:
            self.uploadBtn.config(state=tk.DISABLED)
            self.compareBtn.config(state=tk.DISABLED)

    def learn(self):
        learnWindow =tk.Toplevel()
        learnWindow.title("Celebs \"R\" Us - Learning")
        learnWindow.geometry("500x500")
        learnWindow.resizable(0, 0)
        learnWindow.lift()
        learnWindow.attributes('-topmost', True)

        self.learnBtn.config(state=tk.DISABLED)
        self.uploadBtn.config(state=tk.DISABLED)
        self.compareBtn.config(state=tk.DISABLED)
        self.window.update()

        dbImg =tk.Label(learnWindow, image="", bg="black")
        dbImg.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
        dbName =tk.Label(learnWindow, text="")
        dbName.place(relx=0.5, rely=0.85, anchor=tk.CENTER)

        pictures =[]  #region of face to compare
        names =[]  #people names from the database
        personId =0
        count =0
        kpTemp ={}

        # iterate through the database looking for images
        for root, dirs, files in os.walk(self.db):
            for file in files:
                if file.endswith("jpeg") or file.endswith("png") or file.endswith("jpg"):
                    p =os.path.join(root, file)
                    print(p)
                    # get name
                    person =os.path.basename(os.path.dirname(p))
                    print(person)

                    # Creates kv for database
                    if person not in kpTemp:
                        dbName.config(text=person)
                        kpTemp[person] =personId
                        personId +=1
                    id =kpTemp[person]

                    #show user
                    dbPic =Image.open(p)
                    showDBPic =dbPic.resize((325, 325), Image.ANTIALIAS)
                    tkDbPic =ImageTk.PhotoImage(showDBPic)
                    dbImg.config(image=tkDbPic)
                    dbImg.image =tkDbPic
                    learnWindow.update()

                    pImage =cv.imread(p)
                    bImage =cv.GaussianBlur(pImage,(7, 7),0)
                    gImage = cv.cvtColor(bImage, cv.COLOR_BGR2GRAY)

                    #detects face within the image
                    theFace =self.casc.detectMultiScale(gImage, scaleFactor=1.1, minNeighbors=5)
                    if (len(theFace) !=1):
                        count +=1
                        theFace =self.casc.detectMultiScale(gImage, scaleFactor=1.3, minNeighbors=5)
                        if (len(theFace) !=0):
                            count =count -1
                    print(len(theFace))
                    for (x, y, w, h) in theFace:
                        face =gImage[y:y +h, x:x +w]
                        pictures.append(face)
                        names.append(id)

        print("Pictures failed : ", count)
        print(kpTemp)

        self.recognizer.train(pictures, np.array(names))
        self.recognizer.save(self.dbYml)
        self.knownPeople ={v: k for k, v in kpTemp.items()}
        with open(self.dbMav, 'wb') as file:
            pickle.dump(self.knownPeople, file)

        self.uploadBtn.config(state="normal")
        if (self.userPic):
            self.compareBtn.config(state="normal")
        self.learnBtn.config(state="normal")
        self.window.update()
        learnWindow.destroy()
        print("Training done.")

    def upload(self):
        self.userPic =askopenfilename()
        print(self.userPic)
        if (self.userPic):

            uPhoto =Image.open(self.userPic)
            uPhoto =uPhoto.resize((325, 325), Image.ANTIALIAS)
            tkUPic =ImageTk.PhotoImage(uPhoto)
            print(tkUPic)
            self.user.config(image=tkUPic, bg="black")
            self.user.image =tkUPic

            if (self.compareBtn['state'] ==tk.DISABLED):
                print ("inside upload")
                self.compareBtn.config(state="normal")
                self.moveLogo()

                tkAPic =ImageTk.PhotoImage(self.arrowPic.rotate(270))
                self.arrow.config(image=tkAPic)
                self.arrow.place(relx=0.59, rely=.8, anchor=tk.CENTER)
                self.arrow.image =tkAPic

                self.tip.config(text="Next, click \"compare\" to\nsee the celebrity doppelganger")
                self.tip.place(relx=0.65, rely=.70, anchor=tk.CENTER)
            self.window.update()

    def showCeleb(self):
        if (os.path.isfile("supportingFiles/database/" +self.celebPic +"/1.jpg")):
            cPhoto = Image.open("supportingFiles/database/" + self.celebPic + "/1.jpg")
        else:
            cPhoto = Image.open("supportingFiles/database/" + self.celebPic + "/1.png")
        cPhoto =cPhoto.resize((325, 325), Image.ANTIALIAS)
        tkCPic =ImageTk.PhotoImage(cPhoto)
        self.celeb.config(image=tkCPic, bg="black")
        self.celeb.image =tkCPic
        self.celebNT.config(text=self.celebPic)

    def compare(self):
        sF =1.3
        image =cv.imread(self.userPic)
        bImage = cv.GaussianBlur(image, (7, 7), 0)
        sImage = cv.resize(bImage, (375, 375), interpolation=cv.INTER_CUBIC)
        gImage = cv.cvtColor(sImage, cv.COLOR_BGR2GRAY)

        face =self.casc.detectMultiScale(gImage, scaleFactor=sF, minNeighbors=5)
        while (len(face) != 1 and sF <10.0):
            sF =sF +.1
            face = self.casc.detectMultiScale(gImage, scaleFactor=sF, minNeighbors=5)
            print(sF)


        if (len(face) ==1):

            if (self.celebPic ==""):
                self.arrow.config(state=tk.DISABLED)
                self.tip.config(text="")

            for (x, y, w, h) in face:
                theFace =gImage[y:y +h, x:x +w]  # region of interest
                id, confidence =self.recognizer.predict(theFace)
                self.celebPic =self.knownPeople[id]
                print(self.knownPeople[id])
                print(confidence)

            self.showCeleb()
        else:
            mb.showerror('Error', 'Please enter a better picture.')

    def moveLogo(self):
        self.logoPic =self.logoPic.resize((250, 75), Image.ANTIALIAS)
        tkLPic =ImageTk.PhotoImage(self.logoPic)
        self.logo.config(image=tkLPic)
        self.logo.place(relx=0.14, rely=0.07, anchor=tk.CENTER)
        self.logo.image =tkLPic

    def end(self):
        sys.exit()

class M4in:

    window =tk.Tk()
    window.title("Celebs \"R\" Us")
    window.geometry("900x550")
    window.resizable(0, 0)

    logoPic =Image.open("supportingFiles/uiImages/logo.png")
    tkLPic =ImageTk.PhotoImage(logoPic)
    logo =tk.Label(window, image=tkLPic)
    logo.place(relx=0.5, rely=0.42, anchor=tk.CENTER)
    logo.image =tkLPic

    arrowPic =Image.open("supportingFiles/uiImages/a1.jpg")
    arrowPic =arrowPic.resize((50, 50), Image.ANTIALIAS)
    tkAPic =ImageTk.PhotoImage(arrowPic)
    arrow =tk.Label(window, image=tkAPic)
    arrow.place(relx=0.21, rely=.8, anchor=tk.CENTER)
    arrow.image =tkAPic

    tip =tk.Label(window, text="Start by\nuploading a photo", font=("Courier", 18))
    tip.place(relx=0.12, rely=.70, anchor=tk.CENTER)

    uploadBtn =tk.Button(window, text="Upload Image", width="10")
    uploadBtn.place(relx=0.3, rely=0.87, anchor=tk.CENTER)
    compareBtn =tk.Button(window, text="Compare", width="10",
                          state=tk.DISABLED)
    compareBtn.place(relx=0.5, rely=0.87, anchor=tk.CENTER)
    learnBtn =tk.Button(window, text="Learn", width="10")
    learnBtn.place(relx=0.7, rely=0.87, anchor=tk.CENTER)

    user =tk.Label(window, image="")
    user.place(relx=0.27, rely=0.43, anchor=tk.CENTER)

    celeb =tk.Label(window, image="")
    celeb.place(relx=0.73, rely=0.43, anchor=tk.CENTER)

    celebNT =tk.Label(window, text="", font=("Courier", 18))
    celebNT.place(relx=0.73, rely=0.78, anchor=tk.CENTER)

    m =Medtodi(window, uploadBtn, compareBtn, learnBtn,
               logoPic, logo, celebNT, user, celeb,
               arrowPic, arrow, tip)

    uploadBtn.configure(command =m.upload)
    compareBtn.configure(command =m.compare)
    learnBtn.configure(command=m.learn)

    window.lift()
    window.attributes('-topmost', True)
    window.after_idle(window.attributes, '-topmost', False)
    window.mainloop()