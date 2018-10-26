import os
import sys
import cv2 as cv
import numpy as np
from PIL import Image

#Gets path of app and databse images
dir =os.path.dirname(os.path.abspath(__file__))
db =os.path.join(dir, "database")
casc =cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
recognizer =cv.face.LBPHFaceRecognizer_create()
#recognizer =cv.face.EigenFaceRecognizer_create()
#recognizer =cv.face.FisherFaceRecognizer_create()
knownPeople ={} #Database

class knowledge:
    def __init__(self):
        self

    def learn(self):
        pictures =[]  #region of face to compare
        names =[]  #people names from the database
        personId =0
        count =0

        # iterate through the database looking for images
        for root, dirs, files in os.walk(db):
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    p =os.path.join(root, file)
                    # get name
                    person =os.path.basename(os.path.dirname(p))
                    print(person)
                    # Creates kv for database
                    if person not in knownPeople:
                        knownPeople[person] =personId
                        personId +=1
                    id =knownPeople[person]

                    pImage =Image.open(p).convert('L')  # makes it greyscale
                    scaleImage =pImage.resize((400, 400), Image.ANTIALIAS)
                    picArray =np.array(scaleImage, 'uint8')  # converts image into numbers for comparsion

                    #detects face within the image
                    faces =casc.detectMultiScale(picArray, scaleFactor=1.1, minNeighbors=5)
                    if (len(faces) !=1):
                        count +=1
                        faces =casc.detectMultiScale(picArray, scaleFactor=1.2, minNeighbors=5)
                        if (len(faces) !=0):
                            count =count -1
                    print(len(faces))
                    for (x, y, w, h) in faces:
                        face =picArray[y:y +h, x:x +w]
                        pictures.append(face)
                        names.append(id)
        print("Pictures failed : ", count)
        print(knownPeople)
        recognizer.train(pictures, np.array(names))
        print("Training done.")

class m4in:

    run =True
    k = knowledge()
    k.learn()
    knownPeople ={v:k for k, v in knownPeople.items()}

    while run:
        i =input()

        if (i =="-1"):
            run =False
            sys.exit()
        elif (i =="relearn"):
            k.learn()
            # flips key value so we are able to search
            knownPeople ={v:k for k, v in knownPeople.items()}
        else:

            # To get pic have user select it
            pic ="trialPics/" +i +".jpg"

            image =cv.imread(pic)
            grey =cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            #TODO: Try scaling the pics for better results?
            face =casc.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5)

            if (len(face) ==1):
                for (x, y, w, h) in face:
                    # print(x, y, w, h)
                    theFace =grey[y:y +h, x:x +w]  # region of interest

                    id, confidence =recognizer.predict(theFace)
                    # while confidence <70:
                    #     id, confidence = recognizer.predict(theFace)
                    #     print(knownPeople[id])
                    #     print(confidence)
                    print(knownPeople[id])
                    print(confidence)
                    # cv.rectangle(image, (x, y), (x + w, y + h), (255, 75, 255), 2)  # draws rec

                # display image
                # cv.namedWindow('image', cv.WINDOW_NORMAL)
                # cv.resizeWindow('image', 600, 600)
                # cv.startWindowThread()
                # cv.imshow("image", image)
                # #cv.moveWindow('image', 0, 0)
                # cv.waitKey(1)
                # cv.destroyWindow('image')
                # cv.waitKey(1)
            else:
                print("Please retry with a better picture.")