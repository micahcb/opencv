import os 
import cv2
import numpy

people = ['Ben Affleck', 'Barack Obama', 'Lebron James', 'Jerry Seinfeld']

DIR = '/Users/micahblackburn/downloads/faces'

haar_cascade = cv2.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
       path = os.path.join(DIR,person)
       label = people.index(person)

       for img in os.listdir(path):
           img_path = os.path.join(path,img)
           img_array = cv2.imread(img_path)
           gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) 

           faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=7)
           for (x,y,w,h) in faces_rect:
               faces_roi = gray[y:y+h, x:x+h]
               features.append(faces_roi)
               labels.append(label)
            

create_train()
print(len(features))
print(len(labels))
print ('done training -------------------')
features = numpy.array(features, dtype='object')
labels = numpy.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')

numpy.save('features.npy', features)
numpy.save('labels.npy', labels)

        