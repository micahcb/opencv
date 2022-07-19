import numpy
import cv2

haar_cascade = cv2.CascadeClassifier('haar_face.xml')

people = ['Ben Affleck', 'Barack Obama', 'Lebron James', 'Jerry Seinfeld']
#features = numpy.load('features.npy')
#labels = numpy.load('labels.npy')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv2.imread('/Users/micahblackburn/downloads/faces/Lebron James/james.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('person', gray)

faces_rect = haar_cascade.detectMultiScale(gray,1.1, 7)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+h]
    label, confidence = face_recognizer.predict(faces_roi)
    print(label)
    print (confidence)
    cv2.putText(img, str(people[label]), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness = 2 )
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness = 2)

cv2.imshow('detected faces', img)

cv2.waitKey(0)


