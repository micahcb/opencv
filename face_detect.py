import cv2
import numpy
#import matplotlib.pyplot 

#showImage

path = '/Users/micahblackburn/downloads/woman.jpeg'

img = cv2.imread(path)


cv2.imshow('woman',img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

haar_cascade = cv2.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)


print (len(faces_rect))

for (x,y,w,h) in faces_rect:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness = 2)

cv2.imshow('detectedfaces', img)

## haar cascades a very sensitive to noise
cv2.waitKey(0)