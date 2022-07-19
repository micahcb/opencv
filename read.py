import cv2
import numpy
#import matplotlib.pyplot 

#showImage

path = '/Users/micahblackburn/downloads/dogs.jpeg'

img = cv2.imread(path)

window_name = 'image'

#canny = cv2.Canny(img, 125,175)
#cv2.imshow('canny', canny)

#make all green
#img[:] = 0,255,0

cv2.imshow(window_name,img)


#gradients
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
#laplation
lap = cv2.Laplacian(gray, cv2.CV_64F)
lap = numpy.uint8(numpy.absolute(lap))
cv2.imshow('laplacian', lap)

#sobel
sobelx = cv2.Sobel(gray,cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray,cv2.CV_64F, 0, 1)
combined_sobel = cv2.bitwise_or(sobelx, sobely)

cv2.imshow('sobel x', sobelx)
cv2.imshow('sobel y', sobely)


cv2.imshow('sobel combined', combined_sobel)
canny = cv2.Canny(gray, 150, 175)
cv2.imshow('canny', canny)











#THRESHHOLDING
#makeGray
##gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##cv2.imshow('gray',gray)

#simple
##threshold, thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
##
##cv2.imshow('threshholded simply',thresh)
###inverse thresholding
##threshold, thresh_inv = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)
##
##cv2.imshow('threshholded simply inverted',thresh_inv)
##
###adaptive thresholding
###no need to manually make threshold value, let computer find value itself
##
##adaptive_thresh=cv2.adaptiveThreshold(gray,225,cv2.ADAPTIVE_THRESH_MEAN_C,
##                                      cv2.THRESH_BINARY, 11, 3)
##cv2.imshow('Adaptive', adaptive_thresh)
##
##cv2.waitKey(0)
##
##
##
##




#masked image with histogram computation

#blank = numpy.zeros(img.shape[:2], dtype='uint8')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',gray)

#circle = cv2.circle(blank, (img.shape[1]//2,img.shape[0]//2),100,255,-1)
#mask = cv2.bitwise_and(gray,gray,mask=circle)

#cv2.imshow('Mask', mask)

#gray_hist = cv2.calcHist([gray], [0], mask, [256], [0,256])



#matplotlib.pyplot.figure()
#matplotlib.pyplot.title('Grayscale Histogram')
#matplotlib.pyplot.plot(gray_hist)
#matplotlib.pyplot.xlim([0,256])
#matplotlib.pyplot.show()















#cv2.waitKey(0)

#rotation
#def rotate(img,angle,rotPoint=None):
     #height,width = img.shape[:2]

     #if rotPoint is None:
        # rotPoint = (width//2,height//2)

    # rotMat = cv2.getRotationMatrix2D(rotPoint,angle, 1.0)
    # dimensions = (width,height)

    # return cv2.warpAffine(img, rotMat, dimensions)

#rotated = rotate(img,45)

#cv2.imshow('rotated', rotated)



#translation
#def translate(img,x,y):
    #transMat = numpy.float32([[1,0,x],[0,1,y]])
    #dimensions = (img.shape[1], img.shape[0])
    #return cv2.warpAffine(img,transMat,dimensions)

#translated = translate(img,100,100)
#cv2.imshow('translated',translated)


#showVideo

#capture = cv2.VideoCapture('/Users/micahblackburn/downloads/opencv.mp4')

#while True:
    #isTrue, frame = capture.read()
    #cv2.imshow('video', frame)

   # if cv2.waitKey(20) & 0xFF==ord('d'):
      #  break


#capture.release()
#cv2.destroyAllWindows()


