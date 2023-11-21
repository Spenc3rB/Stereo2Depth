import cv2
import os
import glob

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

num = 0

width = 640
height = 640
# resize the resolution to 320x320
cap.set(3, width)
cap.set(4, height)
cap2.set(3, width)
cap2.set(4, height)

# remove any previous images from both folders
left = glob.glob('images/stereoLeft/*')
right = glob.glob('images/stereoRight/*')
for l in left:
    os.remove(l)
for r in right:
    os.remove(r)

while cap.isOpened():

    succes1, img = cap.read()
    succes2, img2 = cap2.read()


    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img) # left camera
        cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', img2) # right camera
        print("images saved!")
        num += 1
    elif k == ord('q'):
        exit()

    cv2.imshow('Img 1',img)
    cv2.imshow('Img 2',img2)
