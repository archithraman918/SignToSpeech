import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
capture = cv2.VideoCapture(0)
detect = HandDetector(maxHands=1)

offset = 22
imgSize = 300

folder = "Data/Z" #signs for K and V are switched
#Z is just lit sign
#google teachable machine creates model and trains model on our training data
#Link: https://www.researchgate.net/publication/328396430/figure/fig1/AS:683619848830976@1539999081795/The-26-letters-and-10-digits-of-American-Sign-Language-ASL.jpg
counter = 0

while True:
    success, img = capture.read()
    hands, img = detect.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 #represent rgb
        #means 0-255 8 bit unsigned integer

        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        if (aspectRatio > 1):
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))

            imgResizeShape = imgResize.shape

            wGap = math.ceil((imgSize - wCal) / 2)

            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))

            imgResizeShape = imgResize.shape

            hGap = math.ceil((imgSize - hCal) / 2)

            imgWhite[hGap:hCal+hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
        if (counter == 150):
            break
