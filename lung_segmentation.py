import cv2
import numpy as np

def segment_lungs(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur to remove noise
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # threshold
    _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # invert
    thresh = cv2.bitwise_not(thresh)

    # morphological closing
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=3)

    # find contours
    contours,_ = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(gray.shape,np.uint8)

    # keep largest contours (lungs)
    contours = sorted(contours,key=cv2.contourArea,reverse=True)[:2]

    for cnt in contours:
        cv2.drawContours(mask,[cnt],-1,255,-1)

    segmented = cv2.bitwise_and(img,img,mask=mask)

    return segmented