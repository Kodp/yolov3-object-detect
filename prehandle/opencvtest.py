import numpy as np
import cv2
from matplotlib import pyplot as plt
# img = cv2.imread('img/timg (2).jpg',0)
# cv2.imshow('image',img)
# k = cv2.waitKey(0)
# if k==27:
# 		cv2.destroyAllWindows()  #wait for ESC key to exit
# elif k == ord('s'):
# 	cv2.imwrite('46.png',img)  #wait for 's' key to save and exit
np.set_printoptions(threshold=np.inf)
# cv2.destoryAllWindows()
cap = cv2.VideoCapture(0)
while(True):
	#capture frame-by-frame
    ret , frame = cap.read()
    #our operation on the frame come here
    
    #display the resulting frame
    cv2.imshow('frame',frame)
    print(frame)
    input()
    if cv2.waitKey(1) &0xFF ==ord('q'):  #按q键退出
    	break
#when everything done , release the capture
cap.release()
cv2.destroyAllWindows()