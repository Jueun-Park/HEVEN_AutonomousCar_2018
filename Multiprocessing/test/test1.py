
import numpy as np
import cv2


black_image = np.zeros((512,512,3), np.uint8 )

cv2.rectangle( black_image, (0,0), (511,511),(255,0,0), 3 )
cv2.circle(black_image,(256,256), 256,(0,0,255),1)
cv2.ellipse(black_image,(256,256),(256,100),0,0,360,(0,255,0),1)

#cv2.imshow( "image", black_image )

a=black_image+black_image[0]
print(a)

cv2.waitKey(0)
