import cv2
import numpy as np
# import torch
from PIL import Image
from scipy.stats import mode


pic1 = np.asarray(Image.open('./train_data/091749.png'))
pic2 = np.asarray(Image.open('./train_data/091755.png'))
x = 1080
y = 1920
gray = cv2.cvtColor(pic1, cv2.COLOR_RGB2HSV)
gray[:,:,0] = 0
gray = cv2.cvtColor(gray, cv2.COLOR_HSV2RGB)

cv2.imwrite('0hsv.png', gray)
gray = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)
font=cv2.FONT_HERSHEY_SIMPLEX

cv2.imwrite('1gray.png', gray)
ke1 = np.array([[0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, -4, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]])


filter2D = cv2.filter2D(gray,-1, ke1)

filter2D = cv2.dilate(filter2D, None, iterations=1)
filter2D = np.where(filter2D<=5,filter2D,255)
filter2D = np.where(filter2D==255,filter2D,0)
cv2.imwrite('2filter2D.png', filter2D)
deline = filter2D[:]

i = 0
for r in filter2D:
    temp = np.argmax(np.bincount(r))
    #print([list(r).count(x) for x in list(r)], temp)
    #print(temp)
    k = 0
    for j in r:
        if j == temp:
            deline[i][k] = 0
        k += 1
    i += 1

cv2.imwrite('3deline.png', deline)


image, contours, _ = cv2.findContours(deline[:int(y/1.5),:], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print('contours', len(contours))
bigest_area = 0
bigest = contours[0]
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    area = w * h
    if area>bigest_area:
        bigest_area = area
        ret = cv2.minAreaRect(c)
        box = cv2.boxPoints(ret)
        box = np.int0(box)
        area = w * h
        bigest = [box]
    ret = cv2.minAreaRect(c)
    box = cv2.boxPoints(ret)
    box = np.int0(box)
    cv2.drawContours(pic1, [box], -1, (255, 255, 0), 3)
cv2.drawContours(pic1, bigest, -1, (255, 255, 0), 3)
pic1 = cv2.putText(pic1, str(bigest_area), (x, y), font, 1, (255, 255, 0), 2)
# TODO:寻找阴影,去除阴影
#cv2.drawContours(pic1, contours, -1, (255, 255, 0), 3)

cv2.imwrite('5drawContours.png', pic1)

cv2.waitKey()
cv2.destroyAllWindows()
