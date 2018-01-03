import cv2
import numpy as np
# import torch
# import keras
from PIL import Image

pic1 = np.asarray(Image.open('./train_data/091749.png'))
pic2 = np.asarray(Image.open('./train_data/091755.png'))
y, x = pic1.shape[:2]

del_shade = cv2.cvtColor(pic1, cv2.COLOR_RGB2HSV)  # 转成HSV

i = 0
for r in del_shade:
    # 由于颜色是从上到下渐变的,而同一行的背景色不变,因此对每一行求众数,即背景色
    # 通过观察,阴影的明度(V)大概是背景的0.68~0.72倍,通过这个方法去除阴影
    # 这段应该能修一下
    temp = np.argmax(np.bincount(r[:, 2]))
    k = 0
    for j in r[:, 2]:
        if temp * 0.72 >= j >= temp * 0.68:
            del_shade[i][k][2] = temp
        k += 1
    i += 1

cv2.imwrite('1del_shade.png', del_shade)

gray = cv2.cvtColor(del_shade, cv2.COLOR_HSV2RGB)
gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
cv2.imwrite('2gray.png', gray)

ke1 = np.array([[0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, -4, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]])
# 通过这个卷积核,将上下相连的颜色平均,为去除渐变背景做准备
filter2D = cv2.filter2D(gray, -1, ke1)

filter2D = cv2.dilate(filter2D, None, iterations=1)  # 膨胀
_, filter2D = cv2.threshold(filter2D, 10, 255, cv2.THRESH_BINARY)  # 经测定 阈值10就差不多了

cv2.imwrite('3filter2D.png', filter2D)

image, contours, _ = cv2.findContours(filter2D, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找轮廓
print('contours:', len(contours))
'''
# 这一段原来是用来寻找最大面积的最小轮廓矩形

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
    #cv2.drawContours(pic1, [box], -1, (255, 255, 0), 3)
#cv2.drawContours(pic1, bigest, -1, (255, 255, 0), 3)
#pic1 = cv2.putText(pic1, str(bigest_area), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

'''

cv2.drawContours(pic1, contours, -1, (128, 128, 128), 2)  # 这一段突然就只能画出来白色的线了

cv2.imwrite('5drawContours.png', pic1)
