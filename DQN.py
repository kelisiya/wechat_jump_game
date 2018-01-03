import cv2
import numpy as np
from PIL import Image


def save_pic(pic,n):
    pic1 = pic
    del_shade = cv2.cvtColor(pic1, cv2.COLOR_RGB2HSV)  # 转成HSV

    i = 0
    for r in del_shade:
        # 由于颜色是从上到下渐变的,而同一行的背景色不变,因此对每一行求众数,即背景色
        # 通过观察,阴影的明度(V)大概是背景的0.68~0.72倍,通过这个方法去除阴影
        # 这段应该能修一下
        temp = np.argmax(np.bincount(r[:, 2]))
        k = 0
        del_shade[:, :, 2] = np.where((temp * 0.71 < r[:, 2]) & (r[:, 2] < temp * 0.69), r[:, 2], temp)
        '''
        # 只是之前的代码,和上面那一句难道不等价吗?
        for j in r[:, 2]:
            if temp * 0.71 >= j >= temp * 0.69:
                del_shade[i][k][2] = temp
            k += 1
        i += 1
        '''
    gray = cv2.cvtColor(del_shade, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    '''
    # 这是之前用于 将上下相连的颜色平均,为去除渐变背景做准备 的
    ke1 = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, -4, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0]])
    '''
    ke1 = np.array([[0,-1,0],
                    [-1,4,-1],
                    [0,-1,0]])

    filter2D = cv2.filter2D(gray, -1, ke1)
    filter2D = cv2.dilate(filter2D, None, iterations=1)  # 膨胀

    _, filter2D = cv2.threshold(filter2D, 10, 255, cv2.THRESH_BINARY)  # 经测定 阈值10就差不多了
    filter2D = Image.fromarray(filter2D)
    filter2D = filter2D.resize((108, 192), Image.ANTIALIAS)
    filter2D = filter2D.crop((4, 50, 104, 150))
    filter2D.save('D:\\Documents\\GitHub\\LetsJump\\train_data\\%s.png' % n)


if __name__ == 'main':
    save_pic(cv2.imread('0.png'), 0)