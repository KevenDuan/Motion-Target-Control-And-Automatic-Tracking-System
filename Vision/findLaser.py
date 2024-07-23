import cv2
import numpy as np

def findLaser(img):
    """
    找到图片中点绿色激光点与红色激光点并定位中心
    :param img: 需要处理点图片
    :return: 绿色激光点中心（x1, y1）;红色激光点中心（x2, y2)
    """
    cX1, cY1, cX2, cY2 = None, None, None, None
    greenLaser = 'green'
    redLaser = 'red'
    # 色系下限上限表
    color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
                  'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
                  }
    # 灰度图像处理
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#灰度图
    # cv2.imshow('gray', gray)

    # 高斯滤波
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    # cv2.imshow('blurred', blurred)
    # 创建运算核
    kernel = np.ones((1, 1), np.uint8)
    
    # 开运算
    opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    
    cv2.imshow('oepn', opening)
    # 二值化处理
    thresh = cv2.threshold(opening, 230, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('thresh', thresh)

    hsv = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
    # 颜色二值化筛选处理
    inRange_hsv_green = cv2.inRange(hsv, color_dist[greenLaser]['Lower'], color_dist[greenLaser]['Upper'])
    inRange_hsv_red = cv2.inRange(hsv, color_dist[redLaser]['Lower'], color_dist[redLaser]['Upper'])
    cv2.imshow('inrange_hsv_green', inRange_hsv_green)
    cv2.imshow('inrange_hsv_red', inRange_hsv_red)
    # 找绿色激光点
    try:
        cnts1 = cv2.findContours(inRange_hsv_green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c1 = max(cnts1, key=cv2.contourArea)
        M = cv2.moments(c1)
        cX1 = int(M["m10"] / M["m00"])
        cY1 = int(M["m01"] / M["m00"])
        cv2.circle(img, (cX1, cY1), 3, (0, 255, 0), -1)
        rect = cv2.minAreaRect(c1)
        box = cv2.boxPoints(rect)
        cv2.drawContours(img, [np.int0(box)], -1, (0, 255, 0), 2)
    except:
        print('没有找到绿色的激光')

    # 找红色激光点
    try:
        cnts2 = cv2.findContours(inRange_hsv_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c2 = max(cnts2, key=cv2.contourArea)
        M = cv2.moments(c2)
        cX2 = int(M["m10"] / M["m00"])
        cY2 = int(M["m01"] / M["m00"])
        cv2.circle(img, (cX2, cY2), 3, (0, 0, 255), -1)
        rect = cv2.minAreaRect(c2)
        box = cv2.boxPoints(rect)
        cv2.drawContours(img, [np.int0(box)], -1, (0, 0, 255), 2)
    except:
        print('没有找到红色的激光')
    return cX1, cY1, cX2, cY2


if __name__ == '__main__':
    img = cv2.imread('/Users/duanhao/Documents/运动目标控制与自动追踪系统/TestPhoto/laser/11.jpg')
    x1, y1, x2, y2 = findLaser(img)
    print(f'绿色激光点坐标点({x1}, {y1})')
    print(f'红色激光点坐标点({x2}, {y2})')

    cv2.imshow('camera', img)
    cv2.waitKey()
    cv2.destroyAllWindows()