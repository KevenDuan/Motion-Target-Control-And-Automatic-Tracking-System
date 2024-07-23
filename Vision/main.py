import cv2
import numpy as np

def findLaser(crop):
    """
    找到图片中点绿色激光点与红色激光点并定位中心
    :param crop: 需要处理点图片
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
    # gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)#灰度图
    # cv2.imshow('gray', gray)

    # 高斯滤波
    blurred = cv2.GaussianBlur(crop, (11, 11), 0)
    # cv2.imshow('blurred', blurred)
    # 创建运算核
    kernel = np.ones((1, 1), np.uint8)
    # 腐蚀
    # erode = cv2.erode(crop, kernel, iterations=1)
    # 膨胀
    # crop_dilate = cv2.dilate(blurred, kernel, iterations = 5)
    # 开运算
    opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    # 二值化处理
    thresh = cv2.threshold(opening, 230, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('thresh', thresh)

    hsv = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
    # 颜色二值化筛选处理
    inRange_hsv_green = cv2.inRange(hsv, color_dist[greenLaser]['Lower'], color_dist[greenLaser]['Upper'])
    inRange_hsv_red = cv2.inRange(hsv, color_dist[redLaser]['Lower'], color_dist[redLaser]['Upper'])
    # cv2.imshow('inrange_hsv_green', inRange_hsv_green)
    # cv2.imshow('inrange_hsv_red', inRange_hsv_red)
    # 找绿色激光点
    try:
        cnts1 = cv2.findContours(inRange_hsv_green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c1 = max(cnts1, key=cv2.contourArea)
        M = cv2.moments(c1)
        cX1 = int(M["m10"] / M["m00"])
        cY1 = int(M["m01"] / M["m00"])
        cv2.circle(crop, (cX1, cY1), 3, (0, 255, 0), -1)
        rect = cv2.minAreaRect(c1)
        box = cv2.boxPoints(rect)
        cv2.drawContours(crop, [np.int0(box)], -1, (0, 255, 0), 2)
    except:
        print('没有找到绿色的激光')

    # 找红色激光点
    try:
        cnts2 = cv2.findContours(inRange_hsv_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c2 = max(cnts2, key=cv2.contourArea)
        M = cv2.moments(c2)
        cX2 = int(M["m10"] / M["m00"])
        cY2 = int(M["m01"] / M["m00"])
        cv2.circle(crop, (cX2, cY2), 3, (0, 0, 255), -1)
        rect = cv2.minAreaRect(c2)
        box = cv2.boxPoints(rect)
        cv2.drawContours(crop, [np.int0(box)], -1, (0, 0, 255), 2)
    except:
        print('没有找到红色的激光')
    return cX1, cY1, cX2 + w1, cY2 + h1

def draw_fit_line(image, data_np, num):
    """
    画出拟合直线
    :param image: 需要拟合直线的图片
    :param data_np: 黑线中心点坐标矩阵 -> data_np必须为矩阵格式
    :param num -> 代表绘制拟合直线的编号（up / down / left / up）
    :return: 直线斜率k，截距b;
    若斜率不存在：返回x, None
    num -> 返回直线的类型
    """
    try:
        if data_np[0, 0] != data_np[1, 0]:  # 斜率存在
            # cv2.fitLine(InputArray  points, distType, param, reps, aeps)
            vline = cv2.fitLine(data_np, cv2.DIST_L2, 0, 0.01, 0.01)  # 直线拟合
            # 算出斜率k以及直线与y轴的交点(0, b)
            k = vline[1] / vline[0]
            b = vline[3] - k * vline[2]
            return k, b, num
        else:  # 斜率不存在
            x = data_np[0, 0]
            return x, None, num
    except:
        print('找不到直线，没有裁剪完整，请重新调整程序！')
        return None, None, num

def findWhite(cnt, num):
    """
    找到二值化后白色的像素点 -> 转入拟合直线函数画出该边线点拟合直线
    :param cnt: 传入的图片
    :param num：代表绘制拟合直线的编号（up / down / left / up）
    :return: 斜率k， 截距b; num -> 返回直线的类型
    """
    axis = []
    for i in range(len(cnt)):
        for j in range(len(cnt[i])):
            if cnt[i][j] == 255: # 判断为白色的像素
                axis.append([j, i])
    arr = np.array(axis)
    k, b, num = draw_fit_line(cropImg, arr, num)
    # print(f'num:{num}, k:{k}, b:{b}')
    return k, b, num


def cropMain(cnt):
    """
    裁剪出屏幕四个边长的图片进行主操作
    :param cnt: 需要裁剪的图片
    :return: (xcentr, ycenter) -> 中心点
    A(x1, y1), B(x2, y2)....
    """
    global w_min, h_min
    h, w = cnt.shape
    w_min, h_min = w // 5, h // 5
    up = cnt[0:h_min, w_min:w_min * 4]
    down = cnt[h_min * 4:, w_min:w_min * 4]
    left = cnt[h_min:h_min * 4,0:w_min]
    right = cnt[h_min:h_min * 4, w_min*4:]
    # 显示出裁剪出来的四个边界线
    cv2.imshow('up', up)
    cv2.imshow('down', down)
    cv2.imshow('left', left)
    cv2.imshow('right', right)
    up_k, up_b, up_num = findWhite(up, 1)

    down_k, down_b, down_num = findWhite(down, 2)

    left_k, left_b, left_num = findWhite(left, 3)

    right_k, right_b, right_num = findWhite(right, 4)

    # 找到四个边界交点
    x1, y1 = getCoordinate(up_k, up_b, left_k, left_b, up_num, left_num)
    x2, y2 = getCoordinate(up_k, up_b, right_k, right_b, up_num, right_num)
    x3, y3 = getCoordinate(down_k, down_b, left_k, left_b, down_num, left_num)
    x4, y4 = getCoordinate(down_k, down_b, right_k, right_b, down_num, right_num)

    # 画边界线
    cv2.line(cropImg, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.line(cropImg, (x1, y1), (x3, y3), (0, 255, 0), 1)
    cv2.line(cropImg, (x3, y3), (x4, y4), (0, 255, 0), 1)
    cv2.line(cropImg, (x2, y2), (x4, y4), (0, 255, 0), 1)
    # 得到中心点坐标
    xCenter, yCenter = getCenter(x1, x2, x3, x4, y1, y2, y3, y4)
    print(xCenter, yCenter)
    return xCenter + w1, \
        yCenter + h1, x1 + w1, x2 + w1, x3 + w1, x4 + w1, y1 + h1, y2 + h1, y3 + h1, y4 + h1

def getCoordinate(k1, b1, k2, b2, num1, num2):
    """
    :param k1: 直线1斜率
    :param b1: 直线1截距
    :param k2: 直线2斜率
    :param b2: 直线2截距
    :param num1, num2: 表示什么类型的直线进行相交
    :return: 交点坐标(x, y)
    """
    try:
        if b1 != None and b2 != None:
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
        elif b1 == None:
            x = k1
            y = k2 * k1 + b2
        elif b2 == None and num2 == 3:
            x = k2
            y = k1 * k2 + b1
        else:
            x = k2
            y = k1 * (k2 + w_min * 4) + b1

        if num1 == 1 and num2 == 3:
            # 坐标化
            x, y = int(x), int(y)
            cv2.circle(cropImg, (x, y), 3, (0, 0, 255), -1)
        elif num1 == 1 and num2 == 4:
            x, y = int(x) + w_min * 4, int(y)
            cv2.circle(cropImg, (x, y), 3, (0, 0, 255), -1)
        elif num1 == 2 and num2 == 3:
            x, y = int(x), int(y) + h_min * 4
            cv2.circle(cropImg, (x, y), 3, (0, 0, 255), -1)
        else:
            x, y = int(x) + w_min * 4, int(y) + h_min * 4
            cv2.circle(cropImg, (x, y), 3, (0, 0, 255), -1)
        return x, y
    except:
        print('不存在交点')
        return -1, -1

def getCenter(x1, x2, x3, x4, y1, y2, y3, y4):
    # 返回中心点坐标并画出
    xCenter = (x1 + x2 + x3 + x4) // 4
    yCenter = (y1 + y2 + y3 + y4) // 4
    # 画出中心点
    cv2.circle(cropImg, (xCenter, yCenter), 4, (0, 0, 255), -1)
    return xCenter, yCenter

def divide(x1, x2, x3, x4, y1, y2, y3, y4, num):
    # 左 -> 上 -> 右 -> 下
    dotAB = (x2 - x1)/100
    dotBD = (y4 - y2)/100
    dotAC = (y3 - y1)/100
    dotCD = (x4 - y4)/100



if __name__ == '__main__':
    img = cv2.imread('/Users/duanhao/Desktop/8.jpg')
    # 裁剪图片 -> 矫正好摄像头水平对准屏幕
    h1, h2, w1, w2 = 50, 650,450, 600
    cropImg = img[:, :]
    print(img.shape)
    xc1, yc1, xc2, yc2 = findLaser(img)
    print(f'绿色激光点坐标点({xc1}, {yc1})')
    print(f'红色激光点坐标点({xc2}, {yc2})')

    # print(cropImg.shape)
    hsv = cv2.cvtColor(cropImg, cv2.COLOR_BGR2HSV)
    l_g = np.array([0, 0, 0])  # 阈值下限
    u_g = np.array([255, 225, 100])  # 阈值上限

    # 小于下限和大于上线的都为黑色, 在阈值中间的为白色
    mask = cv2.inRange(hsv, l_g, u_g)
    #cv2.imshow('mask', mask)

    # 裁剪屏幕照片后进行所有操作
    xCenter, yCenter, x1, x2, x3, x4, y1, y2, y3, y4 = cropMain(mask)
    # 打印出关键点
    print(f'A:({x1}, {y1})')
    print(f'B:({x2}, {y2})')
    print(f'C:({x3}, {y3})')
    print(f'D:({x4}, {y4})')
    print(f'中心点坐标：({xCenter}, {yCenter})')

    # cv2.imshow('cropImg', cropImg)
    cv2.imshow('final', img)
    cv2.waitKey()
    cv2.destroyAllWindows()