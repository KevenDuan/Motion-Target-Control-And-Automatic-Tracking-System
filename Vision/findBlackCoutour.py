import cv2
import numpy as np

#定义形状检测函数
def ShapeDetection(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
    for obj in contours:
        area = cv2.contourArea(obj)  #计算轮廓内区域的面积
        # cv2.drawContours(imgContour, obj, -1, (255, 0, 0), 4)  #绘制轮廓线
        perimeter = cv2.arcLength(obj,True)  #计算轮廓周长
        approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
        CornerNum = len(approx)   #轮廓角点的数量
        x, y, w, h = cv2.boundingRect(approx)  #获取坐标值和宽度、高度
        if CornerNum == 4:
            for i in range(0,4):
                cv2.circle(imgContour, (approx[i][0][0], approx[i][0][1]), 2, (0,0,255), 3)
                axis.append([approx[i][0][0], approx[i][0][1]])

        # cv2.rectangle(imgContour,(x, y),(x+w, y+h),(0,0,255),2)  #绘制边界框

if __name__ == '__main__':
    axis = []

    # 导入图片img
    img = cv2.imread('/Users/duanhao/Desktop/Electronic Design Competion/2023电子设计大赛/black/5.jpg')
    # 需要裁剪的区域###################################
    hStart, hEnd, wStart, wEnd = 250, 650, 320, 700
    ################################################
    cropImg = img[hStart:hEnd, wStart:wEnd]
    cv2.imshow('cropImg', cropImg)

    # 灰度图像处理
    gray = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)#灰度图
    cv2.imshow('gray', gray)

    imgContour = cropImg.copy()

    # 图像inRange二值化处理
    hsv = cv2.cvtColor(cropImg, cv2.COLOR_BGR2HSV)
    l_g = np.array([0, 0, 0])  # 阈值下限
    u_g = np.array([255, 113, 46])  # 阈值上限
    mask = cv2.inRange(hsv, l_g, u_g)
    cv2.imshow('mask', mask)

    ShapeDetection(mask)

    # 从逆时针改为顺时针
    axis[0], axis[2] = axis[2], axis[0]
    print(axis)

    cv2.imshow("crop", imgContour)


    cv2.waitKey(0)
    cv2.destroyAllWindows()