import cv2
import math
import numpy as np
import RPi.GPIO as GPIO
import time

# 规定GPIO引脚
IN1 = 18  # 接PUL-
IN2 = 16  # 接PUL+
IN3 = 15  # 接DIR-
IN4 = 13  # 接DIR+

# 云台上面的电机驱动器，物理引脚
IN12 = 38  # 接PUL-
IN22 = 40  # 接PUL+
IN32 = 36  # 接DIR-
IN42 = 32  # 接DIR+

delay = 0.0001


def downsetup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)  # Numbers GPIOs by physical location
    GPIO.setup(IN1, GPIO.OUT)  # Set pin's mode is output
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)


def upsetup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)  # Numbers GPIOs by physical location
    GPIO.setup(IN12, GPIO.OUT)  # Set pin's mode is output
    GPIO.setup(IN22, GPIO.OUT)
    GPIO.setup(IN32, GPIO.OUT)
    GPIO.setup(IN42, GPIO.OUT)


def downsetStep(w1, w2, w3, w4):
    GPIO.output(IN1, w1)
    GPIO.output(IN2, w2)
    GPIO.output(IN3, w3)
    GPIO.output(IN4, w4)


def upsetStep(w1, w2, w3, w4):
    GPIO.output(IN12, w1)
    GPIO.output(IN22, w2)
    GPIO.output(IN32, w3)
    GPIO.output(IN42, w4)


def downstop():
    # 下面的电机停止
    downsetStep(0, 0, 0, 0)


def upstop():
    # 上面的电机停止
    upsetStep(0, 0, 0, 0)


def rightward(steps):
    # 下面的电机向右转动
    global delay
    for i in range(0, steps):
        downsetStep(1, 0, 1, 0)
        time.sleep(delay)
        downsetStep(0, 1, 1, 0)
        time.sleep(delay)
        downsetStep(0, 1, 0, 1)
        time.sleep(delay)
        downsetStep(1, 0, 0, 1)
        time.sleep(delay)
    downstop()


# def rightward2(steps):
#     global delay
#     for i in range(0, steps):
#         upsetStep(1, 0, 1, 0)
#         time.sleep(delay)
#         upsetStep(0, 1, 1, 0)
#         time.sleep(delay)
#         upsetStep(0, 1, 0, 1)
#         time.sleep(delay)
#         upsetStep(1, 0, 0, 1)
#         time.sleep(delay)
#     upstop()

def leftward(steps):
    # 下面的电机向左转动
    global delay
    for i in range(0, steps):
        downsetStep(1, 0, 0, 1)
        time.sleep(delay)
        downsetStep(0, 1, 0, 1)
        time.sleep(delay)
        downsetStep(0, 1, 1, 0)
        time.sleep(delay)
        downsetStep(1, 0, 1, 0)
        time.sleep(delay)
    downstop()


# def leftward2(steps):
#     global delay
#     for i in range(0, steps):
#         upsetStep(1, 0, 0, 1)
#         time.sleep(delay)
#         upsetStep(0, 1, 0, 1)
#         time.sleep(delay)
#         upsetStep(0, 1, 1, 0)
#         time.sleep(delay)
#         upsetStep(1, 0, 1, 0)
#         time.sleep(delay)
#     upstop()

# def downward(steps):
#     global delay
#     for i in range(0, steps):
#         downsetStep(1, 0, 0, 1)
#         time.sleep(delay)
#         downsetStep(0, 1, 0, 1)
#         time.sleep(delay)
#         downsetStep(0, 1, 1, 0)
#         time.sleep(delay)
#         downsetStep(1, 0, 1, 0)
#         time.sleep(delay)
#     downstop()

def downward2(steps):
    # 上面的电机向下转动
    global delay
    for i in range(0, steps):
        upsetStep(1, 0, 0, 1)
        time.sleep(delay)
        upsetStep(0, 1, 0, 1)
        time.sleep(delay)
        upsetStep(0, 1, 1, 0)
        time.sleep(delay)
        upsetStep(1, 0, 1, 0)
        time.sleep(delay)
    upstop()


# def upward(steps):
#     global delay
#     for i in range(0, steps):
#         downsetStep(1, 0, 1, 0)
#         time.sleep(delay)
#         downsetStep(0, 1, 1, 0)
#         time.sleep(delay)
#         downsetStep(0, 1, 0, 1)
#         time.sleep(delay)
#         downsetStep(1, 0, 0, 1)
#         time.sleep(delay)
#     downstop()

def upward2(steps):
    # 上面的电机向上转动
    global delay
    for i in range(0, steps):
        upsetStep(1, 0, 1, 0)
        time.sleep(delay)
        upsetStep(0, 1, 1, 0)
        time.sleep(delay)
        upsetStep(0, 1, 0, 1)
        time.sleep(delay)
        upsetStep(1, 0, 0, 1)
        time.sleep(delay)
    upstop()


# 测试函数
# angle单位：度  0.036= 360/10000
# def loop(angle):
#     print ("rightward--->leftward:")
#     rightward(int(angle/360*6400))   # 发射脉冲时间间隔0.0001（单位秒）   脉冲个数angle/0.036
#     leftward(int(angle/360*6400))
#     print ("downstop...")
#     time.sleep(1)          # sleep 1s

# def loop2(angle):
#     print ("upward--->downward:")
#     upward2(int(angle/360*6400))   # 发射脉冲时间间隔0.0001（单位秒）   脉冲个数angle/0.036
#     upstop()
#     downward2(int(angle/360*6400))

#     print ("downstop...")
#     upstop()                 # downstop
#     time.sleep(1)          # sleep 3s

def destroy():
    GPIO.cleanup()  # 释放数据


curpos = np.array([-1000, 1000])  # mm，mm

pi = 3.14159


# 坐标原点在左上角，向左为x轴，向下为y轴。坐标单位像素个数
# x轴移动距离：dx：移动的距离
# 转换：1像素=1.75mm

#################测试程序
def xrunpixel(dxpixel):
    # 下面的电机转动多少像素的位置
    xangle = math.atan(dxpixel / 1000 / 1.75) * 180 / pi  # 单位：度,1000mm需转换为像素个数单位
    if xangle > 0:
        rightward(int(xangle / 360 * 6400))
    else:
        xangle = -xangle
        leftward(int(xangle / 360 * 6400))


# y轴移动距离：dy：移动的距离(像素=mm/1.75)
def yrunpixel(dypixel):
    # 上面的电机转动多少像素的位置
    yangle = math.atan(dypixel / 1000 / 1.75) * 180 / pi
    if yangle > 0:
        upward(int(yangle / 360 * 6400))
    else:
        yangle = -yangle
        downward(int(yangle / 360 * 6400))


#####################


# dxstep：x移动的距离（step,int,步进电机步数）
def xrunstep(dxstep):
    # 下面的电机转动的步数
    if dxstep > 0:
        rightward(dxstep)
    else:
        dxstep = -dxstep
        leftward(dxstep)


# dystep：x移动的距离（步长数,int）
def yrunstep(dystep):
    # 上面的电机转动的步数
    if dystep > 0:
        downward2(dystep)
    else:
        dystep = -dystep
        upward2(dystep)

    # linepos = np.array([[1,2],[3,4],[5,6]])  # 像素单位


# def runline(linepos):
#     points = linepos.shape[0]  # 总点数
#     xcumsum = np.cumsum(linepos[:,0])
#     ycumsum = np.cumsum(linepos[:,1])
#     xcumsum = int(math.atan(ycumsum/1374)*1018.591636)     # 1374=1000*1.374,1018.591636 = 1/math.pi*180/360*6400
#     ycumsum = int(math.atan(ycumsum/1374)*1018.591636)

#     for i in range(1,points):
#         xrunstep(xcumsum[i] - xcumsum[i-1])
#         yrunstep(ycumsum[i] - ycumsum[i-1])

# points[0,:] 起点位置，points[1,:] 目标位置,# 像素单位
points = np.array([[1, 2], [3, 4]])


def runtwopoints(points):
    # 从第一个点的位置移动到第二点的位置
    temp = points[1, 0] - points[0, 0]
    xstep = int(math.atan(temp / 738) * 1018.591636)  # int((points[1,0]-points[0,0])*0.57142857)
    temp = points[1, 1] - points[0, 1]
    ystep = int(math.atan(temp / 738) * 1018.591636)
    print('xstep = ', xstep)
    print('ystep = ', ystep)
    xrunstep(xstep)
    yrunstep(ystep)


#########测试程序
# x坐标max: 570,1个像素0.738mm，对应距离1000mm
zeropos = np.array([100, 100])  # 像素
curpos = np.array([500, 350])  # 像素


def returnZero():
    # 从当前位置curpos回到原点zeropos
    global curpos
    global zeropos
    global points
    points[0, :] = curpos
    points[1, :] = zeropos
    runtwopoints(points)


# boundary =  np.array([[300, 300], [295, 295], [290, 290], [285, 285], [280, 280], [275, 275], [270, 271], [265, 266], [260, 261], [255, 256], [250, 251], [245, 246], [240, 242], [235, 237], [230, 232], [225, 227], [220, 222], [215, 218], [210, 213], [205, 208], [200, 203], [195, 198], [190, 193], [185, 189], [180, 184], [175, 179], [170, 174], [165, 169], [160, 165], [155, 160],
#                      [150, 155], [145, 150], [140, 145], [135, 140], [130, 136], [125, 131], [120, 126], [115, 121], [110, 116], [105, 111], [100, 107], [95, 102], [90, 97], [85, 92], [80, 87], [75, 83], [70, 78], [65, 73], [60, 68], [55, 63], [50, 58], [45, 54], [40, 49], [35, 44], [30, 39], [25, 34], [20, 30], [25, 29], [30, 29], [35, 29], [40, 29], [45, 29],
#                      [50, 29], [55, 29], [60, 29], [65, 29], [70, 28], [75, 28], [80, 28], [85, 28], [90, 28], [95, 28], [100, 28], [105, 28], [110, 28], [115, 28], [120, 27], [125, 27], [130, 27], [135, 27], [140, 27], [145, 27], [150, 27], [155, 27], [160, 27], [165, 26], [170, 26], [175, 26], [180, 26], [185, 26], [190, 26], [195, 26], [200, 26], [205, 26], [210, 26],
#                      [215, 25], [220, 25], [225, 25], [230, 25], [235, 25], [240, 25], [245, 25], [250, 25], [255, 25], [260, 25], [265, 24], [270, 24], [275, 24], [280, 24], [285, 24], [290, 24], [295, 24], [300, 24], [305, 24], [310, 23], [315, 23], [320, 23], [325, 23], [330, 23], [335, 23], [340, 23], [345, 23], [350, 23], [355, 23], [360, 22], [365, 22], [370, 22], [375, 22],
#                      [380, 22], [385, 22], [390, 22], [395, 22], [400, 22], [405, 21], [410, 21], [415, 21], [420, 21], [425, 21], [430, 21], [435, 21], [440, 21], [445, 21], [450, 21], [455, 20], [460, 20], [465, 20], [470, 20], [475, 20], [480, 20], [485, 20], [490, 20], [495, 20], [500, 20], [500, 20], [499, 24], [498, 29], [497, 34], [496, 39], [495, 44],
#                      [494, 48], [493, 53], [492, 58], [491, 63], [490, 68], [489, 72], [488, 77], [487, 82], [486, 87], [485, 92], [484, 96], [483, 101], [482, 106], [481, 111], [480, 116], [479, 120], [478, 125], [477, 130], [476, 135], [475, 140], [474, 144], [473, 149], [472, 154], [471, 159], [470, 164], [469, 168], [468, 173], [467, 178], [466, 183], [465, 188], [464, 192],
#                      [463, 197], [462, 202], [461, 207], [460, 212], [459, 216], [458, 221], [457, 226], [456, 231], [455, 236], [454, 240], [453, 245], [452, 250], [451, 255], [450, 260], [449, 264], [448, 269], [447, 274], [446, 279], [445, 284], [444, 288], [443, 293], [442, 298], [441, 303], [440, 308], [439, 312], [438, 317], [437, 322], [436, 327], [435, 332],
#                      [434, 336], [433, 341], [432, 346], [431, 351], [430, 356], [429, 360], [428, 365], [427, 370], [426, 375], [425, 380], [424, 384], [423, 389], [422, 394], [421, 399], [420, 404], [419, 408], [418, 413], [417, 418], [416, 423], [415, 428], [414, 432], [413, 437], [412, 442], [411, 447], [410, 452], [409, 456], [408, 461], [407, 466], [406, 471], [405, 476],
#                      [404, 480], [403, 485], [402, 490], [400, 500], [395, 497], [390, 494], [385, 491], [380, 489], [375, 486], [370, 483], [365, 481], [360, 478], [355, 475], [350, 472], [345, 470], [340, 467], [335, 464], [330, 462], [325, 459], [320, 456], [315, 454], [310, 451], [305, 448], [300, 445], [295, 443], [290, 440], [285, 437], [280, 435], [275, 432],
#                      [270, 429], [265, 427], [260, 424], [255, 421], [250, 418], [245, 416], [240, 413], [235, 410], [230, 408], [225, 405], [220, 402], [215, 400], [210, 397], [205, 394], [200, 391], [195, 389], [190, 386], [185, 383], [180, 381], [175, 378], [170, 375], [165, 372], [160, 370], [155, 367], [150, 364], [145, 362], [140, 359], [135, 356], [130, 354], [125, 351],
#                      [120, 348], [115, 345], [110, 343], [105, 340], [100, 337], [95, 335], [90, 332], [85, 329], [80, 327], [75, 324], [70, 321], [65, 318], [60, 316], [55, 313], [50, 310], [45, 308], [40, 305], [35, 302], [30, 300], [29, 273], [28, 246], [27, 219], [26, 192], [25, 165], [24, 138], [23, 111], [22, 84]])
# def runboundary():
#     global boundary
#     num = boundary.shape[0]
#     #print('num = ',num)
#     for i in range(0,num-1):
#         #print('i = ',i)
#         temp = boundary[i:i+2,:]
#         #print('temp = ',temp)
#         runtwopoints(temp)


def runMotor(x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # 从(x1, y1) -> (x2, y2)
    points = np.array([[x1, y1], [x2, y2]])
    print(points)
    runtwopoints(points)


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
    return cX1, cY1, cX2, cY2

def draw_fit_line(dire, type):
    """
    :param dire: 传入需要拟合的含有白色直线的图像
    :return: 传出直线一般式参数a，b，c
    """
    dic = {'up': (hStart, wStart + leftRight_distance), 'down': (hEnd - upDown_distance, wStart + leftRight_distance),
           'left': (hStart + upDown_distance, wStart), 'right': (hStart + upDown_distance, wEnd - leftRight_distance)}

    # 存放直线的像素点坐标
    axis = []
    for i in range(len(dire)):
        for j in range(len(dire[i])):
            if dire[i][j] == 255: # 判断为白色像素
                # 对应在原图中的高宽坐标
                h = i + dic[type][0]
                w = j + dic[type][1]
                axis.append([w, h])

    data_np = np.array(axis) # 坐标矩阵化
    output = cv2.fitLine(data_np, cv2.DIST_L2, 0, 0.01, 0.01)  # 直线拟合
    k = output[1] / output[0]
    b = output[3] - k * output[2]
    return k, b

def crop(h, w, cnt):
    '''
    将图形裁剪出上下左右四个部分的中间位置
    :param cnt: 需要裁剪的图片
    :return:
    '''
    global upDown_distance, leftRight_distance
    # 上下部分裁剪的宽度
    upDown_distance = h
    # 左右部分裁剪的宽度
    leftRight_distance = w
    up = cnt[0:upDown_distance, leftRight_distance:wEnd - wStart - leftRight_distance]
    down = cnt[hEnd - hStart - upDown_distance:hEnd - hStart, leftRight_distance:wEnd - wStart - leftRight_distance]
    left = cnt[upDown_distance:hEnd - hStart - upDown_distance, 0:leftRight_distance]
    right = cnt[upDown_distance:hEnd - hStart - upDown_distance, wEnd -wStart - leftRight_distance:wEnd - wStart]

    # 显示四个边长裁剪过后的图片
    cv2.imshow('up', up)
    cv2.imshow('down', down)
    cv2.imshow('left', left)
    cv2.imshow('right', right)
    return up, down, left, right

def getCoordinate(k1, b1, k2, b2):
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return int(x), int(y)

def drawCenter(a, b, c, d):
    # 显示边长交点
    cv2.circle(img, (a[0], a[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (b[0], b[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (c[0], c[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (d[0], d[1]), 3, (0, 0, 255), -1)
    # 画边界线
    cv2.line(img, (a[0], a[1]), (b[0], b[1]), (0, 255, 0), 1)
    cv2.line(img, (b[0], b[1]), (c[0], c[1]), (0, 255, 0), 1)
    cv2.line(img, (a[0], a[1]), (d[0], d[1]), (0, 255, 0), 1)
    cv2.line(img, (c[0], c[1]), (d[0], d[1]), (0, 255, 0), 1)

    xCenter = (a[0] + b[0] + c[0] + d[0])//4
    yCenter = (a[1] + b[1] + c[1] + d[1])//4
    cv2.circle(img, (xCenter, yCenter), 4, (0, 0, 255), -1)

    return xCenter, yCenter

def photo():
    index=1
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)  #设置窗口的大小
    # cap.set(15,100)
    flag = cap.isOpened()
    index = 1
    ret, frame = cap.read()
    k = cv2.waitKey(1) & 0xFF
    if ret:  # 按下s键，进入下面的保存图片操作
        cv2.imwrite("/home/pi/Desktop/" + str(index) + ".jpg", frame)

    cap.release() # 释放摄像头
    cv2.destroyAllWindows()# 释放并销毁窗口

if __name__ == "__main__":
    # 初始化步进电机
    downsetup()
    upsetup()

    photo()

    img = cv2.imread("/home/pi/Desktop/1.jpg")  # 只有红色激光

    # 导入图片img
    # img = cv2.imread('/Users/duanhao/Desktop/laser/2.jpg')
    img = cv2.imread("/home/pi/Desktop/1.jpg")  # 只有红色激光

    # 需要裁剪的区域###################################
    hStart, hEnd, wStart, wEnd = 50, 650, 450, 1100
    ################################################
    cropImg = img[hStart:hEnd, wStart:wEnd]
    cv2.imshow('cropImg', cropImg)

    # 找激光点 ->
    xc1, yc1, xc2, yc2 = findLaser(cropImg)
    # 对应到原点中的激光坐标
    try:
        xc1, yc1 = xc1 + wStart, yc1 + hStart
        print('绿色激光点坐标:', xc2, yc2)
        xc2, yc2 = xc2 + wStart, yc2 + hStart
        print('红色激光点坐标:', xc2, yc2)
    except:
        print('无法返回对应原图中激光笔坐标')

    # 图像inRange二值化处理
    hsv = cv2.cvtColor(cropImg, cv2.COLOR_BGR2HSV)
    l_g = np.array([0, 0, 0])  # 阈值下限
    u_g = np.array([255, 255, 116])  # 阈值上限
    mask = cv2.inRange(hsv, l_g, u_g)
    cv2.imshow('mask', mask)

    # 传入二值化后的图片返回裁剪出来的四个边长举行
    # 上下部分裁剪的宽度
    upDown_distance = 80
    # 左右部分裁剪的宽度
    leftRight_distance = 80
    try:
        # 传入参数1：上下部分裁剪宽度; 参数2：左右部分裁剪宽度
        up, down, left, right = crop(upDown_distance, leftRight_distance, mask)
        # 在img原图中拟合四条直线 -> 得到拟合直线的k，b
        up_k, up_b = draw_fit_line(up, 'up')
        down_k, down_b = draw_fit_line(down, 'down')
        left_k, left_b = draw_fit_line(left, 'left')
        right_k, right_b = draw_fit_line(right, 'right')
        # 求出A，B， C， D四个边长交点坐标 -> 顺时针方向定义ABCD点
        A = getCoordinate(up_k, up_b, left_k, left_b)
        B = getCoordinate(up_k, up_b, right_k, right_b)
        C = getCoordinate(down_k, down_b, right_k, right_b)
        D = getCoordinate(down_k, down_b, left_k, left_b)

        # 画出边界线以及中心点
        xCenter, yCenter = drawCenter(A, B, C, D)
        print('中心点坐标：', xCenter, yCenter)
    except:
        print('没有裁剪好图片，识别不到黑色方框！')
        print(f'原来图形尺寸为{img.shape[0:1]}')


    runMotor(xc2, yc2, xc1, yc1)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()