import cv2
import math
import numpy as np
import RPi.GPIO as GPIO
import time

# 坐标方向：水平向右为x正向， left-----right，垂直向下为y正方向，向下为up，向上为down
# 像素间隔：0.738mm/每像素

# 规定GPIO引脚
IN1 = 18      # 接PUL-
IN2 = 16      # 接PUL+
IN3 = 15      # 接DIR-
IN4 = 13      # 接DIR+

# 云台上面的电机驱动器，物理引脚
IN12 = 38      # 接PUL-
IN22 = 40      # 接PUL+
IN32 = 36      # 接DIR-
IN42 = 32      # 接DIR+

delay = 0.0001

def downsetup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
    GPIO.setup(IN1, GPIO.OUT)      # Set pin's mode is output
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)

def upsetup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
    GPIO.setup(IN12, GPIO.OUT)      # Set pin's mode is output
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

  

def upward2(steps):
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


def downward2(steps):
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
    
    
def destroy():
    GPIO.cleanup()             # 释放数据
  
curpos = np.array([-1000,1000])  # mm，mm

pi = 3.14159
# 坐标原点在左上角，向左为x轴，向下为y轴。坐标单位像素个数
# x轴移动距离：dx：移动的距离
# 转换：1像素=1.75mm

def xrunpixel(dxpixel):
    # 下面的电机转动多少像素的位置
    xangle = math.atan(dxpixel/1000/1.75)*180/pi   # 单位：度,1000mm需转换为像素个数单位
    if xangle > 0:
        rightward(int(xangle/360*6400))
    else:
        xangle = -xangle
        leftward(int(xangle/360*6400))


# y轴移动距离：dy：移动的距离(像素=mm/1.75)
def yrunpixel(dypixel):
    # 上面的电机转动多少像素的位置
    yangle = math.atan(dypixel/1000/1.75)*180/pi
    if yangle > 0:
        upward(int(yangle/360*6400))
    else:
        yangle = -yangle
        downward(int(yangle/360*6400))

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
        upward2(dystep)
    else:
        dystep = -dystep
        downward2(dystep)        


#points[0,:] 起点位置，points[1,:] 目标位置,# 像素单位
points = np.array([[1,2],[3,4]])  
def runtwopoints(points):
    points = np.array(points)
   
    steps = []
    dstep = 2

    temp = points[1,0]-points[0,0]
    xstep = int(math.atan(temp/732)*1018.591636)  #int((points[1,0]-points[0,0])*0.57142857)
#     xstep = int(math.atan(temp/890)*1018.591636)  #int((points[1,0]-points[0,0])*0.57142857)
    temp = points[1,1]-points[0,1]
    ystep = int(math.atan(temp/740)*1018.591636)
#     ystep = int(math.atan(temp/974)*1018.591636)
    #print('xstep =',xstep,'ystep =',ystep)
    #print()

    if xstep >= 0:
       absxstep = xstep
       xdir = 1
    else:
       absxstep = -xstep
       xdir = 0

    if ystep >= 0:
       absystep = ystep
       ydir = 1
    else:
       absystep = -ystep
       ydir = 0       
       
    # 得到xstep和ystep的最大值
    if absxstep >= absystep:   # 说明较长边在x轴
        maxdir = 'x' 
        absmaxstep = absxstep
        maxstep = xstep
        minstep = ystep
        
    else:
        maxdir = 'y'
        absmaxstep = absystep
        maxstep = ystep
        minstep = xstep


    # 分段运行节数
    sections = int(absmaxstep/dstep)+1
    dstepmin = minstep/sections
    dstepmax = maxstep/sections


    for i in range(1,sections+1):
        if maxdir == 'x': # 沿x边走,先算累计步长
            steps.append([round(i * dstepmax),round(i * dstepmin)])
        else:  # 沿y边走,先算累计步长
            steps.append([round(i * dstepmin),round(i * dstepmax)])

    if  steps[sections-1][0] != xstep:
        steps[sections-1][0] == xstep
    if  steps[sections-1][1] != ystep:
        steps[sections-1][1] == ystep 
    #print('steps=', steps)
    #print()

    for i in range(sections-1,0,-1):
        steps[i][0] -= steps[i-1][0]
        steps[i][1] -= steps[i-1][1]  
    #print('steps=', steps)       
    
    points = len(steps)  # 总点数
    for i in range(points):   
        xrunstep(steps[i][0])
        yrunstep(steps[i][1])
    upstop()
    downstop()
   

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
#         cv2.drawContours(crop, [np.int0(box)], -1, (0, 255, 0), 2)
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
#         cv2.drawContours(crop, [np.int0(box)], -1, (0, 0, 255), 2)
    except:
        print('没有找到红色的激光')
    return cX1, cY1, cX2, cY2

def dengfen(point, k):
    dfpoint = np.zeros((k+1,2))
    dfpoint[0,:] = point[0,:]
    for i in range(k):
        dfpoint[i+1,0] = point[0,0] + int((i+1)*(point[1,0]-point[0,0])/k)
        dfpoint[i+1,1] = point[0,1] + int((i+1)*(point[1,1]-point[0,1])/k)

    return dfpoint

if __name__ == "__main__":
    
    downsetup()
    upsetup()
    upstop()
    downstop()
    
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)  #设置窗口的大小

    while 1:
        for j in range(6):
            ret, img = cap.read()
       
        # 需要裁剪的区域###################################
        hStart, hEnd, wStart, wEnd = 180, 630, 300, 900
        ################################################
        cropImg = img[hStart:hEnd, wStart:wEnd]

        # 找激光点 ->
        xc1, yc1, xc2, yc2 = findLaser(cropImg)
        xc1b, yc1b, xc2b, yc2b = xc1, yc1, xc2, yc2 
        if xc1 != None or xc2 != None:
            # 对应到原点中的激光坐标
            xc1, yc1 = xc1 + wStart, yc1 + hStart
            print('绿色激光点坐标:', xc1, yc1)
            xc2, yc2 = xc2 + wStart, yc2 + hStart
            print('红色激光点坐标:', xc2, yc2)        
        
#         cv2.imshow('cropImg', cropImg)           
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

            points = np.array([[xc1, yc1],[xc2, yc2]])  
            print('points', points)
            runtwopoints(points)



