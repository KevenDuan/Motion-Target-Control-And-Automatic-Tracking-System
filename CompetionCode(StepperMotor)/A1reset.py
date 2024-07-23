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
   

# x坐标max: 570,1个像素0.738mm，对应距离1000mm
zeropos = np.array([100,100]) # 像素
curpos = np.array([500,350])  # 像素
def returnZero():
    # 从当前位置curpos回到原点zeropos
    global curpos
    global zeropos
    global points
    points[0,:] = curpos
    points[1,:] = zeropos
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

    # 高斯滤波
    blurred = cv2.GaussianBlur(crop, (11, 11), 0)
    # cv2.imshow('blurred', blurred)
    # 创建运算核
    kernel = np.ones((1, 1), np.uint8)
    # 开运算
    opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    # 二值化处理
    thresh = cv2.threshold(opening, 230, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('thresh', thresh)

    hsv = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
    # 颜色二值化筛选处理
    inRange_hsv_green = cv2.inRange(hsv, color_dist[greenLaser]['Lower'], color_dist[greenLaser]['Upper'])
    inRange_hsv_red = cv2.inRange(hsv, color_dist[redLaser]['Lower'], color_dist[redLaser]['Upper'])

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
#     print(output)
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
#     cv2.imshow('up', up)
#     cv2.imshow('down', down)
#     cv2.imshow('left', left)
#     cv2.imshow('right', right)
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

if __name__ == "__main__":
    
    downsetup()
    upsetup()
    upstop()
    downstop()
    
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)  #设置窗口的大小

    for i in range(4):
        for j in range(6):
            ret, img = cap.read()
       
        # 需要裁剪的区域###################################
        hStart, hEnd, wStart, wEnd = 200, 640, 320, 760
        ################################################
        cropImg = img[hStart:hEnd, wStart:wEnd]


        # 找激光点 ->
        xc1, yc1, xc2, yc2 = findLaser(cropImg)
        # 对应到原点中的激光坐标
        xc2, yc2 = xc2 + wStart, yc2 + hStart
        print('红色激光点坐标:', xc2, yc2)

        # 图像inRange二值化处理
        hsv = cv2.cvtColor(cropImg, cv2.COLOR_BGR2HSV)
        l_g = np.array([0, 0, 0])  # 阈值下限
        u_g = np.array([255, 255, 110])  # 阈值上限   90-120
        mask = cv2.inRange(hsv, l_g, u_g)
        
#         cv2.imshow('cropImg', cropImg)        
#         cv2.imshow('mask', mask)        
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

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
            print('四个顶点坐标', A, B, C, D)
            print('中心点坐标：', xCenter, yCenter)
        except:
            print('没有裁剪好图片，识别不到黑色方框！')
            print(f'原来图形尺寸为{img.shape[0:1]}')
            xCenter, yCenter = 545, 420


        points = np.array([[xc2, yc2],[xCenter,yCenter]])  
        print('points', points)
        runtwopoints(points)

