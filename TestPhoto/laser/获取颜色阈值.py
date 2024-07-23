import cv2
import numpy as np


def nothing(x):
    pass
    

cv2.namedWindow("Tracking")

cv2.createTrackbar("LH","Tracking",0,255,nothing)
cv2.createTrackbar("LS","Tracking",0,255,nothing)
cv2.createTrackbar("LV","Tracking",0,255,nothing)
cv2.createTrackbar("UH","Tracking",255,255,nothing)
cv2.createTrackbar("US","Tracking",255,255,nothing)
cv2.createTrackbar("UV","Tracking",100,255,nothing)

#cv2.createTrackbar:绑定滑动条和窗口，定义滚动条的数值
#参数
#第一个参数时滑动条的名字，
#第二个参数是滑动条被放置的窗口的名字，
#第三个参数是滑动条默认值，
#第四个参数时滑动条的最大值，
#第五个参数时回调函数，每次滑动都会调用回调函数。


while True:
    frame = cv2.imread('2.jpg')
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #转换图像格式
    
    l_h = cv2.getTrackbarPos("LH","Tracking")
    l_s = cv2.getTrackbarPos("LS","Tracking")
    l_v = cv2.getTrackbarPos("LV","Tracking")
    
    u_h = cv2.getTrackbarPos("UH","Tracking")
    u_s = cv2.getTrackbarPos("US","Tracking")
    u_v = cv2.getTrackbarPos("UV","Tracking")
    #cv2.getTrackbarPos：得到滑动条的数值
    #参数
    #第一个参数是滑动条名字，
    #第二个时所在窗口，
    #返回值是滑动条的数值。
    
    l_g = np.array([l_h, l_s, l_v]) # 阈值下限
    u_g = np.array([u_h,u_s,u_v])   # 阈值上限
    print(l_g,u_g)

    mask = cv2.inRange(hsv,l_g,u_g) # 二值化
    
    res=cv2.bitwise_and(frame,frame,mask=mask) 

    #cv2.bitwise_and是对二进制数据进行“与”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作，将原图与二值化图像与运算，将阈值内的颜色以原本颜色显示
     
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    #显示窗口
    key = cv2.waitKey(1) 
    if key == 27: 
        break        
cv2.destroyAllWindows()
#延时，当按下ESC时关闭窗口，如果用户没有按下键，则继续等待下一个delay时间(循环)，直到用户按键触发#