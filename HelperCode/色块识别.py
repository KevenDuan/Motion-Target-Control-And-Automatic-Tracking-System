import cv2
import numpy as np

greenLaser = 'green'
redLaser = 'red'

color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
              }

frame = cv2.imread('/Users/duanhao/Desktop/laser.png')
gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)                     # 高斯模糊
hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)                 # 转化成HSV图像

kernel = np.ones((5, 5), np.uint8)

opening = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel)

cv2.imshow('erode_hsv', opening)

inRange_hsv_green = cv2.inRange(opening, color_dist[greenLaser]['Lower'], color_dist[greenLaser]['Upper'])
inRange_hsv_red = cv2.inRange(opening, color_dist[redLaser]['Lower'], color_dist[redLaser]['Upper'])

cv2.imshow('inrange_hsv_green', inRange_hsv_green)
cv2.imshow('inrange_hsv_red', inRange_hsv_red)

cnts = cv2.findContours(inRange_hsv_green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
c = max(cnts, key=cv2.contourArea)
rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 2)
cv2.imshow('camera', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
