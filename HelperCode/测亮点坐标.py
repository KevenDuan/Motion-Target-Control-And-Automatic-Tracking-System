from skimage import measure
import numpy as np
import cv2

# videoCapture = cv2.VideoCapture(0)
# success = videoCapture.set(3, 640)  # 设置帧宽
# success = videoCapture.set(4, 480)  # 设置帧高
while True:
    # success, frame = videoCapture.read()
    frame = cv2.imread('/Users/duanhao/Desktop/laser/18.jpg')
    if True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#灰度图
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)#高斯滤波
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)#求最亮点坐标 maxLoc
        thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        labels = measure.label(thresh,background=0, return_num = False, connectivity = None)
        mask = np.zeros(thresh.shape, dtype="uint8")
        for label in np.unique(labels):
          if label == 0:
            continue
          labelMask = np.zeros(thresh.shape, dtype="uint8")
          labelMask[labels == label] = 255
          numPixels = cv2.countNonZero(labelMask)
        print(maxLoc)       
        cv2.circle(frame,maxLoc, 1, (255,0,0),10)
        cv2.imshow("thresh", thresh)
        cv2.imshow("gray", gray)
        cv2.imshow("frame", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    else:
        break
# videoCapture.release()
cv2.destroyAllWindows()