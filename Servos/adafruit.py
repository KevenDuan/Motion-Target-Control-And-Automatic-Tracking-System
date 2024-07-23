import time
from adafruit_servokit import ServoKit

#初始化PCA9685模块
kit = ServoKit(channels=16)

#设置舵机初始位置
kit.servo[0].angle = 90 # 水平方向舵机 
kit.servo[1].angle = 90 # 垂直方向舵机

#定义舵机移动函数
def move_servo(channel, angle):
    kit.servo[channel].angle = angle
    time.sleep(0.1) # 等待舵机稳定

#控制云台运动
move_servo(0, 10) # 水平方向舵机转动至45度
move_servo(1, 50) # 垂直方向舵机转动至60度