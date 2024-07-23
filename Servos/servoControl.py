from PCA9685 import PCA9685
import time

pwm = PCA9685(0x40) # init 0x40
pwm.setPWMFreq(50) # init freq

def smooth(road,before,after):
    if before <=after:
        for i in range(before,after+1,1):
            pwm.setServoPulse(road,i)
    else:
        for i in range(before,after+1,-1):
            pwm.setServoPulse(road,i)
            
pwm.setServoPulse(0,500)
