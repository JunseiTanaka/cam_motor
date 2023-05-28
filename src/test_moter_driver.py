import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
mode = GPIO.getmode()
print(mode)

IN1 = 17
PWM = 13

GPIO.setup(PWM, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
pwm = GPIO.PWM(PWM, 20)

try:
    GPIO.output(IN1, 1)
    while True:
        pwm.start(5)        

except KeyboardInterrupt:
    pwm.ChangeDutyCycle(0)

finally:
    GPIO.cleanup(PWM)
    GPIO.cleanup(IN1)
    print("finished cleary")
