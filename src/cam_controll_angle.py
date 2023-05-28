import cv2
import sys
import numpy as np
import math
import RPi.GPIO as GPIO
import time
from matplotlib import pyplot as plt


class Camera:
    def __init__(self):
        self.lower_color = np.array([20, 70, 50])    #yellow
        self.upper_color = np.array([50, 255, 255])  #yellow
        self.kernel = np.ones((5,5),np.uint8)

        self.DEG_THRESH = 2
        angle = 0
        goal_angle = 0
        self.first_loop_flag = True
        self.delay = 1
        self.window_name = 'frame'

        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("comera does not open. quit process")
            sys.exit()
        print("start")
        self.set_cap()

    def read_frame(self):
        ret, frame = self.cap.read()

        return ret, frame
    
    def set_cap(self):
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 100)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)
        self.cap.set(cv2.CAP_PROP_FPS, 100)

    def is_integer(self, txt):
        try:
            int(txt)
            return True
        
        except ValueError:
            return False
        
    def input_goal_angle(self):
        goal_angle = input("input goal angle(0~179): ")
        if not self.is_integer(goal_angle):
            print("Please input int value")
            self.input_goal_angle()
            
        goal_angle = int(goal_angle)
        if 0 > goal_angle or 179 < goal_angle:
            print("Please input within 0 ~ 179")
            self.input_goal_angle()

        return goal_angle
    
    def center_coordinate(self, frame):
        height, width = frame.shape[:2]
        x0 = width // 2
        y0 = height // 2

        return x0, y0

    def masking(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)

        return mask

    def noise_processing(self, mask):
        erosion = cv2.erode(mask,self.kernel,iterations = 1)
        erosion = cv2.erode(erosion,self.kernel,iterations = 1)
        dilation = cv2.dilate(erosion,self.kernel,iterations = 1)
        dilation = cv2.dilate(dilation,self.kernel,iterations = 1)
        mask = cv2.erode(dilation,self.kernel,iterations = 1)

        return mask
        
    def calc_cog(self, contours):
        if len(contours) > 0:
            contour = contours[0]  # 最大の輪郭を使用
            M = cv2.moments(contour)
            if M["m00"]: #sometimes, M[m00] = 0
                x2 = int(M["m10"] / M["m00"])
                y2 = int(M["m01"] / M["m00"])
                return x2, y2

        return None, None

    def calc_angle(self, x0, y0, x1, y1, x2, y2):
        vec1=[x1-x0,y1-y0]
        vec2=[x2-x0,y2-y0]
        absvec1=np.linalg.norm(vec1)
        absvec2=np.linalg.norm(vec2)
        inner=np.inner(vec1,vec2)
        cos_theta=inner/(absvec1*absvec2)
        try:
            theta=math.degrees(math.acos(cos_theta))
            
        except ValueError:
            return None
        
        angle = round(theta,2)
        return angle

    def calc_deviation(self):
        pass

    def is_within_goal_angle(self, goal_angle, angle):
        if (angle >= goal_angle - self.DEG_THRESH) and (angle <= goal_angle + self.DEG_THRESH):
            print("Reached goal")
            return True

        return False


    def clear_capture(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("closed camera")


class MortorControll:
    def __init__(self):
        #TODO: add program what user can choose clockwise or anti-clockwise turn
        self.__IN1 = 17
        self.__PWM = 13
        
        self.set_gpio()

        self.Kp = 0.085
        self.Ki = 0.0015
        self.Kd = 0.5
        
        self.PID_MAX_THRESH = 12
        self.PID_MIN_THRESH = 3
        
    def set_gpio(self):
        GPIO.setmode(GPIO.BCM)

        GPIO.setup(self.__PWM, GPIO.OUT)
        GPIO.setup(self.__IN1, GPIO.OUT)

        self.pwm = GPIO.PWM(self.__PWM, 20)
        GPIO.output(self.__IN1, 1)

    def rotate(self, duty):
        self.pwm.start(duty)

    def calc_pid(self, e, b_e, sum_e, dt):
        p = self.Kp * e
        i = self.Ki * sum_e
        d = self.Kd * (e - b_e) / dt
        
        return max(self.PID_MIN_THRESH, min(self.PID_MAX_THRESH, p + i+ d)) 
    
    def goal_event(self):
        self.pwm.ChangeDutyCycle(0)

    def safty(self, goal_angle, angle, total_time, DEG_THRESH):
        if angle > goal_angle + DEG_THRESH:
            print("Over goal degree")
            return True

        if total_time > 5:
            print("Error: over 5 sec. Couldn't reach the goal.")
            return True

        return False

    def return_pid_gain(self):
        return [self.Kp, self.Ki, self.Kd]
        
    def clean_gpio(self):
        self.pwm.ChangeDutyCycle(0)
        GPIO.cleanup(self.__PWM)
        GPIO.cleanup(self.__IN1)
        print("clean up GPIO")
        
    
class Visualizer:
    def image_show(self, img, x0, y0, x1, y1, x2, y2, window_name="frame"):
        cv2.circle(img, (x0, y0), 10, (255, 255, 255), -1)
        cv2.circle(img, (x1, y1), 10, (255, 255, 255), -1)
        cv2.circle(img, (x2, y2), 5, (0, 0, 0), -1)
        
        cv2.imshow(window_name, img)

    def evaluation_graph(self, goal, i_list, angle_list, pid_list, DEG_THRESH, PID_MAX_THRESH, PID_MIN_THRESH, pid_gain_list, total_time):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        
        ax1.hlines([goal + DEG_THRESH], 0, i_list[-1], "black", linestyle="dashed")
        ax1.hlines([goal - DEG_THRESH], 0, i_list[-1], "black", linestyle="dashed")
        ax1.plot(i_list, angle_list, color="b")
        ax1.set_ylim(0, 200)
        ax1.set_ylabel("degree")
        ax1.set_xlabel("loop count")
        ax1.text(0.99, 0.11, "Final angle " + str(angle_list[-1]), va="bottom", ha="right", transform=ax1.transAxes)
        ax1_txt = "Total time " + str(round(total_time, 4)) + " sec"
        ax1.text(0.99, 0.01, ax1_txt, va="bottom", ha="right", transform=ax1.transAxes)


        ax2.plot(i_list, pid_list, color="r")
        ax2.set_ylim(0, PID_MAX_THRESH + int(PID_MAX_THRESH/2))
        ax2.set_ylabel("PID value")
        ax2.set_xlabel("loop count")
        ax2.hlines([PID_MIN_THRESH], 0, i_list[-1], "black", linestyle="dashed")
        ax2.hlines([PID_MAX_THRESH], 0, i_list[-1], "black", linestyle="dashed")
        ax2.text(0.99, 0.99, "P gain " + str(pid_gain_list[0]), va="top", ha="right", transform=ax2.transAxes)
        ax2.text(0.99, 0.90, "I gain " + str(pid_gain_list[1]), va="top", ha="right", transform=ax2.transAxes)
        ax2.text(0.99, 0.81, "D gain " + str(pid_gain_list[2]), va="top", ha="right", transform=ax2.transAxes)

        plt.tight_layout()
        plt.show()


def main():
    camera = Camera()
    motor = MortorControll()
    visual = Visualizer()
    first_loop_flag = True
    sum_e = 0
    i = 0
    i_list = []
    angle_list = []
    pid_list = []
    pid = 0
    total_time = 0
    t_t = time.time()
    
    try:
        goal_angle = camera.input_goal_angle()
        while True:
            t = time.time()
            
                
            ret, frame = camera.read_frame()
            x0, y0 = camera.center_coordinate(frame)
            if ret:
                mask_ = camera.masking(frame)
                mask = camera.noise_processing(mask_)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x2, y2 = camera.calc_cog(contours)
                if x2 and y2:
                    if first_loop_flag: #defined initial coordinate as x1, y1
                        x1 = x2
                        y1 = y2
                else:
                    continue

                angle = camera.calc_angle(x0, y0, x1, y1, x2, y2)
                if angle == None: # Do not write "if !angle:" to do not "continue" when angle is 0.
                    continue

                if not first_loop_flag:
                    dt = t - b_t
                    e = goal_angle - angle
                    pid = motor.calc_pid(e, b_e, sum_e, dt)
                    #print(angle, pid)
                    motor.rotate(pid)

                if first_loop_flag:
                    first_loop_flag = False
                    e = goal_angle - angle
                    
                b_t = t
                b_e = e
                sum_e += e
                i_list.append(i)
                angle_list.append(angle)
                pid_list.append(pid)
                visual.image_show(mask, x0, y0, x1, y1, x2, y2)
                i += 1
                
                if camera.is_within_goal_angle(goal_angle, angle) or motor.safty(goal_angle, angle, total_time, camera.DEG_THRESH):
                    motor.goal_event()
                    total_time = time.time() - t_t
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break    
                
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
    except KeyboardInterrupt:
        pass

    finally:
        camera.clear_capture()
        motor.clean_gpio()
        visual.evaluation_graph(goal_angle, i_list, angle_list, pid_list, camera.DEG_THRESH, motor.PID_MAX_THRESH, motor.PID_MIN_THRESH, motor.return_pid_gain(), total_time)

        

if __name__ == "__main__":
    main()
    
    
