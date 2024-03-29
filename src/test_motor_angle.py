import cv2
import sys
import numpy as np
import math

delay = 1
window_name = 'frame'
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'));

if not cap.isOpened():
    sys.exit()

lower_color = np.array([20, 80, 50])
upper_color = np.array([50, 255, 255])
kernel = np.ones((5,5),np.uint8)

try:
    while True:
        ret, frame = cap.read()
        # フレームの幅と高さを取得
        height, width = frame.shape[:2]

        # 画面の中心座標を計算
        x0 = width // 2
        y0 = height // 2

        x1 = x0 + 200
        y1 = y0

        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_color, upper_color)

            erosion = cv2.erode(mask,kernel,iterations = 1)
            erosion = cv2.erode(erosion,kernel,iterations = 1)
            dilation = cv2.dilate(erosion,kernel,iterations = 1)
            dilation = cv2.dilate(dilation,kernel,iterations = 1)
            mask = cv2.erode(dilation,kernel,iterations = 1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                contour = contours[0]  # 最大の輪郭を使用
                M = cv2.moments(contour)
                if M["m00"]:
                    x2 = int(M["m10"] / M["m00"])
                    y2 = int(M["m01"] / M["m00"])
                else:
                    continue
            else:
                continue

            cv2.circle(erosion, (x0, y0), 10, (255, 255, 255), -1)
            cv2.circle(erosion, (x1, y1), 10, (255, 255, 255), -1)
            cv2.circle(erosion, (x2, y2), 5, (0, 0, 0), -1)
            cv2.imshow(window_name, erosion)
            
            vec1=[x1-x0,y1-y0]
            vec2=[x2-x0,y2-y0]
            absvec1=np.linalg.norm(vec1)
            absvec2=np.linalg.norm(vec2)
            inner=np.inner(vec1,vec2)
            cos_theta=inner/(absvec1*absvec2)

            theta=math.degrees(math.acos(cos_theta))
            if y2 <= y0:
                print('deg='+str(round(theta,2)))
            else:
                print('deg='+str(180-round(theta,2)+180))

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyWindow(window_name)