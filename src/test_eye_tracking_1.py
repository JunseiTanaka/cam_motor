import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
i = 0

def get_cog(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    if len(contours) > 0:
        contour = contours[0]  # 最大の輪郭を使用
        M = cv2.moments(contour)
        if M["m00"]:
            x2 = int(M["m10"] / M["m00"])
            y2 = int(M["m01"] / M["m00"])
            return x2, y2
        else:
            return 0, 0
    else:
        return 0, 0

def trim(frame):
    # カーネルのサイズと形状を定義
    kernel_size = (8, 8)  # カーネルのサイズを指定
    kernel_shape = cv2.MORPH_RECT  # 矩形形状のカーネルを使用

    # カーネルを作成
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    erosion = cv2.erode(frame,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    
    return dilation


lower_black = np.array([0, 0, 0], dtype=np.uint8)
upper_black = np.array([180, 255, 30], dtype=np.uint8)

while i<100:
    _, frame = cap.read()
    
    #グレースケール化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #ランドマーク
    faces = detector(gray)
    if(len(faces)==0):
        print("顔がカメラに移っていないです。")
    else:
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            landmarks = predictor(gray, face)

            # for n in range(0,68):
            #     x = landmarks.part(n).x
            #     y = landmarks.part(n).y
            #     cv2.circle(frame, (x,y), 2, (255,0,0), -1)

        # 瞳のトリミング処理
        # 右目：[36,,37,39, 40]　左目：[42, 43, 45, 46]

        # Right eye
        r_x1,r_y1 = landmarks.part(36).x,landmarks.part(36).y
        r_x2,r_y2 = landmarks.part(37).x,landmarks.part(37).y
        r_x3,r_y3 = landmarks.part(39).x,landmarks.part(39).y
        r_x4,r_y4 = landmarks.part(40).x,landmarks.part(40).y
        # Left eye
        l_x1,l_y1 = landmarks.part(42).x,landmarks.part(42).y
        l_x2,l_y2 = landmarks.part(43).x,landmarks.part(43).y
        l_x3,l_y3 = landmarks.part(45).x,landmarks.part(45).y
        l_x4,l_y4 = landmarks.part(46).x,landmarks.part(46).y

        #　トリミング範囲補正
        trim_val = 10
        r_frame_trim = frame[r_y2-trim_val:r_y4+trim_val, r_x1:r_x3]
        l_frame_trim = frame[l_y2-trim_val:l_y4+trim_val, l_x1:l_x3]

        # 拡大処理（5倍）
        r_height,r_width = r_frame_trim.shape[0],r_frame_trim.shape[1]
        l_height,l_width = l_frame_trim.shape[0],l_frame_trim.shape[1]
        r_frame_trim_resize = cv2.resize(r_frame_trim , (int(r_width*7.0), int(r_height*7.0)))
        l_frame_trim_resize = cv2.resize(l_frame_trim , (int(l_width*7.0), int(l_height*7.0)))

        r_trim = trim(r_frame_trim_resize)
        l_trim = trim(l_frame_trim_resize)
        
        r_hsv = cv2.cvtColor(r_trim, cv2.COLOR_BGR2HSV)
        r_black_mask = cv2.inRange(r_hsv, lower_black, upper_black)
        l_hsv = cv2.cvtColor(l_trim, cv2.COLOR_BGR2HSV)
        l_black_mask = cv2.inRange(l_hsv, lower_black, upper_black)

        """
        # グレースケール処理
        r_frame_gray = cv2.cvtColor(r_frame_trim_resize, cv2.COLOR_BGR2GRAY)
        l_frame_gray = cv2.cvtColor(l_frame_trim_resize, cv2.COLOR_BGR2GRAY)

        #平滑化（ぼかし）
        r_frame_gray = cv2.GaussianBlur(r_frame_gray,(7,7),0)
        l_frame_gray = cv2.GaussianBlur(l_frame_gray,(7,7),0)

        # 2値化処理
        thresh = 0
        maxval = 255
        e_th,r_frame_black_white = cv2.threshold(r_frame_gray,thresh,maxval,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        l_th,l_frame_black_white = cv2.threshold(l_frame_gray,thresh,maxval,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        """

        rx, ry = get_cog(r_black_mask)
        lx, ly = get_cog(l_black_mask)    

        if rx and lx:
            if rx <= 100 and lx <= 100:
                print("right", rx, lx)
            
            elif rx >= 250 and lx >= 250:
                print("left", rx, lx)

        """
        #輪郭の表示
        r_eye_contours, _ = cv2.findContours(r_frame_black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        r_eye_contours = sorted(r_eye_contours, key=lambda x: cv2.contourArea(x), reverse=True) #輪郭が一番大きい順に並べる

        if(len(r_eye_contours)==0):
            print("Right Blink")
        else:
            for cnt in r_eye_contours:
                (rx, ry, rw, rh) = cv2.boundingRect(cnt)
                # cv2.drawContours(r_frame_trim_resize, [cnt], -1, (0,0,255),3) #輪郭の表示
                # cv2.rectangle(r_frame_trim_resize, (x, y), ((x + w, y + h)), (255, 0, 0), 2)#矩形で表示
                cv2.circle(r_frame_trim_resize, (int(rx+rw/2), int(ry+rh/2)), int((rw+rh)/4), (255, 0, 0), 2) #円で表示
                cv2.circle(frame, (int(r_x1+(rx+rw)/10), int(r_y2-3+(ry+rh)/10)), int((rw+rh)/20), (0, 255, 0), 1)    #元画像に表示
                break

        l_eye_contours, _ = cv2.findContours(l_frame_black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        l_eye_contours = sorted(l_eye_contours, key=lambda x: cv2.contourArea(x), reverse=True) #輪郭が一番大きい順に並べる
        if(len(l_eye_contours)==0):
            print("Left Blink")
        else:
            for cnt in l_eye_contours:
                (lx, ly, lw, lh) = cv2.boundingRect(cnt)
                # cv2.drawContours(l_frame_trim_resize, [cnt], -1, (0,0,255),3) #輪郭の表示
                # cv2.rectangle(l_frame_trim_resize, (x, y), ((x + w, y + h)), (255, 0, 0), 2)#矩形で表示
                cv2.circle(l_frame_trim_resize, (int(lx+lw/2), int(ly+lh/2)), int((lw+lh)/4), (255, 0, 0), 2) #円で表示
                cv2.circle(frame, (int(l_x1+(lx+lw)/10), int(l_y2-3+(ly+lh)/10)), int((lw+lh)/20), (0, 255, 0), 1)    #元画像に表示
                break
        """
        #画像の表示    
        cv2.imshow("frame",frame)
        
        cv2.imshow("right eye trim",r_frame_trim_resize)
        cv2.imshow("left eye trim",l_frame_trim_resize)

        cv2.imshow("right eye black white",r_black_mask)
        cv2.imshow("left eye black white",l_black_mask)


        #ウィンドウの配置変更
        cv2.moveWindow('frame', 200,0)
        cv2.moveWindow('right eye trim', 100,100)
        cv2.moveWindow('left eye trim', 500,100)
        cv2.moveWindow('right eye black white', 100,400)
        cv2.moveWindow('left eye black white', 500,400)

    key = cv2.waitKey(1)
    if key ==27:
        break

cv2.destroyAllWindows()