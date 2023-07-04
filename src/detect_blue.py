import cv2
import numpy as np

# 青色の範囲を定義
lower_blue = np.array([250, 200, 30])
upper_blue = np.array([255, 255, 90])

# カメラキャプチャを作成
cap = cv2.VideoCapture(0)

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()

    # フレームの取得に失敗した場合は終了する
    if not ret:
        print("Failed to capture frame")
        break


    # 青色の範囲内のピクセルを白、それ以外を黒にするマスクを作成
    mask = cv2.inRange(frame, lower_blue, upper_blue)

    # マスクを元のフレームに適用して、青色の領域のみ表示
    blue_area = cv2.bitwise_and(frame, frame, mask=mask)

    # フレームと青色の領域を表示
    cv2.imshow("Frame", frame)
    cv2.imshow("Blue Area", blue_area)

    # 'q'を押すとループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後片付け
cap.release()
cv2.destroyAllWindows()

