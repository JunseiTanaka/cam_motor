import cv2

def get_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        rgb = frame[y, x]
        print("RGB:", rgb)

# カメラキャプチャを作成
cap = cv2.VideoCapture(0)

# カメラキャプチャが正常に開かれているか確認
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# カメラキャプチャからフレームを取得して表示し、マウスイベントを処理する
cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", get_rgb)

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()

    # フレームの取得に失敗した場合は終了する
    if not ret:
        print("Failed to capture frame")
        break

    # フレームを表示
    cv2.imshow("Camera", frame)

    # 'q'を押すとループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後片付け
cap.release()
cv2.destroyAllWindows()


