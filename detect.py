from ultralytics import YOLO
import cv2


# Загружаем обученную модель
model = YOLO("ex/best.pt")

# 0 — первая веб-камера
cap = cv2.VideoCapture('ex/Pins.mp4')
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Детекция
    results = model(frame, conf=0.1)

    # Рисуем рамки и подписи
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detect", annotated_frame)

    # Выход по q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()