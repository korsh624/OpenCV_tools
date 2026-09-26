
import cv2

img = cv2.imread("task2/images/1r0HsD-udt-SJ6-sMBz7E.jpg")

# Переменные для рисования
drawing = False
start_x, start_y = 0, 0
preview = img.copy()


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, preview

    # Начало выделения
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    # Перемещение мыши с зажатой кнопкой
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            preview = img.copy()
            cv2.rectangle(
                preview,
                (start_x, start_y),
                (x, y),
                (0, 255, 0),
                2
            )

    # Завершение выделения
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        # Нормализуем координаты
        x1 = min(start_x, x)
        y1 = min(start_y, y)
        x2 = max(start_x, x)
        y2 = max(start_y, y)

        w = x2 - x1
        h = y2 - y1

        print(f"Начало: ({x1}, {y1})")
        print(f"Конец:  ({x2}, {y2})")
        print(f"Ширина: {w}, высота: {h}")

        print(
            f"cv2.rectangle(img, ({x1}, {y1}), "
            f"({x2}, {y2}), (0, 255, 0), 2)"
        )

        print(f"ROI: img[{y1}:{y2}, {x1}:{x2}]")
        print("-" * 40)

        # Сохраняем прямоугольник на изображении
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        preview = img.copy()


cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

while True:
    cv2.imshow("Image", preview)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if key == ord("c"):  # Очистить прямоугольники
        img = cv2.imread("image.jpg")
        preview = img.copy()

cv2.destroyAllWindows()
