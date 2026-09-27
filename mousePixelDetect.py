import cv2

# Загружаем изображение
image = cv2.imread("ex/8tecUX-fv6-aIv-wcR6UM.jpg")
# image=cv2.resize(image,(640,480))

# Проверяем, загрузилось ли изображение
if image is None:
    print("Ошибка: изображение не найдено")
    exit()

# Получаем размеры изображения
height, width, channels = image.shape

print("Ширина:", width)
print("Высота:", height)
print("Количество каналов:", channels)


# Функция вызывается при работе мышью
def mouse_callback(event, x, y, flags, param):

    # Если нажата левая кнопка мыши
    if event == cv2.EVENT_LBUTTONDOWN:

        # Получаем цвет пикселя
        b, g, r = image[y, x]

        print()
        print("Координаты:")
        print("X =", x)
        print("Y =", y)

        print("Цвет пикселя:")
        print("B =", b)
        print("G =", g)
        print("R =", r)

        # Создаём копию изображения
        img_copy = image.copy()

        # Рисуем точку
        cv2.circle(
            img_copy,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

        # Выводим координаты
        text = f"({x}, {y})"

        cv2.putText(
            img_copy,
            text,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow("Image", img_copy)


# Создаём окно
cv2.namedWindow("Image")

# Подключаем обработчик мыши
cv2.setMouseCallback("Image", mouse_callback)

# Показываем изображение
cv2.imshow("Image", image)

print()
print("Щёлкните мышкой по изображению.")
print("Для выхода нажмите ESC.")

while True:

    key = cv2.waitKey(1)

    # ESC
    if key == 27:
        break

cv2.destroyAllWindows()