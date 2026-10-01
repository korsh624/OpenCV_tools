import cv2
import os


VIDEO_PATH = "res/inputvideo2.mp4"      # исходное видео
OUTPUT_DIR = "frames"         # папка для кадров

SAVE_FPS = 2                # сколько кадров в секунду сохранять


os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Ошибка: не удалось открыть видео")
    exit()

# FPS исходного видео
video_fps = cap.get(cv2.CAP_PROP_FPS)

# Количество кадров
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Длительность видео
duration = total_frames / video_fps

print(f"FPS видео: {video_fps}")
print(f"Всего кадров: {total_frames}")
print(f"Длительность: {duration:.2f} сек.")

# Через сколько исходных кадров сохранять изображение
frame_step = max(1, int(video_fps / SAVE_FPS))

frame_number = 0
saved_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % frame_step == 0:

        filename = os.path.join(
            OUTPUT_DIR,
            f"frame_{saved_number:05d}.jpg"
        )

        cv2.imwrite(filename, frame)

        print(f"Сохранён: {filename}")

        saved_number += 1

    frame_number += 1


cap.release()

print()
print("Готово!")
print(f"Сохранено кадров: {saved_number}")
print(f"Папка: {OUTPUT_DIR}")