import json
import os
import random
import shutil
from collections import defaultdict

# =========================
# НАСТРОЙКИ
# =========================

SOURCE_DIR = "ex/frames"       # здесь лежат JPG + JSON от Labelme
DATASET_DIR = "ex/dataset"     # сюда создадим готовый YOLO-датасет

TRAIN_RATIO = 0.9           # 80% train, 20% val
RANDOM_SEED = 42


# =========================
# ПОДГОТОВКА ПАПОК
# =========================

random.seed(RANDOM_SEED)

train_images_dir = os.path.join(DATASET_DIR, "images", "train")
val_images_dir = os.path.join(DATASET_DIR, "images", "val")

train_labels_dir = os.path.join(DATASET_DIR, "labels", "train")
val_labels_dir = os.path.join(DATASET_DIR, "labels", "val")

for folder in [
    train_images_dir,
    val_images_dir,
    train_labels_dir,
    val_labels_dir
]:
    os.makedirs(folder, exist_ok=True)


# =========================
# ЧИТАЕМ ВСЕ JSON
# =========================

json_files = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith(".json")
]

if not json_files:
    print("JSON-файлы не найдены")
    exit()


# =========================
# СОБИРАЕМ ВСЕ КЛАССЫ
# =========================

all_classes = set()

for filename in json_files:

    json_path = os.path.join(SOURCE_DIR, filename)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for shape in data["shapes"]:
        all_classes.add(shape["label"])


classes = sorted(all_classes)

class_to_id = {
    name: i
    for i, name in enumerate(classes)
}


print("Найдены классы:")

for name, class_id in class_to_id.items():
    print(class_id, "->", name)


# =========================
# АНАЛИЗ ИЗОБРАЖЕНИЙ
# =========================

samples = []

for filename in json_files:

    json_path = os.path.join(SOURCE_DIR, filename)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data["imagePath"]

    image_path = os.path.join(
        SOURCE_DIR,
        image_name
    )

    if not os.path.exists(image_path):
        print("Не найдено изображение:", image_path)
        continue

    # Какие классы присутствуют на этом изображении
    image_classes = set()

    for shape in data["shapes"]:
        image_classes.add(shape["label"])

    samples.append({
        "json": filename,
        "image": image_name,
        "classes": image_classes
    })


# =========================
# ГРУППИРУЕМ ПО КЛАССАМ
# =========================

class_samples = defaultdict(list)

for sample in samples:

    for class_name in sample["classes"]:
        class_samples[class_name].append(sample)


print()
print("Количество изображений по классам:")

for class_name in classes:
    print(
        class_name,
        ":",
        len(class_samples[class_name])
    )


# =========================
# АВТОБАЛАНСИРОВКА TRAIN / VAL
# =========================

train_samples = []
val_samples = []

used = set()


for class_name in classes:

    current_samples = class_samples[class_name].copy()

    random.shuffle(current_samples)

    # Убираем уже распределённые изображения
    current_samples = [
        s for s in current_samples
        if s["image"] not in used
    ]

    count = len(current_samples)

    if count == 0:
        continue

    train_count = int(count * TRAIN_RATIO)

    # Если объектов мало, стараемся хотя бы один
    # положить в validation
    if count > 1 and train_count == count:
        train_count -= 1

    if count > 1 and train_count == 0:
        train_count = 1

    class_train = current_samples[:train_count]
    class_val = current_samples[train_count:]

    for sample in class_train:

        train_samples.append(sample)
        used.add(sample["image"])

    for sample in class_val:

        val_samples.append(sample)
        used.add(sample["image"])


# =========================
# ОСТАВШИЕСЯ ИЗОБРАЖЕНИЯ
# =========================

remaining = [
    s for s in samples
    if s["image"] not in used
]

random.shuffle(remaining)

for sample in remaining:

    current_total = (
        len(train_samples)
        + len(val_samples)
    )

    if current_total == 0:
        train_samples.append(sample)
        continue

    current_ratio = (
        len(train_samples)
        / current_total
    )

    if current_ratio < TRAIN_RATIO:
        train_samples.append(sample)
    else:
        val_samples.append(sample)


# =========================
# ФУНКЦИЯ LABELME -> YOLO
# =========================

def convert_labelme_to_yolo(json_path, output_txt):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    yolo_lines = []

    for shape in data["shapes"]:

        # Для YOLO Detection используем rectangle
        if shape["shape_type"] != "rectangle":
            print(
                "Пропущена фигура:",
                shape["shape_type"],
                "в",
                json_path
            )
            continue

        label = shape["label"]

        class_id = class_to_id[label]

        points = shape["points"]

        x1, y1 = points[0]
        x2, y2 = points[1]

        xmin = min(x1, x2)
        xmax = max(x1, x2)

        ymin = min(y1, y2)
        ymax = max(y1, y2)

        # Центр
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        # Размер
        width = xmax - xmin
        height = ymax - ymin

        # Нормализация 0..1
        x_center /= image_width
        y_center /= image_height

        width /= image_width
        height /= image_height

        yolo_line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        yolo_lines.append(yolo_line)

    with open(
        output_txt,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(yolo_lines))


# =========================
# КОПИРОВАНИЕ ДАННЫХ
# =========================

def process_samples(
    sample_list,
    images_dir,
    labels_dir
):

    for sample in sample_list:

        image_source = os.path.join(
            SOURCE_DIR,
            sample["image"]
        )

        json_source = os.path.join(
            SOURCE_DIR,
            sample["json"]
        )

        image_destination = os.path.join(
            images_dir,
            sample["image"]
        )

        label_name = (
            os.path.splitext(sample["image"])[0]
            + ".txt"
        )

        label_destination = os.path.join(
            labels_dir,
            label_name
        )

        # Копируем изображение
        shutil.copy2(
            image_source,
            image_destination
        )

        # Создаём YOLO-разметку
        convert_labelme_to_yolo(
            json_source,
            label_destination
        )


print()
print("Создание TRAIN...")

process_samples(
    train_samples,
    train_images_dir,
    train_labels_dir
)


print("Создание VAL...")

process_samples(
    val_samples,
    val_images_dir,
    val_labels_dir
)


# =========================
# DATA.YAML
# =========================

yaml_path = os.path.join(
    DATASET_DIR,
    "data.yaml"
)

# Абсолютный путь к папке dataset
dataset_absolute_path = os.path.abspath(DATASET_DIR)

# Для YAML на Windows удобнее использовать /
dataset_absolute_path = dataset_absolute_path.replace("\\", "/")

with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"path: {dataset_absolute_path}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
    )

    for class_id, class_name in enumerate(classes):
        f.write(
            f"  {class_id}: {class_name}\n"
        )

print("Создан:", yaml_path)
# =========================
# CLASSES.TXT
# =========================

classes_path = os.path.join(
    DATASET_DIR,
    "classes.txt"
)

with open(
    classes_path,
    "w",
    encoding="utf-8"
) as f:

    for class_name in classes:
        f.write(class_name + "\n")


# =========================
# СТАТИСТИКА
# =========================

print()
print("==========================")
print("ДАТАСЕТ СОЗДАН")
print("==========================")

print("Всего изображений:", len(samples))
print("Train:", len(train_samples))
print("Val:", len(val_samples))

print()

print(
    "Train:",
    round(
        len(train_samples)
        / len(samples)
        * 100,
        1
    ),
    "%"
)

print(
    "Val:",
    round(
        len(val_samples)
        / len(samples)
        * 100,
        1
    ),
    "%"
)

print()

print("Классы:")

for class_id, class_name in enumerate(classes):
    print(
        class_id,
        "->",
        class_name
    )

print()
print("Готово!")
print("data.yaml:", yaml_path)