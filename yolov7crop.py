# Обрезаем изображения до содержания
# Оставляем только интересующие классы:
# (2, 4, 6) - DRONE, (5) - Helicopter

import cv2
import os


drone = (2, 4, 6)
helicopter = (5,)


def convert_yolo_to_pixel_cords(
    x_center, y_center, width, height, img_width, img_height
):
    x_center = int(x_center * img_width)
    y_center = int(y_center * img_height)
    box_width = int(width * img_width)
    box_height = int(height * img_height)
    x_min = max(0, int(x_center - (box_width / 2)))
    y_min = max(0, int(y_center - (box_height / 2)))
    x_max = max(img_width, int(x_center + (box_width / 2)))
    y_max = max(img_height, int(y_center + (box_height / 2)))

    return x_min, y_min, x_max, y_max


def crop_image_to_content(image_path, label_path, output_path):
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape

    with open(label_path, "r") as file:
        line = file.readline().strip()
        parts = line.split()
        if len(parts) < 5:
            print(f"Labels error for {image_path}")
            return
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

    if class_id in drone:
        class_id = "drone"
    elif class_id in helicopter:
        class_id = "helicopter"
    else:
        return

    x_min, y_min, x_max, y_max = convert_yolo_to_pixel_cords(
        x_center, y_center, width, height, img_width, img_height
    )

    cropped_image = image[y_min:y_max, x_min:x_max]
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_filename}.jpg"
    output_path = os.path.join(output_path, str(class_id), output_filename)

    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved to {output_path}")


def process_images(image_folder, label_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        label_path = os.path.join(
            label_folder, filename.replace(".jpg", ".txt")
        )
        if os.path.exists(label_path):
            crop_image_to_content(image_path, label_path, output_folder)
        else:
            print(f"No label file found for {filename}")



image_folder = (
    ""
)
label_folder = (
    ""
)
output_folder = (
    ""
)

process_images(image_folder, label_folder, output_folder)
