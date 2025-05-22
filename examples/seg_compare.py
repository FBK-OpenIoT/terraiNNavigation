import time
import tensorflow as tf
import contextlib
import datetime
import subprocess
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from pycocotools.coco import COCO
from ultralytics import YOLO
from tqdm import tqdm

from utils import *


yolov11_model_path = 'data/models/YOLOv11_seg/train/weights/best.pt' # Path to YOLOv11 model
yolov8_model_path = 'data/models/YOLOv8_seg/train/weights/best.pt' # Path to YOLOv8 model 
mn_model_path = 'data/models/MNv3_LRASPP/segmentation.tflite' # Path to MobileNetv3+LRASPP model
test_images_dir = 'data/inference/test/' # Path to test images
coco_annotation_file = 'data/inference/test/_annotations.coco.json' # Path to test COCO annotations

# Load models
yolov8_model = YOLO(yolov8_model_path)
yolov8_model.eval()

yolov11_model = YOLO(yolov11_model_path)
yolov11_model.eval()

mn_model = tf.lite.Interpreter(mn_model_path)

# Load COCO annotations
with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stdout(fnull):  # Suppress standard output
        coco = COCO(coco_annotation_file)
image_ids = coco.getImgIds()

# Metrics
accuracy_yolov8, accuracy_yolov11, accuracy_mn = [], [], []
f1_yolov8, f1_yolov11, f1_mn = [], [], []
miou_yolov8, miou_yolov11, miou_mn = [], [], []
inference_times_yolov8, inference_times_yolov11, inference_times_mn = [], [], []
memory_usage_yolov8, memory_usage_yolov11, memory_usage_mn = [], [], []
original_images, masks_true, masks_yolov8, masks_yolov11, masks_mn = [], [], [], [], []

warmup = 0
masksIds = [0, 2, 3, 8]
for img_id in tqdm(image_ids):
    # Get image info
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(test_images_dir, img_info['file_name'])
    if img_id in masksIds:
        img = image.load_img(img_path, target_size=(224, 224))  # Load image
        original_images.append(image.img_to_array(img))

    # Get ground truth mask
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    gt_mask = np.zeros((224, 224), dtype=np.uint8)
    for ann in anns:
        mask = coco.annToMask(ann)
        gt_mask[mask == 1] = ann['category_id']
    if img_id  in masksIds:
        masks_true.append(gt_mask)

    # Run YOLOv8 inference
    start_time = time.time()
    yolov8_pred_mask = yolo_inference(yolov8_model, img_path)
    if img_id >= warmup:
        inference_times_yolov8.append(time.time() - start_time)
        memory_usage_yolov8.append(get_memory_usage())
        if img_id  in masksIds:
            masks_yolov8.append(yolov8_pred_mask.squeeze())

    # Run YOLOv11 inference
    start_time = time.time()
    yolov11_pred_mask = yolo_inference(yolov11_model, img_path)
    if img_id >= warmup:
        inference_times_yolov11.append(time.time() - start_time)
        memory_usage_yolov11.append(get_memory_usage())
        if img_id in masksIds:
            masks_yolov11.append(yolov11_pred_mask.squeeze())


    # Run MobileNet inference
    start_time = time.time()
    #mn_pred_mask = tf_inference(mn_model, img_path)
    mn_pred_mask = tflite_inference(mn_model, img_path)
    if img_id >= warmup:
        inference_times_mn.append(time.time() - start_time)
        memory_usage_mn.append(get_memory_usage())
        if img_id in masksIds:
            masks_mn.append(mn_pred_mask.squeeze())

    # Metrics Calculation
    if img_id >= warmup:
        accuracy_yolov8.append(accuracy_score(gt_mask.flatten(), yolov8_pred_mask.flatten()))
        accuracy_yolov11.append(accuracy_score(gt_mask.flatten(), yolov11_pred_mask.flatten()))
        accuracy_mn.append(accuracy_score(gt_mask.flatten(), mn_pred_mask.flatten()))

        f1_yolov8.append(f1_score(gt_mask.flatten(), yolov8_pred_mask.flatten(), average='weighted'))
        f1_yolov11.append(f1_score(gt_mask.flatten(), yolov11_pred_mask.flatten(), average='weighted'))
        f1_mn.append(f1_score(gt_mask.flatten(), mn_pred_mask.flatten(), average='weighted'))

        miou_yolov8.append(mean_iou(gt_mask, yolov8_pred_mask))
        miou_yolov11.append(mean_iou(gt_mask, yolov11_pred_mask))
        miou_mn.append(mean_iou(gt_mask, mn_pred_mask))

# Summary of metrics
print(f"YOLOv8 - Accuracy: {np.mean(accuracy_yolov8):.4f}, mIoU: {np.mean(miou_yolov8):.4f}, F1: {np.mean(f1_yolov8):.4f}")
print(f"YOLOv11 - Accuracy: {np.mean(accuracy_yolov11):.4f}, mIoU: {np.mean(miou_yolov11):.4f}, F1: {np.mean(f1_yolov11):.4f}")
print(f"MobileNet - Accuracy: {np.mean(accuracy_mn):.4f}, mIoU: {np.mean(miou_mn):.4f}, F1: {np.mean(f1_mn):.4f}")

print(f"YOLOv8 - Average Inference Time: {np.mean(inference_times_yolov8):.4f} seconds")
print(f"YOLOv11 - Average Inference Time: {np.mean(inference_times_yolov11):.4f} seconds")
print(f"MobileNet - Average Inference Time: {np.mean(inference_times_mn):.4f} seconds")

print(f"YOLOv8 - Average Memory Usage: {np.mean(memory_usage_yolov8):.2f} MB")
print(f"YOLOv11 - Average Memory Usage: {np.mean(memory_usage_yolov11):.2f} MB")
print(f"MobileNet - Average Memory Usage: {np.mean(memory_usage_mn):.2f} MB")

# Save the comparison results
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
res_dir = f"results/YOLOvsMN_{current_time}/"
subprocess.run(f'mkdir -p {res_dir}', shell = True)

with open(res_dir + "performances.txt", "w") as file:
    file.write(f"YOLOv8 - Accuracy: {np.mean(accuracy_yolov8):.4f}, mIoU: {np.mean(miou_yolov8):.4f}, F1: {np.mean(f1_yolov8):.4f}\n")
    file.write(f"YOLOv11 - Accuracy: {np.mean(accuracy_yolov11):.4f}, mIoU: {np.mean(miou_yolov11):.4f}, F1: {np.mean(f1_yolov11):.4f}\n")
    file.write(f"MobileNet - Accuracy: {np.mean(accuracy_mn):.4f}, mIoU: {np.mean(miou_mn):.4f}, F1: {np.mean(f1_mn):.4f}\n")

    file.write(f"YOLOv8 - Average Inference Time: {np.mean(inference_times_yolov8):.4f} seconds\n")
    file.write(f"YOLOv11 - Average Inference Time: {np.mean(inference_times_yolov11):.4f} seconds\n")
    file.write(f"MobileNet - Average Inference Time: {np.mean(inference_times_mn):.4f} seconds\n")

    file.write(f"YOLOv8 - Average Memory Usage: {np.mean(memory_usage_yolov8):.2f} MB\n")
    file.write(f"YOLOv11 - Average Memory Usage: {np.mean(memory_usage_yolov11):.2f} MB\n")
    file.write(f"MobileNet - Average Memory Usage: {np.mean(memory_usage_mn):.2f} MB\n")

# Create a figure with multiple subplots (one for each metric)
fig, axs = plt.subplots(2, 2, figsize=(14, 14))

# Plot 1: Accuracy comparison
axs[0, 0].plot(accuracy_mn, label="MNV3+LRASPP Accuracy", color='orange', marker='x', linestyle='-', markersize=4)
axs[0, 0].plot(accuracy_yolov8, label="YOLOv8s Accuracy", color='green', marker='s', linestyle='-', markersize=4)
axs[0, 0].plot(accuracy_yolov11, label="YOLOv11s Accuracy", color='blue', marker='o', linestyle='-', markersize=4)
axs[0, 0].set_ylabel("Accuracy", fontsize=18)
axs[0, 0].legend(fontsize=18)
axs[0, 0].tick_params(axis='x', labelsize=16)
axs[0, 0].tick_params(axis='y', labelsize=16)

# Plot 2: mIoU comparison
axs[0, 1].plot(miou_mn, label="MNV3+LRASPP mIoU", color='orange', marker='x', linestyle='-', markersize=4)
axs[0, 1].plot(miou_yolov8, label="YOLOv8s mIoU", color='green', marker='s', linestyle='-', markersize=4)
axs[0, 1].plot(miou_yolov11, label="YOLOv11s mIoU", color='blue', marker='o', linestyle='-', markersize=4)
axs[0, 1].set_ylabel("mIoU", fontsize=18)
axs[0, 1].legend(fontsize=18)
axs[0, 1].tick_params(axis='x', labelsize=16)
axs[0, 1].tick_params(axis='y', labelsize=16)

# Plot 3: F1 Score comparison
axs[1, 0].plot(f1_mn, label="MNV3+LRASPP F1 Score", color='orange', marker='x', linestyle='-', markersize=4)
axs[1, 0].plot(f1_yolov8, label="YOLOv8s F1 Score", color='green', marker='s', linestyle='-', markersize=4)
axs[1, 0].plot(f1_yolov11, label="YOLOv11s F1 Score", color='blue', marker='o', linestyle='-', markersize=4)
axs[1, 0].set_xlabel("Test Images", fontsize=18)
axs[1, 0].set_ylabel("F1 Score", fontsize=18)
axs[1, 0].legend(fontsize=18)
axs[1, 0].tick_params(axis='x', labelsize=16)
axs[1, 0].tick_params(axis='y', labelsize=16)

# Plot 4: Inference Time comparison
axs[1, 1].plot(inference_times_mn, label="MNV3+LRASPP Inference Time", color='orange', marker='x', linestyle='-', markersize=4)
axs[1, 1].plot(inference_times_yolov8, label="YOLOv8s Inference Time", color='green', marker='s', linestyle='-', markersize=4)
axs[1, 1].plot(inference_times_yolov11, label="YOLOv11s Inference Time", color='blue', marker='o', linestyle='-', markersize=4)
axs[1, 1].set_xlabel("Test Images", fontsize=18)
axs[1, 1].set_ylabel("Inference Time (s)", fontsize=18)
axs[1, 1].legend(fontsize=18)
axs[1, 1].tick_params(axis='x', labelsize=16)
axs[1, 1].tick_params(axis='y', labelsize=16)
axs[1, 1].set_ylim(0.025, 0.05)

# Adjust layout to avoid overlapping
plt.tight_layout()

# Save the plots in the res_dir directory
plt.savefig(res_dir + "metrics.png")

# Plots predictions for further comparison
list2plot = [original_images, masks_yolov8, masks_yolov11, masks_mn]
plotName = ["Original Images", "YOLOv8s", "YOLOv11s", "MNV3+LRASPP"]
for i, img_list in enumerate(list2plot):
    fig, axs = plt.subplots(int(len(img_list)/2), int(len(img_list)/2), figsize=(14, 14))
    fig.suptitle(plotName[i], fontsize=26)
    if i == 0:
        for j, img_in_list in enumerate(img_list):
            if j ==0:
                axs[0, 0].imshow(np.clip(img_in_list, 0, 255).astype(np.uint8))
                axs[0, 0].imshow(masks_true[j], cmap='gray', alpha=0.25)
            if j ==1:
                axs[0, 1].imshow(np.clip(img_in_list, 0, 255).astype(np.uint8))
                axs[0, 1].imshow(masks_true[j], cmap='gray', alpha=0.25)
            if j ==2:
                axs[1, 0].imshow(np.clip(img_in_list, 0, 255).astype(np.uint8))
                axs[1, 0].imshow(masks_true[j], cmap='gray', alpha=0.25)
            if j ==3:
                axs[1, 1].imshow(np.clip(img_in_list, 0, 255).astype(np.uint8))
                axs[1, 1].imshow(masks_true[j], cmap='gray', alpha=0.25)
    else:
        for j, img_in_list in enumerate(img_list):
            if j ==0:
                axs[0, 0].imshow(img_in_list, cmap='gray')
            if j ==1:
                axs[0, 1].imshow(img_in_list, cmap='gray')
            if j ==2:
                axs[1, 0].imshow(img_in_list, cmap='gray')
            if j ==3:
                axs[1, 1].imshow(img_in_list, cmap='gray')

    axs[0, 0].tick_params(axis='x', labelsize=20)
    axs[0, 1].tick_params(axis='x', labelsize=20)
    axs[1, 0].tick_params(axis='x', labelsize=20)
    axs[1, 1].tick_params(axis='x', labelsize=20)
    axs[0, 0].tick_params(axis='y', labelsize=20)
    axs[0, 1].tick_params(axis='y', labelsize=20)
    axs[1, 0].tick_params(axis='y', labelsize=20)
    axs[1, 1].tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    if i == 0:
        plt.savefig(res_dir + "original_images.png")
        plt.savefig(res_dir + "original_images.pdf", format='pdf')
    else:
        plt.savefig(res_dir + plotName[i] + "_masks.png")
        plt.savefig(res_dir + plotName[i] + "_masks.pdf", format='pdf')

# Optionally, display the plots
# plt.show()