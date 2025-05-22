from sklearn.metrics import confusion_matrix
import psutil
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# Define the function to calculate memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # in MB

# Function to calculate mean IoU
def mean_iou(y_true, y_pred, num_classes=2):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=np.arange(num_classes))
    iou = cm.diagonal() / (cm.sum(axis=1) + cm.sum(axis=0) - cm.diagonal())
    return np.nanmean(iou)

# Run inference on a YOLO models
def yolo_inference(model, image_path):
    #with torch.no_grad():
    masks = model(image_path, verbose=False)[0].masks
    if masks is not None:
        results = model(image_path, verbose=False)[0].masks.data[0]  # Run inference
        inf_res = results.cpu().numpy().astype(np.float32)
    #results = model(image_path)[0].masks.data[0]  # Run inference
    else:
        results = np.zeros((224, 224), dtype=np.float32)
        inf_res = results

    '''
    # Overlay mask onto image (you can adjust the transparency as needed)
    overlay = np.array(img) * 0.7 + np.expand_dims(results, axis=-1) * 255 * 0.3  # Adjust the transparency
    # Plot the original image and the mask
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img).astype(np.uint8))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay.astype(np.uint8))
    plt.title("Inference Results (Overlay)")
    plt.axis('off')

    # Show plot and wait for user to close it
    plt.show()
    '''

    return inf_res

# Run inference on a TensorFlow model
def tf_inference(model, image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Load image
    img_array = np.expand_dims(image.img_to_array(img), axis=0)  # Convert image to array
    predictions = model.predict(img_array, verbose=0)  # Run inference
    results = predictions[0].reshape(1, 224, 224)
    binary_mask = results.squeeze()
    binary_mask = (binary_mask > 0.5).astype(np.float32)
    results = np.expand_dims(binary_mask, axis=0)
    return results # Return predicted mask

# Run inference on a TensorFlow Lite model
def tflite_inference(interpreter, image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Load image
    img_array = np.expand_dims(image.img_to_array(img), axis=0)  # Convert image to array
    # Allocate tensors (this is necessary before running inference)
    interpreter.allocate_tensors()
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Prepare input data (ensure it matches the model's input size and shape)
    input_data = img_array.astype(np.float32)
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # Run inference
    interpreter.invoke()
    # Get the output tensor
    binary_mask = interpreter.get_tensor(output_details[0]['index']).squeeze()
    binary_mask = (binary_mask > 0.5).astype(np.uint8)

    # Perform distance transform to highlight the object regions
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    # Threshold the distance transform to get sure foreground for the watershed algorithm
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)  # 0.5 can be adjusted
    sure_fg = np.uint8(sure_fg)
    # Threshold dilated distance transform for unknown region
    kernel = np.ones((3,3),np.uint8)
    dilated_dist_transform = cv2.distanceTransform(cv2.dilate(binary_mask,kernel,iterations=5), cv2.DIST_L2, 5)
    _, unknown = cv2.threshold(dilated_dist_transform, 0.01 * dist_transform.max(), 255, 0)  # Adjust 0.3 for background
    unknown = np.uint8(unknown)
    # Create markers: sure background (1), sure foreground (2), unknown region (0)
    markers = np.ones_like(binary_mask, dtype=np.int32)
    markers[unknown == 255] = 0
    markers[sure_fg == 255] = 2
    # Apply watershed algorithm
    mask_rgb = cv2.cvtColor(binary_mask*255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask_rgb, markers)    
    # Mark watershed results
    results = np.zeros_like(binary_mask, dtype=np.float32)
    results[markers == 2] = 1

    '''
    # Overlay mask onto image (you can adjust the transparency as needed)
    overlay = np.array(img) * 0.7 + np.expand_dims(results, axis=-1) * 255 * 0.3  # Adjust the transparency
    # Plot the original image and the mask
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img).astype(np.uint8))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay.astype(np.uint8))
    plt.title("Inference Results (Overlay)")
    plt.axis('off')

    # Show plot and wait for user to close it
    plt.show()
    '''

    return results # Return predicted mask