import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

layer = 'block5_conv3'
target_size = (224, 224)

def visual_GradCAM(model, layer, img):
    # img_path = img_path
    # img = image.load_img(img_path, target_size=img_size)
    x = img.resize((224, 224))
    x = np.array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get the prediction
    preds = model.predict(x)
    class_idx = np.argmax(preds[0])

    # Get the output feature map of the last convolutional layer
    last_conv_layer = model.get_layer(layer)
    # grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    grad_model = tf.keras.models.Model(model.inputs, [last_conv_layer.output, model.output[0]])

    # Calculate gradients
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(x)
        grads = tape.gradient(preds[:, class_idx], conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each feature map by its importance (gradients) and compute the heatmap
    heatmap = tf.reduce_mean(tf.multiply(conv_output, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Chọn ngưỡng heatmap
    threshold = np.percentile(heatmap, 0)
    # Heat map theo ngưỡng được đặt ra
    masked_heatmap = np.where(heatmap > threshold, heatmap, 0)
    # Plot the heatmap
    masked_heatmap = masked_heatmap[0]

    # Ghép heatmap lên ảnh gốc
    # Đọc và chuyển đổi ảnh gốc sang định dạng RGB
    original_img = np.array(img)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (224, 224))  # Đảm bảo kích thước ảnh giống với kích thước input của mô hình

    # Scale heatmap lên kích thước của ảnh gốc
    heatmap_resized = cv2.resize(masked_heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_rescaled = np.uint8(255 * heatmap_resized)  # Scale heatmap để có thể hiển thị

    # Áp dụng colormap cho heatmap
    heatmap_colored = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)

    # Ghép heatmap lên ảnh gốc
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
    
    return superimposed_img