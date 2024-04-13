# imports
import tensorflow as tf
import numpy as np
import saliency.core as saliency
from tensorflow.keras.models import load_model


model = load_model('VGG16-Plant Disease-90.96.h5')
def PreprocessVGGImage(im):
    im = tf.keras.applications.vgg16.preprocess_input(im)
    return im

def visual_XRAI(model):
    conv_layer = model.get_layer('block5_conv3')
    model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
    
class_idx_str = 'class_idx_str'
def call_model_function(images, call_model_args=None, expected_keys=None):
    target_class_idx =  call_model_args[class_idx_str]
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            output_layer = model(images) # _, output_layer
            output_layer = output_layer[:,target_class_idx]
            gradients = np.array(tape.gradient(output_layer, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv_layer, output_layer = model(images)
            gradients = np.array(tape.gradient(output_layer, conv_layer))
            return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                    saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}
            

def visual_XRAI(model, img):
    original_img = img.resize((224, 224))
    original_img = np.array(original_img)
    im = PreprocessVGGImage(original_img)

    predictions = model(np.array([im])) # _, predictions
    prediction_class = np.argmax(predictions[0])
    call_model_args = {class_idx_str: prediction_class}

    xrai_object = saliency.XRAI()
    # Compute XRAI attributions with default parameters
    xrai_attributions = xrai_object.GetMask(im,
                                            call_model_function,
                                            call_model_args,
                                            batch_size=1)

    # Chọn ra top % các điểm có giá trị lớn nhất trong heatmap
    threshold = np.percentile(xrai_attributions, 70)  # Chọn phân vị 70 (top 30%)

    # Tạo mask chỉ chọn ra các điểm có giá trị lớn hơn ngưỡng
    mask = xrai_attributions > threshold

    # Đặt các điểm không thuộc top 30% trong heatmap thành giá trị 0
    heatmap_top = np.zeros_like(xrai_attributions)
    heatmap_top[mask] = xrai_attributions[mask]

    # Ghép heatmap lên ảnh gốc
    im_mask = np.array(original_img)
    # Đặt tất cả các điểm không thuộc top 30% trong heatmap thành giá trị 0 (hoặc màu đen) trên ảnh im_mask.
    im_mask[~mask] = 0
    
    return im_mask

