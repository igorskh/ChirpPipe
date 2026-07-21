import cv2
import numpy as np
from PIL import Image


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(tensor, height=None, width=None):
    result = tensor[:, 1:, :]  # remove the CLS token
    if width is None:
        width = height
    if height is None:
        # infer the height and width from the tensor shape
        num_patches = result.shape[1]
        height = width = int(num_patches ** 0.5)
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def read_image_ary(image_path):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    return np.float32(rgb_img) / 255


def make_input_tensor(rgb_img,
                      mean=[0.5, 0.5, 0.5],
                      std=[0.5, 0.5, 0.5]):
    return preprocess_image(rgb_img,
                            mean=mean,
                            std=std)


def ary_to_image(img_ary):
    img = cv2.cvtColor(img_ary, cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
    return Image.fromarray(img)


def generate_gradcam(model, image_path, output_path):
    rgb_img = read_image_ary(image_path)
    input_tensor = make_input_tensor(rgb_img)
    target_layers = [model.model.visual.transformer.resblocks[-1].ln_1]

    with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=None,
                            eigen_smooth=True,
                            aug_smooth=False)
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam)

        image = ary_to_image(cam_image)

        image.save(output_path)
