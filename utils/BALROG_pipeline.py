import os
import numpy as np
import skimage as ski
import cv2
import torch
import torchvision
from rembg import remove, new_session
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image


species_array = ['aegypti', 'albopictus', 'annulipes-group', 'caspius',
       'cataphylla', 'cinereus-geminus-pair', 'claviger-petragnani-pair',
       'communis-punctor-pair', 'japonicus', 'koreicus',
       'maculipennis s.l.', 'modestus', 'morsitans-fumipennis-pair',
       'other', 'pipiens s.l.-torrentium-pair', 'richiardii', 'rusticus',
       'stephensi', 'sticticus', 'vexans', 'vishnui-group']

def get_species_list():
    return species_array

def find_files_with_extension(directory, extension):
    file_path_ls = []

    # Walk through the directory tree and collect matching files
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                file_path_ls.append(os.path.join(dirpath, filename))

    # Extract just filenames (not full paths)
    file_name_ls = [os.path.basename(file_path) for file_path in file_path_ls]

    return file_path_ls, file_name_ls


def remove_bg_and_rotate(image, bg_session):
    # Remove the background (transparent background)
    image_rembg = remove(image, bgcolor=(0, 0, 0, -1), session=bg_session)
    mask = np.asarray(image_rembg)[:, :, 3] > 10  # Mask where alpha > 10

    # Apply the mask to the image to retain only foreground
    image_masked = image_rembg * np.dstack([mask] * 4)

    # Get orientation from region properties
    properties = ski.measure.regionprops_table(ski.measure.label(mask), properties=("axis_major_length", "orientation"))
    angle = -(properties["orientation"][np.argmax(properties["axis_major_length"])] * (180 / np.pi) + 90)
    angle = angle + 180 if angle < -90 else angle  # Normalize the angle to a valid range

    # Rotate the image and mask based on the calculated angle
    rotated_image = ski.transform.rotate(image_masked, angle, resize=False, mode='edge', preserve_range=True)
    rotated_mask = ski.transform.rotate(mask, angle, resize=False, mode='edge', preserve_range=True)

    # Remove empty rows and columns (crop the image to the non-empty region)
    rows = np.any(rotated_mask, axis=1)
    cols = np.any(rotated_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop the image and mask to the bounding box of non-empty regions
    image_cropped = rotated_image[rmin:rmax, cmin:cmax]
    mask_cropped = rotated_mask[rmin:rmax, cmin:cmax]

    # Return the cropped image and mask (exclude alpha channel from the image)
    return image_cropped[:, :, :3], mask_cropped


def CLAHE_transform(image, clip_limit=0.5, nbins=32):
    # Convert image to grayscale and apply CLAHE
    equalized_img = ski.exposure.equalize_adapthist(np.mean(image, axis=-1), clip_limit=clip_limit, nbins=nbins)
    equalized_img = ski.filters.median(equalized_img, ski.morphology.disk(1))
    return torch.tensor(equalized_img, dtype=torch.float32)


def pad_and_resize(image):
    # Pad the image to make it square, with the longer dimension being the target size
    height, width = image.shape[:2]
    max_dim = max(height, width)
    pad_height = (max_dim - height) // 2
    pad_width = (max_dim - width) // 2
    image_padded = torch.nn.functional.pad(image, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0)

    # Resize the image to 512x512, then crop to the desired region
    image_resized = torchvision.transforms.functional.resize(image_padded.unsqueeze(0), (384, 384)).numpy()[0]
    return image_resized[96:288, :]


def image_preprocessing_pipeline(file_path, bg_session):
    # Load image
    image = ski.io.imread(file_path)

    # Remove background, rotate the image, and get the mask
    image, mask = remove_bg_and_rotate(image, bg_session)

    # Apply CLAHE transformation
    image = CLAHE_transform(image / 255)

    # Apply the mask to remove background from image
    image[~mask] = 0

    # Pad and resize the image to the specified size
    image = pad_and_resize(image)

    # Transform the image to a PyTorch tensor
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)

def get_model_prediction(model, image):
    # Convert the image to a PyTorch tensor and add batch and channel dimensions
    input = image.unsqueeze(0)

    # Get the model prediction
    with torch.no_grad():
        model.eval()
        prediction = model(input).squeeze().numpy()
    return prediction.squeeze()

def get_model_feaure_map(feature_extractor, image):
    # Convert the image to a PyTorch tensor and add batch and channel dimensions

    # Get the model prediction
    with torch.no_grad():
        tensor = feature_extractor(image.unsqueeze(0)).flatten()
    return tensor

def create_cam(model, input_tensor, class_idx):
  # We have to specify the target we want to generate the CAM for.
  target_layers = [model.features[-1]]
  targets = [ClassifierOutputTarget(class_idx)]

  # Construct the CAM object once, and then re-use it on many images.
  with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    return grayscale_cam
  
def create_guided_cam(model, input_tensor, class_idx):
    target_layers = [model.features[-1]]
    targets = [ClassifierOutputTarget(class_idx)]

    # Construct Grad-CAM++
    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

    # Guided Backpropagation
    gb_model = GuidedBackpropReLUModel(model=model, device='cpu')
    guided_grads = gb_model(input_tensor)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * guided_grads)
    result = deprocess_image(guided_grads)

    return grayscale_cam, result