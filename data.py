import os
import numpy as np
from PIL import Image
from skimage.transform import resize


class DataClass:
  def __init__(self,target_shape,images_path, masks_path):
    self.img_rows =target_shape[0]
    self.img_cols = target_shape[1]
    self.masks_path = masks_path
    self.images_path = images_path

  def load_data(self):
    image_files = os.listdir(self.images_path)
    mask_files = os.listdir(self.masks_path)

    # Sort the file lists to ensure the order is the same
    image_files.sort()
    mask_files.sort()

    images = []
    masks = []

    for image_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(self.images_path, image_file)
        mask_path = os.path.join(self.masks_path, mask_file)

        image, mask = self.reshaping(image_path, mask_path)

        # Ensure all images and masks have the same shape
        if image.shape ==  (self.img_rows,self.img_cols) and mask.shape == (self.img_rows,self.img_cols):

          images.append(image)
          masks.append(mask)

    return np.array(images), np.array(masks)

  def reshaping(self,image_path, mask_path):
    # Load and resize image
    image = Image.open(image_path).convert("L").resize((self.img_rows,self.img_cols), Image.NEAREST)
    # Convert image to numpy array and normalize to [0, 1]
    image= np.array(image)

    # Load and resize mask
    mask = Image.open(mask_path).resize((self.img_rows,self.img_cols), Image.NEAREST)
    # Convert mask to numpy array and normalize to [0, 1]
    mask = np.array(mask) / 255.0

    return image, mask

  def preprocess(self,imgs):
    imgs_p = np.ndarray((imgs.shape[0], self.img_rows, self.img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (self.img_cols, self.img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


"""

# Example usage
def run():
  images_path = "raw_data/all_images"
  masks_path = "raw_data/labels"
  print("asd")
  images, masks = load_data(images_path, masks_path,(125,125))

  print("Images shape:", images.shape)
  print("Masks shape:", masks.shape)"""
