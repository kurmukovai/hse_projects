import numpy as np
import torch
from skimage.transform import rotate

# mean_df = 85.15914365510189
# std_df = 172.05383132719692

def random_crop(sample):
    
    image, mask = sample
    delta_h, delta_w = 20, 20
    h, w = image.shape
    new_h, new_w = 160, 160
    top = np.random.randint(0, h - delta_h - new_h)
    left = np.random.randint(0, w - delta_w - new_w)

    image = image[top: top + new_h,
                  left: left + new_w]
    
    mask = mask[top: top + new_h,
                  left: left + new_w]
    
    return image, mask

def random_rotate(sample):
    
    image, mask = sample
    angles = [0, 90, 180, 270]
    np.random.shuffle(angles)
    angle = angles[0]    
    return rotate(image, angle), rotate(mask, angle)

def normalize(sample):
    image, mask = sample
    if image.max() > 1e-6:
        image /= image.max()
    return image, mask
    
def to_tensor(sample):
    image, mask = sample
    image = image.reshape(1, image.shape[0], image.shape[1])
    label = torch.Tensor([mask.sum(axis=(0,1)) > 0.025 * image.shape[0] * image.shape[1]])
    return torch.from_numpy(image), label