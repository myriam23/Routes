from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
import skimage.color as color
import albumentations as albu
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as ndi
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import helper
import numpy as np

"""aug = ['light', 'medium', 'strong', 'grid_shuffle']"""
def transform(aug, original_set, ground_set, convert = False, times = None): 
    
    mylist = []

    if times is None: #if times is none, we augment the whole dataset 
        times = len(original_set)
        mylist = np.arange(0, times)
    else: 
        for i in range(0,times): #else we only augment random images of the dataset 
            x = random.randint(0, len(original_set))
            mylist.append(x)
    

    final = original_set.copy()
    gt_final = ground_set.copy()
    
    for j, idx in enumerate(mylist):
        new, gt = augment(aug, original_set[idx], ground_set[idx], convert)
        final.append(new)
        gt_final.append(gt)
        

    return final, gt_final

#use convert = True when doing light or medium augmentation
def augment(aug, original, gt, convert = False):
    
    if convert: 
        #convert all imgs to uint8 type for light & medium augmentation 
        satellite =  helper.img_float_to_uint8(original)
    else: 
        satellite = original.copy()
    
    augmented = aug(image=satellite, mask=gt)

    return augmented['image'], augmented['mask']

"""Does elastic transformation on x-times images & masks out of set of images and masks"""
def elastic(original_set, ground_set, x = 0): 
    
    mylist = []

    if x == 0: #if x is 0, we augment the whole dataset 
        x = len(original_set)
        mylist = np.arange(0, x)
    else: 
        for i in range(0,x): #else we only augment random images of the dataset 
            x = random.randint(0, len(original_set)-1)
            mylist.append(x)
    
   
    final = original_set.copy()
    gt_final = ground_set.copy()
    
    for j, idx in enumerate(mylist):
        new, gt = elastic_transform(original_set[idx], ground_set[idx])
        final.append(new)
        gt_final.append(gt)
        
    return final, gt_final

"""Called on one image + mask """
#def elastic_transform(image, mask, alpha = 120, sigma = 4, random_state=None):
def elastic_transform(image, mask, alpha = 120, sigma = 4, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    
    indices_img = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    #indices_gt = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    
    
    distored_image = map_coordinates(image, indices_img, order=1, mode='reflect')
    #distored_mask = map_coordinates(mask, indices_gt, order=1, mode='reflect')
    
    newim = distored_image.reshape(image.shape)
    #newgt =  distored_mask.reshape(image.shape)

    #return newim,newgt
    return newim, mask
"""Rotate times images out of set of images and masks"""
def rotation(x_set,y_set, rg = 20, times = None, fill_mode = 'mirror'): 
    
    mylist = []
    if times is None: #if times is none, we augment the whole dataset 
        times = len(x_set)
        mylist = np.arange(0, times)
    else: 
        for i in range(0,times): #else we only augment random images of the dataset 
            x = random.randint(0, len(x_set))
            mylist.append(x)
    
   
    final = x_set.copy()
    gt_final = y_set.copy()
    
    for j, idx in enumerate(mylist):
        new, gt = rotation_transform(x_set[idx], y_set[idx])
        final.append(new)
        gt_final.append(gt)
        
    return final, gt_final

"""Rotation of one image with fill_mode padding (default is mirror) randomly in the range +rg; -rg"""
def rotation_transform(x, y, rg=20, is_random=True, row_index=0, col_index=1, channel_index=2, fill_mode='mirror', cval=0., order=1):
    if is_random:
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
    else:
        theta = np.pi / 180 * rg
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    hy, wy = y.shape[row_index], y.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    transform_matrix_y = transform_matrix_offset_center(rotation_matrix, hy, wy)

    x = affine_transform(x, transform_matrix, False, channel_index, fill_mode, cval, order)
    y = affine_transform(y, transform_matrix_y,True, channel_index, fill_mode, cval, order)
    y = y.reshape(400,400)
    return x,y

def affine_transform(x, transform_matrix, gt = False, channel_index=2, fill_mode='nearest', cval=0., order=1):
    """Return transformed images by given an affine matrix in Scipy format (x is height)"""
    #I modified the original code to fit our needs 
    if gt: 
        x = x.reshape(400,400,1)
        x = np.rollaxis(x, channel_index, 0)
    else: 
        x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    channel_images = [
        ndi.interpolation.affine_transform(
            x_channel, final_affine_matrix, final_offset, order=order, mode=fill_mode, cval=cval
        ) for x_channel in x
    ]

    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x

def transform_matrix_offset_center(matrix, y, x):
    """Convert the matrix from Cartesian coordinates (the origin in the middle of image) to Image coordinates (the origin on the top-left of image)."""
    o_x = (x - 1) / 2.0
    o_y = (y - 1) / 2.0
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def to_RGB(image):
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return RGB_image

def extract_patches(sat, bk, patch_size = 16):
    
    img_patches = [helper.img_crop(sat[i], patch_size, patch_size) for i in range(len(sat))]
    gt_patches = [helper.img_crop(bk[i], patch_size, patch_size) for i in range(len(bk))]
    
    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    gt_patches = helper.patches_labelization(gt_patches)

    return img_patches, gt_patches

def show(original, augmented, mask = None, aug_mask = None, filename = None): 
    
    if mask is None:
        f, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(original)
        ax[0].set_title('Original image')
        
        ax[1].imshow(augmented)
        ax[1].set_title('Augmented image')
    else:
        f, ax = plt.subplots(2, 2, figsize=(16, 16))
        
        ax[0, 0].imshow(original)
        ax[0, 0].set_title('Original image')
        
        ax[0, 1].imshow(augmented)
            
        ax[0, 1].set_title('Augmented image')
        
        ax[1, 0].imshow(mask, interpolation='nearest')
        ax[1, 0].set_title('Original mask')
        ax[1, 1].imshow(aug_mask, interpolation='nearest')
        ax[1, 1].set_title('Augmented mask')

    f.tight_layout()
    if filename is not None:
        f.savefig(filename)
        
def patcher_3000(img, cp_size, p_size):
    list_patches = []
    padding = int((p_size - cp_size) / 2)
    padded_img = np.pad(img, ((padding, padding),(padding, padding), (0,0)), mode='reflect')

    imgwidth = img.shape[0]
    imgheight = img.shape[1]

    for i in range(0, imgheight, cp_size):
        for j in range(0, imgwidth, cp_size):
            im_patch = padded_img[j:j+cp_size+2*padding, i:i+cp_size+2*padding, :]
            list_patches.append(im_patch)
            
    del padded_img
    del im_patch
    return list_patches