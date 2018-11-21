import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
global SHOW_IMAGE 
from scipy.misc import imsave

def rectangle(img, pos, size, color=[1]):
    y, x = pos
    size_y, size_x = size
    print(size_y, size_x)
    print(y, x)
    x_lin = np.linspace(x, x+size_x, num=size_x+1)
    y_lin = np.linspace(y, y+size_y, num=size_y+1)
    xv, yv = np.meshgrid(x_lin, y_lin)
    img[xv.astype(np.uint16),yv.astype(np.uint16)] = color
    return img
def gen_mask(npx, pt = None, blur = 5, rect_size = (20,20), prob = 0.):
    uniform_mask = (np.random.uniform(low=0., high=1., size = (npx,npx)) < prob) * 1.
    mask = np.zeros((npx,npx))
    if pt is None:
        mask = mask + 1.
    else:
        mask = rectangle(mask, pt, rect_size)
        mask = cv2.blur(mask, (blur,blur))
    return mask * uniform_mask
def color_grid_vis(X):
#    if not nhw == (4,4):
#        nh = nw = len(X)**0.5
    nw = nh = int(np.sqrt(X.shape[0]))
    if nh * nw != X.shape[0]:
        nw +=1
        nh +=1
#    (nh, nw) = int(nhw[0]), int(nhw[1])
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = int(n/nw)
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    return img
def plot_compared(img, content_img_orig, save_path = None, show=True):
    combined_imgs = np.zeros((img.shape[0] * 2, img.shape[2], img.shape[2] ,3))
    combined_imgs[::2] = (content_img_orig)/255.
    combined_imgs[1::2] = my_post(img)
    combined_imgs_grid = color_grid_vis(combined_imgs)
    plot_img(combined_imgs_grid, save_path = save_path, show = show)
def plot_img(img, title = None, save_path = None, gray = False, show = True):
    plt.figure()
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    
    if save_path is not None:
        imsave(save_path, img)
    if show:
        plt.show() 
def my_post(img):
    return np.clip(img, 0, 255)/255
def read_imgs(paths, npx = 64):
    imgs = []
    for path in paths:
        img = plt.imread(path)
        if 'celeba' in path.lower():
            img = img[50:50+128,25:25+128,:] 
        imgs.append(cv2.resize(img, dsize=(npx, npx), interpolation=cv2.INTER_AREA).astype('float32').reshape(-1,npx,npx,3))
    return np.concatenate(imgs, axis=0)
def read_img(path, npx = 64):
    img = plt.imread(path)
    
    if 'celeba' in path.lower():
        img = img[50:50+128,25:25+128,:] 
    return cv2.resize(img, dsize=(npx, npx), interpolation=cv2.INTER_AREA).astype('float32')






def preprocess(img, npx = 64):
    # rgb to bgr
    img = img[...,::-1].reshape(-1,npx,npx,3)
    # shape (h, w, d) to (1, h, w, d)
    img -= np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3))
#     img -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    return img


from datetime import datetime
def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_loss(train_loss):
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, color='blue', label='Train loss')
    plt.legend(loc="upper right")
    plt.xlabel('#Iteration')
    plt.ylabel('Loss')
    plt.show()
    print('Final loss: %.4f'%train_loss[-1])

    

from matplotlib import gridspec
def plot_result(content_img_orig, sample_imgs, save_path = 'result.png'):
    plt.figure()
    img_mean = np.mean(sample_imgs, axis=0)
    img_std = np.std(sample_imgs, axis = 0)
    
    orig = content_img_orig/255.
    diff = np.mean((img_mean-content_img_orig)/255. ,axis=2)
    std = np.mean(img_std,axis=2)/255.
    mean = my_post(img_mean)
    grid_img = color_grid_vis(sample_imgs)
    
    npx = content_img_orig.shape[1]
    fig = plt.figure(figsize=(7.75*2,4*2))
    gs = gridspec.GridSpec(2, 4)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.axis('off')
    orig = content_img_orig/255.
    ax1.imshow(content_img_orig/255.)

    ax2 = fig.add_subplot(gs[0,1])
    ax2.axis('off')
    diff = np.mean((img_mean-content_img_orig)/255. ,axis=2)
    ax2.imshow(diff, cmap='gray')

    ax3 = fig.add_subplot(gs[1,0])
    ax3.axis('off')
    mean = my_post(img_mean)
    ax3.imshow(mean)

    ax4 = fig.add_subplot(gs[1,1])
    ax4.axis('off')
    std = np.mean(img_std,axis=2)/255.
    ax4.imshow(std, cmap='gray')

    ax5 = fig.add_subplot(gs[:2,2:])
    ax5.axis('off')
    ax5.imshow(my_post(grid_img))

    gs.update(wspace=0, hspace=0)
    fig.tight_layout()
    fig.savefig(save_path,bbox_inches='tight')

    pass

from os import listdir
from os.path import join, isfile
def listfile(path):
    """
        Input:
            path: 'Dataset directory'
        Output:
            filepaths: All data path including file names.
    """
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    return [join(path, fname) for fname in filenames]