import scipy.misc
import numpy as np


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def inverse_transform(images):
    return (images+1.)/2.

import re



def smooth(list_to_plot, weight=0.6):  # Weight between 0 and 1
    last = list_to_plot[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in list_to_plot:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)


def txt2array(file):
    file = open(file)
    num_list = []
    name_list = []
    while 1:
        line = file.readline()
        if not line:
            break
        else:
            # nums = re.findall(r"\d+\.?\d*", line)
            nums = re.findall('\[(.*?)\]', line)
            nums = np.cast[np.float32](np.array(nums))
            num_list.append(nums)
            names = re.findall('\](.*?)\[', line)
            names = np.array(names)
            name_list.append(names)
    num_array = np.array(num_list)
    name_array = np.array(name_list)
    return num_array, name_array


def draw_train_val(train_dir, val_dir, save_dir=''):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    num_array, name_array = txt2array(train_dir)
    num_array = num_array[:, 1:]  # remove epoch
    line_color_list = ['b','g','r','c','m','y','k','w']
    fig, ax = plt.subplots(2, 1)

    for i in range(2):
        ax[i].plot(np.arange(len(num_array[:, 0])), smooth((num_array[:, i+5])), line_color_list[0], label='train')

    num_array, name_array = txt2array(val_dir)
    num_array = num_array[:, 1:]  # remove epoch
    for i in range(2):
        ax[i].plot(np.arange(len(num_array[:, 0])), smooth(num_array[:, i+5]), line_color_list[1], label='valid')
        name = name_array[0, i+5]
        name = name[1:-2]  # remove space
        ax[i].set_title(name)
        ax[i].grid()
        ax[i].legend(loc='upper right', shadow=True)

    plt.savefig(save_dir+'/myfig', dpi=500)
    plt.close('all')