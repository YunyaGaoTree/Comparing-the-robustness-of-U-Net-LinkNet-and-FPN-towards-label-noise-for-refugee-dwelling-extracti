# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Library

# Basics
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from patchify import patchify, unpatchify
from sklearn.model_selection import train_test_split

# Visualization
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Functions for reading, creating folders of data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def create_new_folder(path_data, folder_new):
    
    path_folder_new = os.path.join(path_data, folder_new)
    os.makedirs(path_folder_new, exist_ok=True)
    
    return path_folder_new


def define_path_of_output_patch(EXP, path_out_npy, data_type):
    
    # data_type: img_tr, lab_va, img_tr_pred1, img_va_pred1...
    out_patch_name = EXP + "_patch_" + data_type + ".npy"
    
    path_out_patch = os.path.join(path_out_npy, out_patch_name)
    
    return path_out_patch


def find_tif(path_tif):
    
    all_file_list = os.listdir(path_tif)
    tif_file_list = []
    
    for i in all_file_list:
        if i[-4:] == '.tif':
            tif_file_list.append(i)
        else:
            continue

    tif_file_list.sort()
    
    return tif_file_list


def get_tif_list(path_tif):
    
    tif_file_list_ = find_tif(path_tif)

    tif_file_list = [os.path.join(path_tif, i) for i in tif_file_list_]
    tif_file_list.sort()

    return tif_file_list

def create_pathlist_of_cropped_tif(path_tif_ori, path_tif_crop):
    
    path_list_ori = find_tif(path_tif_ori)
    
    list_path_crop = []

    num_tif = len(path_list_ori)

    for i in range(num_tif):
        
        tif_name = path_list_ori[i]
        tif_crop_name = tif_name[:-4] + "_crop" + str(i+1) + ".tif"
        path_crop = os.path.join(path_tif_crop, tif_crop_name)
        list_path_crop.append(path_crop)

    return list_path_crop

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Functions related to processing tif
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def convert_tif_list_into_array_list(tif_path_list, ignore_value):
    
    tif_arr_list = []

    for tif_path in tif_path_list:

        with rasterio.open(tif_path, 'r') as ds:
            arr = ds.read()  # read all raster values
            arr[arr==ignore_value] = 0

        tif_arr_list.append(arr)

    return tif_arr_list


def show_array_list_info(arr_list, arr_type):
    
    num_tile = len(arr_list)
    print("The total amount of {} is {}".format(arr_type, num_tile))

    for i in range(num_tile):
        print("Tile {}: {} Maximum value: {}".format(i+1, arr_list[i].shape, arr_list[i].max()))
        
        
def create_cropped_tif_and_get_cropped_arr_list(path_list_arr, path_list_tif_crop, patchsize, ignore_value):
    # path_list_arr: the var includes arrays of original tiff
    # path_list_tif_crop: the var means a list of output cropped tiff paths

    # the var contains a list of cropped arrays from original arrays
    path_list_arr_crop = []

    num_arr = len(path_list_arr)

    for i in range(num_arr):
        path_arr = path_list_arr[i]

        with rasterio.open(path_arr) as src:

            # The size in pixels of your desired window
            xsize, ysize = src.shape[0], src.shape[1]

            xsize_new = int(xsize/patchsize) * patchsize
            ysize_new = int(ysize/patchsize) * patchsize

            # Generate a window
            xoff, yoff = 0, 0

            # Create a Window and calculate the transform from the source dataset    
            window = Window(xoff, yoff, ysize_new, xsize_new)
            transform = src.window_transform(window)

            # Create a new cropped raster to write to
            profile = src.profile
            profile.update({
                'height': xsize_new,
                'width': ysize_new,
                'transform': transform})

            # get cropped tif
            src_new = src.read(window=window)
            src_new[src_new == ignore_value] = 0

            # save the cropped tif in the list for future use
            path_list_arr_crop.append(src_new)

            # define the name of cropped tif
            path_tif_crop = path_list_tif_crop[i]

            with rasterio.open(path_tif_crop, 'w', **profile) as dst:
                # Read the data from the window and write it to the output raster
                dst.write(src_new)

        print(path_tif_crop, " saved.")

    return path_list_arr_crop


# check whether the shape of labels is the same as that of images
def check_shape_of_cropped_image_and_label(arr_tr_img_crop_list, arr_ts_img_crop_list, arr_tr_lab_crop_list, arr_ts_lab_crop_list):
    
    num_arr = len(arr_tr_img_crop_list)
    correct = 0
    
    for i in range(num_arr):
        
        if arr_tr_img_crop_list[i].shape[1:] == arr_tr_lab_crop_list[i].shape[1:]:
            print("Tile {} Shape of Image and Label: Correct".format(i+1))
            correct += 1
        else:
            print("Tile {} Shape of Image and Label: Error".format(i+1))
            print("Shape of image: {}".format(arr_tr_img_crop_list[i].shape))
            print("Shape of label: {}".format(arr_tr_lab_crop_list[i].shape))
            
    if correct == num_arr:
        print("The shapes of all image and label arrays are the same.")
        

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Functions related to processing numpy arrays
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
               
def swap_array_axis(arr):
    
    arr_swap = np.swapaxes(arr, 0, 1)  # exp. (2403, 4, 4848)
    arr_swap = np.swapaxes(arr_swap, 1, 2) # exp. (2403, 4848, 4)
    
    return arr_swap

def swap_array_axis_for_visualization(arr):
    
    arr_swap = np.swapaxes(arr, 2, 0)  # exp. (2403, 4, 4848)
    arr_swap = np.swapaxes(arr_swap, 2, 1) # exp. (2403, 4848, 4)
    
    return arr_swap

def from_img_4bands_to_3bands(img, band_selection):
    
    # band_selection: a list like: ["R", "G", "B"]
    
    img_r = img[:,:,:,2:3]
    img_g = img[:,:,:,1:2]
    img_b = img[:,:,:,0:1]
    img_nir = img[:,:,:,3:]
    
    img_bands = {"R": img_r,
                 "G": img_g,
                 "B": img_b,
                 "NIR": img_nir
                }
    
    img_1 = img_bands[band_selection[0]]
    img_2 = img_bands[band_selection[1]]
    img_3 = img_bands[band_selection[2]]
    
    img_3bands = np.concatenate((img_1, img_2, img_3), axis=3)
    
    return img_3bands

def select_image_bands(tr_img, tr_lab, va_img, va_lab, nbands, band_selection):

    if nbands == 3:

        X_tr = from_img_4bands_to_3bands(tr_img, band_selection)
        X_va = from_img_4bands_to_3bands(va_img, band_selection)

    else:

        X_tr = tr_img
        X_va = va_img

    Y_tr = tr_lab
    Y_va = va_lab

    print("X_tr.shape: {}, X_va.shape:{}".format(X_tr.shape, X_va.shape))
    
    return X_tr, X_va, Y_tr, Y_va


def convert_single_tile_to_patch(tile_array, patchsize, nchannel, step):
    
    target_shape = (patchsize, patchsize, nchannel)
    
    patches = patchify(tile_array, target_shape, step=step)
    num_patches = patches.shape[0]*patches.shape[1]

    patch_final = patches.reshape((num_patches, patchsize, patchsize, nchannel))
    
    return patch_final

def convert_list_tile_to_patch(list_tile_arr, patchsize, nchannel, step):
    
    num_tile = len(list_tile_arr)
    
    for i in range(num_tile):
        
        arr = list_tile_arr[i]
        arr_swap = swap_array_axis(arr)
        patch_single_tile = convert_single_tile_to_patch(arr_swap, patchsize, nchannel, step)
        
        if i == 0:
            patch_final = patch_single_tile
        
        else:
            patch_final = np.concatenate((patch_final, patch_single_tile), axis=0)
            
    return patch_final
            

def one_hot_encoding(arr_stacked_lab, nclass):
    
    label_beforeOH = arr_stacked_lab.copy()
    temp = 5

    for i in range(nclass):

        label_beforeOH = arr_stacked_lab.copy()

        if i == 0:
            label_beforeOH[label_beforeOH != i] = temp # a temporary value to store non-zero values
            label_beforeOH[label_beforeOH == i] = 1
            label_beforeOH[label_beforeOH == temp] = 0
            label_OH = label_beforeOH

        else:
            label_beforeOH[label_beforeOH != i] = 0
            label_beforeOH[label_beforeOH == i] = 1

            label_OH = np.concatenate((label_OH, label_beforeOH), axis=-1)
        
    return label_OH


def calculate_percent_binary(lab):
    
    all_targt = np.sum(lab[:,:,:,1])
    all_other = np.sum(lab[:,:,:,0])
    all_pixel = np.sum(lab[:,:,:,:])

    perc_targt = all_targt / all_pixel
    perc_other = all_other / all_pixel

    return perc_targt, perc_other
        
        
def increase_target_percent_by_deleting_sparse_patch(lab, img, division):

    index_notgt_list = []
    index_tgt_list = []

    for i in range(lab.shape[0]):

        patch = lab[i:i+1,:,:,:]
        perc_targt, perc_other = calculate_percent_binary(patch)

        if perc_targt < division or perc_targt == division:
            index_notgt_list.append(i)
        else:
            index_tgt_list.append(i)

    lab_tgt = np.delete(lab, index_notgt_list, axis=0)
    img_tgt = np.delete(img, index_notgt_list, axis=0)
    print('Label:', lab_tgt.shape, 'Image:', img_tgt.shape)

    perc_targt, perc_other = calculate_percent_binary(lab_tgt)
    print('Target:', perc_targt, 'Other:', perc_other)

    return img_tgt, lab_tgt    


def save_array_as_npyfile(path_arr, arr):
    
    arr = arr.astype('float32')
    
    np.save(path_arr, arr)  
    
def split_image_label_npy(EXP, path_out_npy, division, img, lab, data_type, rs):
    
    img_remain, img_target, lab_remain, lab_target = train_test_split(img, lab, test_size=division, random_state=rs)
    print(data_type)
    print(img_target.shape, lab_target.shape)

    percent = "_" + str(int((division) * 100))

    path_out_patch_img_target = define_path_of_output_patch(EXP, path_out_npy, "img_" + data_type + percent)
    path_out_patch_lab_target = define_path_of_output_patch(EXP, path_out_npy, "lab_" + data_type + percent)

    # save image and label patches
    save_array_as_npyfile(path_out_patch_img_target, img_target)
    save_array_as_npyfile(path_out_patch_lab_target, lab_target)    
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Visualization
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def visualize_image_4bands_RGB_composite(img_list, nbands, figure_size):
    
    for i in range(len(img_list)):

        image = img_list[i]
        print("Tile {}: Shape {}".format(i, image.shape))

        """
        # Show four bands
        ep.plot_bands(image, 
                      cmap = 'gist_earth', 
                      figsize = (figure_size, figure_size), 
                      cols = nbands, 
                      cbar = False)
        plt.show()
        plt.pause(0.5)

        """    
        # RGB composite with strech
        ep.plot_rgb(image,
                rgb=(0, 1, 2),
                stretch=True, # False -->no Strech
                str_clip=0.2,
                figsize=(figure_size, figure_size))
        plt.show()
        plt.pause(0.5)
        
        
def visualize_label(lab_list, nclass, figure_size):
    
    class_names = ["Other", "Dwelling"]
    plt.rcParams["figure.figsize"] = (figure_size, figure_size)
    
    for i in range(len(lab_list)):

        label = lab_list[i][0]

        f, ax = plt.subplots()
        ax.axis('off')
        im = ax.imshow(label, cmap="gnuplot")
        im_ax = ax.imshow(label)
        leg_neg = ep.draw_legend(im_ax = im_ax, titles = class_names)

        plt.show()
        plt.pause(0.5)
                    
        
def visualize_img_patch(num_patch, step_patch, patch_img):
    
    index_patch_min = 0
    index_patch_max = index_patch_min + num_patch * step_patch

    col = 4
    row = round(num_patch / col)

    fig_size = 40

    patch_img_list = [patch_img[i] for i in range(index_patch_min, index_patch_max, step_patch)]
    patch_img_list_swap = [swap_array_axis_for_visualization(img) for img in patch_img_list]

    fig, ax_list = plt.subplots(row, col, figsize=(fig_size, fig_size))

    for i in range(row):
        for j in range(col):    
            num = col * i + j 

            img = patch_img_list_swap[num]
            ax = ax_list[i][j]
            ax.axis('off')
            ep.plot_rgb(
                        img,
                        rgb=(0, 1, 2),
                        ax=ax,
                        stretch=True,
                        str_clip=0.5,
                        )

            ax.set_adjustable("datalim")
            plt.tight_layout()

def visualize_lab_patch(num_patch, step_patch, patch_lab):

    index_patch_min = 0
    index_patch_max = index_patch_min + num_patch * step_patch

    col = 4
    row = round(num_patch / col)

    fig_size = 40

    patch_lab_list = [patch_lab[i] for i in range(index_patch_min, index_patch_max, step_patch)]
    patch_lab_list_swap = [swap_array_axis_for_visualization(lab) for lab in patch_lab_list]

    fig, ax_list = plt.subplots(row, col, figsize=(fig_size, fig_size))

    for i in range(row):
        for j in range(col):    
            num = col * i + j 

            lab = patch_lab_list_swap[num][1]
            ax = ax_list[i][j]
            ax.axis('off')
            ax.imshow(lab)
            ax.set_adjustable("datalim")
            plt.tight_layout()

            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Data Augmentation
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        
def data_augmentation(X, Y, DA_type):
    
    if DA_type == "FL1":
        # flip: left to right
        X_da = np.flip(X, 2)
        Y_da = np.flip(Y, 2)
        
    elif DA_type == "FL2":
        # flip:left to right
        X_da1 = np.flip(X, 2)
        Y_da1 = np.flip(Y, 2)
        
        X_da2 = np.flip(X, 1)
        Y_da2 = np.flip(Y, 1)
        
        X_da = np.concatenate((X_da1, X_da2), axis=0)
        Y_da = np.concatenate((Y_da1, Y_da2), axis=0)
        
    elif DA_type == "RO1":
        # rotate: 90 degrees (from right to left)
        X_da = np.rot90(X, 1, axes=(1, 2))
        Y_da = np.rot90(Y, 1, axes=(1, 2))

    elif DA_type == "RO2":
        # rotate: 90 degrees (from right to left)
        X_da1 = np.rot90(X, 1, axes=(1, 2))
        Y_da1 = np.rot90(Y, 1, axes=(1, 2))
        
        # rotate: 180 degrees (from right to left)
        X_da2 = np.rot90(X, 2, axes=(1, 2))
        Y_da2 = np.rot90(Y, 2, axes=(1, 2))
        
        X_da = np.concatenate((X_da1, X_da2), axis=0)
        Y_da = np.concatenate((Y_da1, Y_da2), axis=0)
        
    elif DA_type == "RO3":
        # rotate: 90 degrees (from right to left)
        X_da1 = np.rot90(X, 1, axes=(1, 2))
        Y_da1 = np.rot90(Y, 1, axes=(1, 2))
        
        # rotate: 180 degrees (from right to left)
        X_da2 = np.rot90(X, 2, axes=(1, 2))
        Y_da2 = np.rot90(Y, 2, axes=(1, 2))
        
        # rotate: 270 degrees (from right to left)
        X_da3 = np.rot90(X, 3, axes=(1, 2))
        Y_da3 = np.rot90(Y, 3, axes=(1, 2))
        
        X_da = np.concatenate((X_da1, X_da2, X_da3), axis=0)
        Y_da = np.concatenate((Y_da1, Y_da2, Y_da3), axis=0)
        
    elif DA_type == "ALL":
        
        # flip: left to right
        X_da1 = np.flip(X, 2)
        Y_da1 = np.flip(Y, 2)
        
        X_da2 = np.flip(X, 1)
        Y_da2 = np.flip(Y, 1)
        
        X_da3 = np.rot90(X, 1, axes=(1, 2))
        Y_da3 = np.rot90(Y, 1, axes=(1, 2))
        
        X_da4 = np.rot90(X, 2, axes=(1, 2))
        Y_da4 = np.rot90(Y, 2, axes=(1, 2))
        
        X_da5 = np.rot90(X, 3, axes=(1, 2))
        Y_da5 = np.rot90(Y, 3, axes=(1, 2))
        
        X_da = np.concatenate((X_da1, X_da2, X_da3, X_da4, X_da5), axis=0)
        Y_da = np.concatenate((Y_da1, Y_da2, Y_da3, Y_da4, Y_da5), axis=0)
        
    return X_da, Y_da   



        
        
        
        
        
        