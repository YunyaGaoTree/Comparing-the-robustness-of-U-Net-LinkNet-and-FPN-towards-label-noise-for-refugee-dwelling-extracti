# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Library

# Basics
import os
import numpy as np
import pandas as pd
from patchify import patchify, unpatchify
import rasterio
from sklearn.model_selection import train_test_split

# Data processing functions
from Data_Preparation import *
import rasterio
from osgeo import gdal, ogr, osr
import sys
#import gdal
#gdal.UseExceptions()


# Visualization
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import deep learning framework and model library
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam

gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
print(f"Tensor Flow Versio n: {tf.__version__}")

import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

from segmentation_models.metrics import f1_score, iou_score
from segmentation_models import Unet, PSPNet, Linknet, FPN
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load testing data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load and Select Data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def define_path_of_output_img_lab_patch(EXP, path_patch, data_type):
    
    path_patch_img = define_path_of_output_patch(EXP, path_patch, "img_"+data_type)
    path_patch_lab = define_path_of_output_patch(EXP, path_patch, "lab_"+data_type)
    
    return path_patch_img, path_patch_lab


def get_training_validation_img_patch(path_data, EXP):
    
    # define path of patches 
    path_patch = os.path.join(path_data, EXP + '_Train_Val_Npy')
    path_patch_img_tr = path_patch_img = define_path_of_output_patch(EXP, path_patch, "img_tr")
    path_patch_img_va = path_patch_img = define_path_of_output_patch(EXP, path_patch, "img_va")
    
    # load patches
    tr_img = np.load(path_patch_img_tr)
    va_img = np.load(path_patch_img_va)
    
    return tr_img, va_img

def get_training_validation_lab_patch(path_data, EXP):
    
    # define path of patches 
    path_patch = os.path.join(path_data, EXP + '_Train_Val_Npy')
    path_patch_lab_tr = path_patch_img = define_path_of_output_patch(EXP, path_patch, "lab_tr")
    path_patch_lab_va = path_patch_img = define_path_of_output_patch(EXP, path_patch, "lab_va")
    
    # load patches
    tr_lab = np.load(path_patch_lab_tr)
    va_lab = np.load(path_patch_lab_va)
    
    return tr_lab, va_lab

def get_training_validation_lab_patch_from_prediction(path_data, EXP, pred_new):
    
    # define path of patches 
    path_patch = os.path.join(path_data, EXP + '_Train_Val_Npy')
    path_patch_lab_tr = path_patch_img = define_path_of_output_patch(EXP + pred_new, path_patch, "lab_tr")
    path_patch_lab_va = path_patch_img = define_path_of_output_patch(EXP + pred_new, path_patch, "lab_va")
    
    # load patches
    tr_lab = np.load(path_patch_lab_tr)
    va_lab = np.load(path_patch_lab_va)
    
    print("Y_tr.shape: {}, Y_va.shape:{}".format(tr_lab.shape, va_lab.shape))
    
    return tr_lab, va_lab

def get_training_validation_patch(path_data, EXP):
    
    # define path of patches 
    tr_img, va_img = get_training_validation_img_patch(path_data, EXP)
    tr_lab, va_lab = get_training_validation_lab_patch(path_data, EXP)

    print("Shape of training image patches: {}".format(tr_img.shape))
    print("Shape of training label patches: {}".format(tr_lab.shape))
    print("Shape of validation image patches: {}".format(va_img.shape))
    print("Shape of validation label patches: {}".format(va_lab.shape))
    
    # calculate percent of objects
    perc_tr = calculate_percent_binary(tr_lab)
    perc_va = calculate_percent_binary(va_lab)
    
    print("Percent of target objects in training label: {}".format(perc_tr))
    print("Percent of target objects in validation label: {}".format(perc_va))
    
    return tr_img, tr_lab, va_img, va_lab


# Get the list of input arrays from input geotiff data
def get_tile_array_list_from_geotiff(path_data, folder_img_ts_crop, folder_lab_ts_crop, ignore_img_value, ignore_lab_value, band_selection):
    
    path_folder_img_ts = os.path.join(path_data, folder_img_ts_crop)
    path_folder_lab_ts = os.path.join(path_data, folder_lab_ts_crop)# + "_" + EXP

    # Get the list of images and labels
    list_ts_img = get_tif_list(path_folder_img_ts)
    list_ts_lab = get_tif_list(path_folder_lab_ts)

    # Import geotiff as arrays in a list
    arr_ts_img_list = convert_tif_list_into_array_list(list_ts_img, ignore_img_value)
    arr_ts_lab_list = convert_tif_list_into_array_list(list_ts_lab, ignore_lab_value)

    # Show the shape and number of arrays
    show_array_list_info(arr_ts_img_list, "testing images")
    show_array_list_info(arr_ts_lab_list, "testing labels")
    
    return arr_ts_img_list, arr_ts_lab_list

def get_patch_from_tile_array_list(arr_ts_img_list, arr_ts_lab_list, index_ts, nclass, patchsize, nbands_all, band_selection):
    
    step = patchsize
    
    img_array = arr_ts_img_list[index_ts]
    img_array = swap_array_axis(img_array)

    lab_array = arr_ts_lab_list[index_ts]
    lab_array = swap_array_axis(lab_array)
    lab_array = one_hot_encoding(lab_array, nclass)

    patch_img_ts_4b = convert_single_tile_to_patch(img_array, patchsize, nbands_all, step)
    patch_img_ts = from_img_4bands_to_3bands(patch_img_ts_4b, band_selection)

    patch_lab_ts = convert_single_tile_to_patch(lab_array, patchsize, nclass, step)

    print(patch_img_ts.max(), patch_lab_ts.max())
    print(patch_img_ts.shape, patch_lab_ts.shape)

    perc_targt, perc_other = calculate_percent_binary(patch_lab_ts)
    print("Overall percent of objects in patches: {}, {}".format(perc_targt, perc_other))
    
    return patch_img_ts_4b, lab_array, patch_img_ts, patch_lab_ts 

def get_label_data_for_comparison(path_data, folder_lab, label_type, index_lab, ignore_lab_value, nclass):
    
    # Define path of labels
    path_folder_lab = os.path.join(path_data, folder_lab)
    
    # Get the list of labels
    list_lab = get_tif_list(path_folder_lab)

    # Import geotiff as arrays in a list
    arr_lab_list = convert_tif_list_into_array_list(list_lab, ignore_lab_value)

    # Show the shape and number of arrays
    show_array_list_info(arr_lab_list, label_type)

    # Convert 3 dimensional array to 4 dimensional array
    lab_array = arr_lab_list[index_lab]
    lab_array = swap_array_axis(lab_array)
    lab_array = one_hot_encoding(lab_array, nclass)
    lab_array = np.expand_dims(lab_array, axis=0)

    return lab_array

# Normalization by minus mean value of each layer and divided by square standard deviation
def normalize_single_layer(layer):
    
    layer_mean = np.mean(layer)
    layer_std = float(np.std(layer, axis=(0,1,2)))
    layer_norm = (layer - layer_mean) / layer_std    
    
    return layer_norm

def normalize_image(img):
    # either 3 bands or 4 bands 
    
    num_layer = img.shape[-1]
    
    for i in range(num_layer):
        
        layer = img[:,:,:,i:i+1]
        layer_norm = normalize_single_layer(layer)
        
        if i == 0:
            img_norm = layer_norm
            
        else:
            img_norm = np.concatenate((img_norm, layer_norm), axis=-1)

    return img_norm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Definition of important hyperparameters and functions in CNN model
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def step_decay(initial_lrate, epoch):
    
    initial_lrate = 1e-2
    drop = 0.5
    epochs_drop = 10.0
    
    lrate = initial_lrate * tf.math.pow(drop, tf.math.floor((1+epoch)/epochs_drop))
    
    return lrate
        
        
def balanced_cross_entropy(beta):
    
    def convert_to_logits(y_pred):
        
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        y_pred = tf.math.log(y_pred / (1 - y_pred))
        
        return y_pred

    def loss(y_true, y_pred):
        
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
    
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss * (1 - beta))

    return loss


def save_history(path_weights, history, model_type):

    hist_df = pd.DataFrame(history.history) 
    
    # save to csv:
    hist_csv_file = os.path.join(path_weights, model_type + "_history.csv") 
    
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

        
def reshape_prediction_by_unpatchify(prediction, patchsize, nclass, lab_array):
    
    num_row = int(lab_array.shape[0]/patchsize)
    num_col = int(lab_array.shape[1]/patchsize)

    prediction_reshape = prediction.reshape((num_row, num_col, 1, patchsize, patchsize, nclass))

    target_shape = (lab_array.shape[0], lab_array.shape[1], nclass)

    prediction_reshape_unpatch = unpatchify(prediction_reshape, target_shape)

    return prediction_reshape_unpatch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Define, Compile, Fit models
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        
def run_unet(EXP, model_name, backbone, pretrained_weights, path_weights, X_tr, Y_tr, X_va, Y_va, callbacks, activation_func, loss_func, metrics_list, OPT, patchsize, nclass, nbands, epoch, batchsize):
    
    model = Unet(backbone, 
             encoder_weights=pretrained_weights, 
             classes=nclass, 
             input_shape=(patchsize, patchsize, nbands), 
             activation=activation_func)

    beta = tf.reduce_mean(1 - Y_tr)
    model.compile(loss=loss_func(beta), metrics=metrics_list, optimizer=OPT)

    model_type = EXP + "_" + model_name + "_" + backbone

    print("Starting training:", model_type)
    history = model.fit( x=X_tr,
                         y=Y_tr,
                         validation_data=(X_va, Y_va), 
                         batch_size=batchsize,
                         epochs=epoch,
                         verbose=2,
                         callbacks=callbacks,
                         shuffle=True
                        )

    # save weights and training and validation history
    weights_output_path = os.path.join(path_weights, model_type + "_weights.h5")
    model.save_weights(weights_output_path)
    save_history(path_weights,history, model_type)   

    # plot training and validation accuracy and loss 
    plot_history_direct_from_model(history)
    
# Dont save weights for unet
def run_unet_0(backbone, pretrained_weights, X_tr, Y_tr, X_va, Y_va, callbacks, activation_func, loss_func, metrics_list, OPT, patchsize, nclass, nbands, epoch, batchsize):
    
    model = Unet(backbone, 
             encoder_weights=pretrained_weights, 
             classes=nclass, 
             input_shape=(patchsize, patchsize, nbands), 
             activation=activation_func)

    beta = tf.reduce_mean(1 - Y_tr)
    model.compile(loss=loss_func(beta), metrics=metrics_list, optimizer=OPT)

    history = model.fit( x=X_tr,
                         y=Y_tr,
                         validation_data=(X_va, Y_va), 
                         batch_size=batchsize,
                         epochs=epoch,
                         verbose=2,
                         callbacks=callbacks,
                         shuffle=True
                        )

    return history, model

def save_history_0(path_weights, weights_folder, history, model_type):

    hist_df = pd.DataFrame(history.history) 
    
    # save to csv:
    hist_csv_file = os.path.join(path_weights, weights_folder, model_type + "_history.csv") 
    
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


def save_weights_history(path_weights, weights_folder, model_type, model, history):
    
    # save weights and training and validation history
    weights_output_path = os.path.join(path_weights, weights_folder, model_type + "_weights.h5")
    model.save_weights(weights_output_path)
    save_history_0(path_weights, weights_folder, history, model_type)

def run_pspnet(backbone, pretrained_weights, X_tr, Y_tr, X_va, Y_va, callbacks, activation_func, loss_func, metrics_list, OPT, patchsize, nclass, nbands, epoch, batchsize):
    
    model = PSPNet(backbone, 
             encoder_weights=pretrained_weights, 
             classes=nclass, 
             input_shape=(patchsize, patchsize, nbands), 
             activation=activation_func)

    beta = tf.reduce_mean(1 - Y_tr)
    model.compile(loss=loss_func(beta), metrics=metrics_list, optimizer=OPT)

    history = model.fit( x=X_tr,
                         y=Y_tr,
                         validation_data=(X_va, Y_va), 
                         batch_size=batchsize,
                         epochs=epoch,
                         verbose=2,
                         callbacks=callbacks,
                         shuffle=True
                        )
    
    return history, model

def run_linknet(backbone, pretrained_weights, X_tr, Y_tr, X_va, Y_va, callbacks, activation_func, loss_func, metrics_list, OPT, patchsize, nclass, nbands, epoch, batchsize):
    
    model = Linknet(backbone, 
             encoder_weights=pretrained_weights, 
             classes=nclass, 
             input_shape=(patchsize, patchsize, nbands), 
             activation=activation_func)

    beta = tf.reduce_mean(1 - Y_tr)
    model.compile(loss=loss_func(beta), metrics=metrics_list, optimizer=OPT)

    history = model.fit( x=X_tr,
                         y=Y_tr,
                         validation_data=(X_va, Y_va), 
                         batch_size=batchsize,
                         epochs=epoch,
                         verbose=2,
                         callbacks=callbacks,
                         shuffle=True
                        )
    
    return history, model
    
def run_fpn(backbone, pretrained_weights, X_tr, Y_tr, X_va, Y_va, callbacks, activation_func, loss_func, metrics_list, OPT, patchsize, nclass, nbands, epoch, batchsize):
    
    model = FPN(backbone, 
             encoder_weights=pretrained_weights, 
             classes=nclass, 
             input_shape=(patchsize, patchsize, nbands), 
             activation=activation_func)

    beta = tf.reduce_mean(1 - Y_tr)
    model.compile(loss=loss_func(beta), metrics=metrics_list, optimizer=OPT)

    history = model.fit( x=X_tr,
                         y=Y_tr,
                         validation_data=(X_va, Y_va), 
                         batch_size=batchsize,
                         epochs=epoch,
                         verbose=2,
                         callbacks=callbacks,
                         shuffle=True
                        )
    
    return history, model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load weights to model for prediction
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def load_weights_to_unet(EXP, model_name, backbone, pretrained_weights, path_weights, X_ts, Y_ts, callbacks, activation_func, loss_func, metrics_list, OPT, patchsize, nclass, nbands):
    
    model = Unet(backbone, 
             encoder_weights=pretrained_weights, 
             classes=nclass, 
             input_shape=(patchsize, patchsize, nbands), 
             activation=activation_func)

    beta = tf.reduce_mean(1 - Y_ts)
    model.compile(loss=loss_func(beta), metrics=metrics_list, optimizer=OPT)

    # define path of weights
    model_type = EXP + "_" + model_name + "_" + backbone
    weights_in_path = os.path.join(path_weights, model_type+"_weights.h5")
    
    model.load_weights(weights_in_path)
    
    return model
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Accuracy Assessment
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def accuracy_assessment(actual, predicted, nclass):
    
    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = []
    
    for i in range(nclass):
        
        actual_single = actual[:, :, :, i:i+1]
        predicted_single = predicted[:, :, :, i:i+1]
        
        precision, recall, f1, iou, _ = precision_recall_f1_iou_oa(actual_single, predicted_single)
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        iou_list.append(iou)
        
    acc_single_dict = {
                        "f1 score": f1_list,
                        "iou": iou_list,
                        "precision": precision_list,
                        "recall": recall_list,
                        }
    
    _, __, f1_oa, iou_oa, oa = precision_recall_f1_iou_oa(actual, predicted)
    
    acc_overall_dict = {
                        "overall accuracy": [oa],
                        "overall f1": [f1_oa],
                        "overall iou": [iou_oa],
                        "mean iou": [np.mean(np.array(iou_list))]
                        }
    
    pd_acc_single_dict = pd.DataFrame(acc_single_dict)
    pd_acc_overall_dict = pd.DataFrame(acc_overall_dict)
    
    return pd_acc_single_dict, pd_acc_overall_dict


def precision_recall_f1_iou_oa(actual, predicted):
    
    predicted = np.round(predicted)
    
    TP = tf.math.count_nonzero(predicted * actual)
    TN = tf.math.count_nonzero((predicted - 1) * (actual - 1))
    FP = tf.math.count_nonzero(predicted * (actual - 1))
    FN = tf.math.count_nonzero((predicted - 1) * actual)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    iou = precision * recall / (precision + recall - precision * recall)
    oa = (TP + TN) / (TP + TN + FP + FN)
    
    return float(precision), float(recall), float(f1), float(iou), float(oa)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         
# Visualization
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def plot_history_direct_from_model(history):
    
    keys = list(history.history.keys())

    tr_acc_key = keys[1]
    va_acc_key = keys[3]

    tr_loss_key = keys[0]
    va_loss_key = keys[2]

    # summarize history for accuracy
    plt.plot(history.history[tr_acc_key])
    plt.plot(history.history[va_acc_key])
    plt.title('model accuracy '+ tr_acc_key)
    plt.ylabel(tr_acc_key)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='center left', bbox_to_anchor=(1.0, 0.9))
    plt.show()

    # summarize history for loss
    plt.plot(history.history[tr_loss_key])
    plt.plot(history.history[va_loss_key])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='center left', bbox_to_anchor=(1.0, 0.9))
    plt.show()

def plot_history_from_saved_csv(EXP, model_name, backbone, path_weights):
    
    model_type = EXP + "_" + model_name + "_" + backbone
    path_history = os.path.join(path_weights, model_type + "_history.csv") 
    history = pd.read_csv(path_history)
    
    keys = list(history.keys())

    tr_acc_key = keys[2]
    va_acc_key = keys[4]

    tr_loss_key = keys[1]
    va_loss_key = keys[3]

    # summarize history for accuracy
    plt.plot(history[tr_acc_key])
    plt.plot(history[va_acc_key])
    plt.title('model accuracy '+ tr_acc_key)
    plt.ylabel(tr_acc_key)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='center left', bbox_to_anchor=(1.0, 0.9))
    plt.show()

    # summarize history for loss
    plt.plot(history[tr_loss_key])
    plt.plot(history[va_loss_key])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='center left', bbox_to_anchor=(1.0, 0.9))
    plt.show()


def visualize_prediction(prediction, nclass, figure_size):

    prediction_vis = np.swapaxes(prediction, 2, 0)
    prediction_vis = np.swapaxes(prediction_vis, 2, 1)

    prediction_vis_list = [np.round(prediction_vis[1:])]

    visualize_label(prediction_vis_list, nclass, figure_size)    
        
        
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         
# Convert prediction into geotiff
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def save_prediction_to_geotiff(path_in_ras, prediction, path_output_ras):

    dataset = rasterio.open(path_in_ras)
    transform = dataset.transform
    crs = dataset.crs
    output_ras = prediction[:,:,1]

    with rasterio.open(
      path_output_ras,
      'w',
      driver='GTiff',
      height=output_ras.shape[0],
      width=output_ras.shape[1],
      count=1,
      dtype=output_ras.dtype,
      crs=crs,
      transform=transform,) as dst:
        dst.write(output_ras, 1)

    message = path_output_ras + " saved."

    return message

  
def convert_geotiff_to_polygons(path_in_raster, path_out_shp):

    sourceRaster = gdal.Open(path_in_raster)
    band = sourceRaster.GetRasterBand(1)

    prj = sourceRaster.GetProjection()
    srs = osr.SpatialReference(wkt=prj)

    outShapefile = "polygonized"
    driver = ogr.GetDriverByName("ESRI Shapefile")

    if os.path.exists(path_out_shp):
        driver.DeleteDataSource(path_out_shp)

    outDatasource = driver.CreateDataSource(path_out_shp)
    outLayer = outDatasource.CreateLayer(outShapefile, srs=srs, geom_type=ogr.wkbPolygon)

    field_name = 'Area'
    newField = ogr.FieldDefn(field_name, ogr.OFTInteger)
    outLayer.CreateField(newField)

    gdal.Polygonize(band, None, outLayer, -1, [], callback=None)

    for feature in outLayer:
        geom = feature.GetGeometryRef()
        area = geom.GetArea()
        feature.SetField(field_name, area)
        outLayer.SetFeature(feature)

    outLayer.SetAttributeFilter("Area > 10000")

    for feature in outLayer:
        outLayer.DeleteFeature(feature.GetFID())

    outLayer.SetAttributeFilter("Area < 1")

    for feature in outLayer:
        outLayer.DeleteFeature(feature.GetFID())

    outDatasource.Destroy()
    sourceRaster = None

    return srs

def clip_polygons_with_mask(path_in_shp, path_mask_shp, path_out_shp, srs):

    # Input
    driverName = "ESRI Shapefile"
    driver = ogr.GetDriverByName(driverName)
    inDataSource = driver.Open(path_in_shp, 0)
    inLayer = inDataSource.GetLayer()
    print(inLayer.GetFeatureCount())

    # Clip
    inClipSource = driver.Open(path_mask_shp, 0)
    inClipLayer = inClipSource.GetLayer()
    print(inClipLayer.GetFeatureCount())

    # Clipped Shapefile
    outDataSource = driver.CreateDataSource(path_out_shp)
    outLayer = outDataSource.CreateLayer('Clipped', srs=srs, geom_type=ogr.wkbMultiPolygon)

    ogr.Layer.Clip(inLayer, inClipLayer, outLayer)
    print(outLayer.GetFeatureCount())
    inDataSource.Destroy()
    inClipSource.Destroy()
    outDataSource.Destroy()

def load_mask(path_pred_base, study_site):

    mask_name = "Mask_" + study_site + ".npy"
    path_mask = os.path.join(path_pred_base, "Mask", mask_name)
    mask = np.load(path_mask)
    mask_single = np.expand_dims(mask[0], axis=-1)
    mask_final = np.concatenate((mask_single, mask_single), axis=-1)

    return mask_final
  
def intersect_using_spatial_index(source_gdf, intersecting_gdf):
    """
    Conduct spatial intersection using spatial index for candidates GeoDataFrame to make queries faster.
    Note, with this function, you can have multiple Polygons in the 'intersecting_gdf' and it will return all the points
    intersect with ANY of those geometries.
    """
    source_sindex = source_gdf.sindex
    possible_matches_index = []
    
    # 'itertuples()' function is a faster version of 'iterrows()'
    for other in intersecting_gdf.itertuples():
        bounds = other.geometry.bounds
        c = list(source_sindex.intersection(bounds))
        possible_matches_index += c
    
    # Get unique candidates
    unique_candidate_matches = list(set(possible_matches_index))
    possible_matches = source_gdf.iloc[unique_candidate_matches]
    
    # Conduct the actual intersect
    result = possible_matches.loc[possible_matches.intersects(intersecting_gdf.unary_union)]
    
    return result
    
    """
    import geopandas as gpd
    df_pred = gpd.read_file(path_shp_pred)
    df_true = gpd.read_file(path_shp_true)
    
    # 4248: source_gdf:df_pred, intersecting_gdf:df_true
    results = intersect_using_spatial_index(df_true, df_pred)
    """