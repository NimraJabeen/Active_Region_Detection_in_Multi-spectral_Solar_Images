import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
import keras_frcnn.resnet_mulBranch_2spects_TIF_dim_reduction_UniqueRoIPool as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import MLMT_data_generators
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure



def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2


def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h

def box_area(box):
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    return area_a


def iou(a, b):
	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0
	area_i = intersection(a, b)
	area_u = union(a, b, area_i)
	return float(area_i) / float(area_u + 1e-6)

def overlap_over_GT_area(a, b):

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    intesection_area = intersection(a, b)
    GT_box_area = box_area(b)

    return float(intesection_area) / float(GT_box_area)

def overlap_over_predictedBox_area(a, b):

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    intesection_area = intersection(a, b)

    Predicted_box_area = box_area(a)

    return float(intesection_area) / float(Predicted_box_area)



def get_map(img_name, img_copy, pred, gt, f, spect):

    num_of_GT_boxes = len(gt)

    print('len gt ' ,len(gt))
    TP, FP, FN, TN = 0, 0, 0, 0
    IoU_value = 0
    T = {}
    P = {}
    fx, fy = f

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])

    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for gt_box in gt:
        gt_class = gt_box['class']
        gt_x1 = gt_box['x1'] / fx
        gt_x2 = gt_box['x2'] / fx
        gt_y1 = gt_box['y1'] / fy
        gt_y2 = gt_box['y2'] / fy

        int_gt_x1, int_gt_y1, int_gt_x2, int_gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)
        real_gt_x1, real_gt_y1, real_gt_x2, real_gt_y2 = get_real_coordinates(ratio, int_gt_x1, int_gt_y1, int_gt_x2, int_gt_y2)
        cv2.rectangle(img_copy, (real_gt_x1, real_gt_y1), (real_gt_x2, real_gt_y2) ,(0,0,255),5)

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']

        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []

        int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2 = int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)
        real_pred_x1, real_pred_y1, real_pred_x2, real_pred_y2 = get_real_coordinates(ratio, int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2)

            
        found_match = False

        sum_GT_intesections = 0
        count = 0

        pick_intersected_GT = []

        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1'] / fx
            gt_x2 = gt_box['x2'] / fx
            gt_y1 = gt_box['y1'] / fy
            gt_y2 = gt_box['y2'] / fy
            gt_seen = gt_box['bbox_matched']

            if gt_class != pred_class:
                continue


            overlap_over_GT = overlap_over_GT_area((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            overlap_over_predictedBox = overlap_over_predictedBox_area((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))

            intersection_pred_GT = intersection((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            gt_box_area = box_area((gt_x1, gt_y1, gt_x2, gt_y2))
            pred_box_area = box_area((pred_x1, pred_y1, pred_x2, pred_y2))

            if (intersection_pred_GT/pred_box_area) >=  0.5  or (intersection_pred_GT/gt_box_area) >= 0.5:

                pick_intersected_GT.append(count)

                sum_GT_intesections += intersection_pred_GT

                IoU_value += MLMT_data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))

            count += 1

        Predicted_box_area = box_area((pred_x1, pred_y1, pred_x2, pred_y2))
        

        if len(pick_intersected_GT) != 0:
            found_match = True
            for idx in pick_intersected_GT:
                gt[idx]['bbox_matched'] = True
                T[pred_class].append(int(found_match))
                P[pred_class].append(pred_prob)
                
            TP = TP + len(pick_intersected_GT)

            int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2 = int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)
            real_pred_x1, real_pred_y1, real_pred_x2, real_pred_y2 = get_real_coordinates(ratio, int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2)
            cv2.rectangle(img_copy, (real_pred_x1, real_pred_y1), (real_pred_x2, real_pred_y2) ,(0,255,0),5)


        if int(found_match)==0:
            int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2 = int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)
            real_pred_x1, real_pred_y1, real_pred_x2, real_pred_y2 = get_real_coordinates(ratio, int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2)
            cv2.rectangle(img_copy, (real_pred_x1, real_pred_y1), (real_pred_x2, real_pred_y2) ,(0,255,0),5)

            FP += 1
            T[pred_class].append(int(found_match))
            P[pred_class].append(pred_prob)


    for gt_box in gt:

        gt_x1 = gt_box['x1'] / fx
        gt_x2 = gt_box['x2'] / fx
        gt_y1 = gt_box['y1'] / fy
        gt_y2 = gt_box['y2'] / fy
        
        if not gt_box['bbox_matched']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)
            FN += 1

            int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)
            real_pred_x1, real_pred_y1, real_pred_x2, real_pred_y2 = get_real_coordinates(ratio, int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2)

    print('TP, FP, FN are as follows ', TP, FP, FN)
    result_path = './Detections_visualization/' + spect + '/' + img_name[:-3] + '.png'
    print('visualised result_path :', result_path)

    cv2.imwrite(result_path, img_copy)

    print('T and P pairs :' , T, P)

    array_T = np.asarray(T['AR'])
    array_P = np.asarray(P['AR'])
    FP_new =  np.count_nonzero(array_T == 0)
    FN_new =  np.count_nonzero(array_P == 0)
    print('FP_new  and FP:', FP_new, FP)
    print('FN_new :', FN_new )

    return T, P, TP, FP, FN_new, IoU_value


sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path_1", dest="test_path_1", default ='F:/Data/label_testing_195.txt' , help="Path to test data 1.")
parser.add_option("-q", "--path_2", dest="test_path_2", default ='F:/Data/label_testing_304.txt' , help="Path to test data 2.")
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", default ="config.pickle", help=
"configuration file")
parser.add_option("-o", "--parser", dest="parser", default = "simple", help="Parser to use. One of simple or pascal_voc")

(options, args) = parser.parse_args()

if not options.test_path_1:
    parser.error('Error: path_1 to test data must be specified. Pass --path_1 to command line')

if not options.test_path_1:
    parser.error('Error: path_2 to test data must be specified. Pass --path_2 to command line')

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data_for_mAP
elif options.parser == 'simple':
    from keras_frcnn.MLMT_simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False
C.num_rois = 100

def format_img_size(img, C):
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio

def format_img(img, C):
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        f = img_min_side / width
        new_height = int(f * height)
        new_width = int(img_min_side)
    else:
        f = img_min_side / height
        new_width = int(f * width)
        new_height = int(img_min_side)
    fx = width / float(new_width)
    fy = height / float(new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, fx, fy

    
def get_imlist_starting_with(path, starting_str):
  return [os.path.join(path, f) for f in os.listdir(path) if f.startswith(starting_str)]


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (1024, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)

input_img_1 = Input(shape=input_shape_img)
input_img_2 = Input(shape=input_shape_img)
roi_input_1 = Input(shape=(None, 4))
roi_input_2 = Input(shape=(None, 4))
feature_map_input_1 = Input(shape=input_shape_features)
feature_map_input_2 = Input(shape=input_shape_features)

shared_layers_merged, shared_layers_x1, shared_layers_x2  = nn.nn_base_res_weights_2spectIII(input_img_1, input_img_2, trainable=True)

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

rpn_layers = nn.rpn(shared_layers_merged, shared_layers_x1, shared_layers_x2, num_anchors)

classifier = nn.classifier(feature_map_input_1, feature_map_input_2, roi_input_1, roi_input_2, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model([input_img_1, input_img_2], rpn_layers)
model_classifier_only = Model([feature_map_input_1, feature_map_input_2, roi_input_1, roi_input_2], classifier)
model_classifier = Model([feature_map_input_1, feature_map_input_2, roi_input_1, roi_input_2], classifier)

RPN_model_path = './model/Exp 195+304 - for paper use/' + '1437_RPN.h5'
DET_model_path = './model/Exp 195+304 - for paper use/' + '1649_DET.h5'

print('Loading RPN weights from {}'.format(RPN_model_path))
print('Loading DET weights from {}'.format(DET_model_path))

model_rpn.load_weights(RPN_model_path, by_name=True)
model_classifier_only.load_weights(DET_model_path, by_name=True)


model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

test_images_dir = 'F:/Deep_projects/MulSpect IMG Cls/Arch_two/Data/testing_images'
spect_1 = "195"
spect_2 = "304"

all_images,  _, _  = get_data(options.test_path_1,
                              options.test_path_2,
                              images_dir = test_images_dir,
                              spect_1 = spect_1,
                              spect_2 = spect_2,
                              data_set = 'test')


test_imgs = [s for s in all_images if s['imageset'] == 'test']

T = {}
P = {}

## accumalating detections across all images
accumalating_TP_1, accumalating_FP_1, accumalating_FN_1 = 0, 0, 0
accumalating_TP_2, accumalating_FP_2, accumalating_FN_2 = 0, 0, 0
accumalating_IoU_value_1, accumalating_IoU_value_2 = 0, 0
for idx, img_data in enumerate(test_imgs):
    print('{}/{}'.format(idx, len(test_imgs)))
    st = time.time()

    Spect_1_filepath = img_data['filepath']

    image_1_name = Spect_1_filepath.split('/')[-1]

    image_ID = image_1_name.split('_')[0]

    spect_dir_2 = test_images_dir + '/' + spect_2 + '/'
    corrosponding_image_spect_2 = get_imlist_starting_with(spect_dir_2, image_ID)
    assert len(corrosponding_image_spect_2) == 1
    
    Spect_2_filepath = corrosponding_image_spect_2[0]
    image_2_name = Spect_2_filepath.split('/')[-1]
    

    img_Spect_1 = Image.open(Spect_1_filepath)
    img_Spect_1 = np.array(img_Spect_1)
    
    img_Spect_1 *= 255

    img_Spect_1 = np.repeat(img_Spect_1[..., np.newaxis], 3, -1)

    img_Spect_2 = Image.open(Spect_2_filepath)
    img_Spect_2 = np.array(img_Spect_2)
    img_Spect_2 *= 255
    img_Spect_2 = img_Spect_2.copy()

    img_Spect_2 = np.repeat(img_Spect_2[..., np.newaxis], 3, -1)



    img_1_copy = img_Spect_1.copy()
    img_2_copy = img_Spect_2.copy()


    X_1, fx_1, fy_1 = format_img(img_Spect_1, C)
    _, ratio = format_img_size(img_Spect_2, C)
    X_2, fx_2, fy_2 = format_img(img_Spect_2, C)

    if K.image_dim_ordering() == 'tf':
        X_1 = np.transpose(X_1, (0, 2, 3, 1))
        X_2 = np.transpose(X_2, (0, 2, 3, 1))


    start = time.time()

    [Y1_1, Y2_1, Y1_2, Y2_2, F1, F2] = model_rpn.predict([X_1, X_2])    

    result_1 = roi_helpers.rpn_to_roi(Y1_1, Y2_1, C, K.image_dim_ordering(), overlap_thresh=1,
                                    max_boxes=150)
    result_2 = roi_helpers.rpn_to_roi(Y1_2, Y2_2, C, K.image_dim_ordering(), overlap_thresh=1,
                                          max_boxes=150)


    mul_res = np.concatenate((result_2, result_1), axis=0)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    mul_res[:, 2] -= mul_res[:, 0]
    mul_res[:, 3] -= mul_res[:, 1]

    result_1 = mul_res
    result_2 = mul_res

    
    bbox_threshold = 0.5
    
    # apply the spatial pyramid pooling to the proposed regions
    bboxes_1 = {}
    bboxes_2 = {}
    probs_1 = {}
    probs_2 = {}


    for jk in range(result_1.shape[0] // C.num_rois + 1):
        ROIs_1 = np.expand_dims(result_1[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs_1.shape[1] == 0:
            break

        if jk == result_1.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs_1.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs_1.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs_1
            ROIs_padded[0, curr_shape[1]:, :] = ROIs_1[0, 0, :]
            ROIs_1 = ROIs_padded


        #Detections from Spect_2
        ROIs_2 = np.expand_dims(result_2[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs_2.shape[1] == 0:
            break

        #pad the last slice of the rois to end with size (1, num_rois, 4)
        if jk == result_2.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs_2.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs_2.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs_2
            ROIs_padded[0, curr_shape[1]:, :] = ROIs_2[0, 0, :]
            ROIs_2 = ROIs_padded


        [P_cls_1, P_regr_1, P_cls_2, P_regr_2] = model_classifier_only.predict([F1, F2, ROIs_1, ROIs_2])


        #spect_1
        for ii in range(P_cls_1.shape[1]):
            if np.max(P_cls_1[0, ii, :]) < bbox_threshold or np.argmax(P_cls_1[0, ii, :]) == (P_cls_1.shape[2] - 1): #if its less than a threshold or if its a background
                continue

            cls_name = class_mapping[np.argmax(P_cls_1[0, ii, :])]
            if cls_name not in bboxes_1:
                bboxes_1[cls_name] = []
                probs_1[cls_name] = []

            (x, y, w, h) = ROIs_1[0, ii, :]

            cls_num = np.argmax(P_cls_1[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr_1[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes_1[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
            probs_1[cls_name].append(np.max(P_cls_1[0, ii, :]))


        #spect_2
        for ii in range(P_cls_2.shape[1]):

            if np.max(P_cls_2[0, ii, :]) < bbox_threshold or np.argmax(P_cls_2[0, ii, :]) == (P_cls_2.shape[2] - 1): #if its less than a threshold or if its a background
                continue

            cls_name = class_mapping[np.argmax(P_cls_2[0, ii, :])]
            if cls_name not in bboxes_2:
                bboxes_2[cls_name] = []
                probs_2[cls_name] = []

            (x, y, w, h) = ROIs_2[0, ii, :]

            cls_num = np.argmax(P_cls_2[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr_2[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes_2[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
            probs_2[cls_name].append(np.max(P_cls_2[0, ii, :]))



    all_dets_1 = []
    all_dets_2 = []
    
    #spect_1
    for key in bboxes_1:
        bbox_1 = np.array(bboxes_1[key])

    for key in probs_1:
        bprobs_1 = np.array(probs_1[key])


        new_boxes_1, new_probs_1 = roi_helpers.non_max_suppression_fast_with_probs(bbox_1, bprobs_1, overlap_thresh=.05)
        for jk in range(new_boxes_1.shape[0]):
            (x1, y1, x2, y2) = new_boxes_1[jk, :]
            det_1 = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs_1[jk]}
            all_dets_1.append(det_1)
            
    #spect_2
    for key in bboxes_2:
        bbox_2 = np.array(bboxes_2[key])

    for key in probs_2:
        bprobs_2 = np.array(probs_2[key])

        new_boxes_2, new_probs_2 = roi_helpers.non_max_suppression_fast_with_probs(bbox_2, bprobs_2, overlap_thresh=.05)
        for jk in range(new_boxes_2.shape[0]):
            (x1, y1, x2, y2) = new_boxes_2[jk, :]
            det_2 = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs_2[jk]}
            all_dets_2.append(det_2)

    end = time.time()
    print('time elapsed is :' ,end - start, 'secs')

    print('Elapsed time = {}'.format(time.time() - st))

    #spect_1 :
    t_1, p_1, TP_1, FP_1, FN_1, IoU_value_1 = get_map(image_1_name ,img_1_copy, all_dets_1, img_data['bboxes'], (fx_1, fy_1), spect = spect_1)

    #spect_2 :
    t_2, p_2, TP_2, FP_2, FN_2, IoU_value_2 = get_map(image_2_name ,img_2_copy ,all_dets_2, img_data['bboxes_2'], (fx_2, fy_2),  spect = spect_2)

    accumalating_IoU_value_1 += IoU_value_1
    accumalating_TP_1 += TP_1
    accumalating_FP_1 += FP_1
    accumalating_FN_1 += FN_1



    #precision and recall for all the testing data
    accumalating_precision_1 = accumalating_TP_1/(accumalating_TP_1 + accumalating_FP_1)
    accumalating_recall_1 = accumalating_TP_1/(accumalating_TP_1 + accumalating_FN_1)
    print('accumulating (for all previous images) precision band 1', accumalating_precision_1)
    print('accumulating (for all previous images) recall band 1', accumalating_recall_1)
    print('accumulating (for all previous images) F1 band 1', (2*accumalating_precision_1*accumalating_recall_1)/(accumalating_precision_1+accumalating_recall_1))
    

    #spect_2
    accumalating_IoU_value_2 += IoU_value_2
    accumalating_TP_2 += TP_2
    accumalating_FP_2 += FP_2
    accumalating_FN_2 += FN_2


    ## precision and recall for all the testing data
    accumalating_precision_2 = accumalating_TP_2/(accumalating_TP_2 + accumalating_FP_2)
    accumalating_recall_2 = accumalating_TP_2/(accumalating_TP_2 + accumalating_FN_2)
    print('accumulating (for all previous images) precision band 2', accumalating_precision_2)
    print('accumulating (for all previous images) recall band 2', accumalating_recall_2)
    print('accumulating (for all previous images) F1 band 2', (2*accumalating_precision_2 * accumalating_recall_2)/(accumalating_precision_2 + accumalating_recall_2))


    for key in t_1.keys():
        if key not in T:
            T[key] = []
            P[key] = []
        T[key].extend(t_1[key])
        P[key].extend(p_1[key])
    all_aps = []

    all_precisions_pts = [] 
    all_recall_pts = []

    labels, predictions = np.array(T[key]), np.array(P[key])

    pred_idx_sorted = np.argsort(predictions)
    
    labels = labels[pred_idx_sorted]
    predictions = predictions[pred_idx_sorted]
    thresholds = np.unique(predictions)

    for thresh in thresholds:

        thresh_predictions = np.where(predictions >= thresh)

        if not thresh_predictions:
            print("predictions list is empty!!")
            continue


        thresh_FP =  np.count_nonzero(labels[thresh_predictions] == 0)

        if thresh == 0.0:
            thresh_FN = len(np.where(predictions == 0)[0])
        else:
            predictions_lt_thresh = np.where(predictions < thresh)
            thresh_FN = len(np.where(labels[predictions_lt_thresh] == 1)[0])

        actual_predictions= np.where(predictions != 0)
        thresh_TP = len(np.where(labels[actual_predictions] == 1)[0])

        if not thresh_TP == 0:
            thresh_precision = thresh_TP/(thresh_TP + thresh_FP)
            thresh_recall = thresh_TP/(thresh_TP + thresh_FN)

            all_precisions_pts.append(thresh_precision)
            all_recall_pts.append(thresh_recall)
        elif thresh_TP == 0:
            continue

    all_precisions_pts.append(1)
    all_recall_pts.append(0)
    

    
    for key in T.keys():
        ap = average_precision_score(T[key], P[key])
        precision, recall, p_r_thresholds = precision_recall_curve(T[key], P[key])
        fpr, tpr, roc_thresholds = metrics.roc_curve(T[key], P[key],  pos_label=1, drop_intermediate =True)

        roc_auc = metrics.auc(fpr, tpr)
        
        # print('{} AP: {}'.format(key, ap))
        all_aps.append(ap)


    print('mAP = {}'.format(np.mean(np.array(all_aps))))




## report results :
print('-'*50)
print('-'*50)
print('-'*50)
print('Precision band 1', accumalating_precision_1)
print('Recall band 1', accumalating_recall_1)
print('F1 band 1', (2 * accumalating_precision_1 * accumalating_recall_1) / (accumalating_precision_1 + accumalating_recall_1))
print('-'*50)
print('Precision band 2', accumalating_precision_2)
print('Recall band 2', accumalating_recall_2)
print('F1 band 2', (2*accumalating_precision_2 * accumalating_recall_2)/(accumalating_precision_2 + accumalating_recall_2))
print('-'*50)
print('mAP = {}'.format(np.mean(np.array(all_aps))))
print('-'*50)
print('-'*50)
print('-'*50)



