from __future__ import division
import random
import time
import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import load_model
from keras.models import Model
from keras.utils import generic_utils
from MLMT_RCNN import config, MLMT_data_generators
from MLMT_RCNN import losses as losses_fn
from MLMT_RCNN import MLMT_ResNet as nn
from MLMT_RCNN.MLMT_simple_parser import get_data
import MLMT_RCNN.roi_helpers as roi_helpers
from MLMT_RCNN.MLMT_ResNet import load_nested_imagenet
import keras


def get_session(gpu_fraction):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def select_Pos_Neg_Sample(cfg, pos_samples, neg_samples):
    if cfg.num_rois > 1:
        if len(pos_samples) < cfg.num_rois // 2:
            selected_pos_samples = pos_samples.tolist()
        else:
            selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois // 2, replace=False).tolist()
        try:
            selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                    replace=False).tolist()
        except:
            selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                    replace=True).tolist()

        sel_samples = selected_pos_samples + selected_neg_samples
    else:
        if np.random.randint(0, 2):
            sel_samples = random.choice(neg_samples)
        else:
            sel_samples = random.choice(pos_samples)

    return sel_samples




def train_kitti():

    cfg = config.Config()

    K.set_session(get_session(cfg.gpu_fraction))

    training_images_dir = cfg.training_images_dir
    spect_1 = cfg.spect_1
    spect_2 = cfg.spect_2
    print('spects {} and {}'.format(spect_1, spect_2))


    all_images, classes_count, class_mapping = get_data(cfg.simple_label_file_A,
                                                        cfg.simple_label_file_B,
                                                        images_dir = training_images_dir,
                                                        spect_1 = spect_1,
                                                        spect_2 = spect_2,
                                                        data_set = 'train')


    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)




    print('class_mapping'.format(class_mapping))
    print('num classes (including bg) = {}'.format(len(classes_count)))



    cfg.class_mapping = class_mapping
    with open(cfg.config_save_file, 'wb') as config_f:
        pickle.dump(cfg, config_f)
        print('Config has been written to {} and can be loaded when testing.'.format(cfg.config_save_file))



    random.shuffle(all_images)
    train_imgs = [s for s in all_images if s['imageset'] == 'train']
    print('Num train samples for one spectrum is {}'.format(len(train_imgs)))



    data_gen_train = MLMT_data_generators.get_anchor_gt(train_imgs, classes_count, cfg, nn.get_img_output_length,
                                                        K.image_dim_ordering(), images_dir = training_images_dir, spect_1_name = spect_1, spect_2_name = spect_2, mode='train')


    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)

    input_img_1, input_img_2 = Input(shape=input_shape_img), Input(shape=input_shape_img)
    
    roi_input_1 = Input(shape=(None, 4))
    roi_input_2 = Input(shape=(None, 4))

    shared_layers_merged, shared_layers_x1, shared_layers_x2  = nn.mlmt_base_nn(input_img_1, input_img_2, trainable=True)

    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn = nn.rpn(shared_layers_merged, shared_layers_x1, shared_layers_x2, num_anchors)

    classifier = nn.classifier(shared_layers_x1, shared_layers_x2, roi_input_1, roi_input_2, cfg.num_rois, nb_classes=len(classes_count), trainable=True)

    model_rpn = Model([input_img_1, input_img_2], rpn[:4])
    model_classifier = Model([input_img_1, input_img_2, roi_input_1, roi_input_2], classifier)



    if cfg.finetune:
        print('loading ImageNet ResNet50 ..')
        model_classifier = load_nested_imagenet(model_classifier)
        model_rpn = load_nested_imagenet(model_rpn)


    optimizer_rpn = Adam(lr=2e-5)
    optimizer_classifier = Adam(lr=2e-5)

    model_rpn.compile(optimizer=optimizer_rpn,
                        loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors),
                              losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                               loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(len(classes_count) - 1),
                                     losses_fn.class_loss_cls, losses_fn.class_loss_regr(len(classes_count) - 1)])

    num_epochs = int(cfg.num_epochs)
    epoch_length = 30

    iter_num = 0

    RPN_overlap_thresh = 0.9
    RPN_max_Bbox = 300

    best_loss = np.Inf
    best_loss_Det = np.Inf
    best_loss_RPN = np.Inf

    losses = np.zeros((epoch_length, 9))

    rpn_1_accuracy_rpn_monitor = []
    rpn_1_accuracy_for_epoch = []
    rpn_2_accuracy_rpn_monitor = []
    rpn_2_accuracy_for_epoch = []

    saving_dir = cfg.saving_path + '/' + cfg.spect_1 + '-and-' + cfg.spect_2 + '/'
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    print('Starting training')
    start_time = time.time()

    for epoch_num in range(num_epochs):
        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
        while True:
            try:
                if len(rpn_1_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
                    mean_overlapping_bboxes_1 = float(sum(rpn_1_accuracy_rpn_monitor)) / len(rpn_1_accuracy_rpn_monitor)
                    rpn_1_accuracy_rpn_monitor = []
                    print('Average number of overlapping bounding boxes from RPN for image 1= {} for {} previous iterations'.format(mean_overlapping_bboxes_1, epoch_length))
                    if mean_overlapping_bboxes_1 == 0:
                        print('RPN is not producing bounding boxes that overlap for the ground truth boxes (spect 1). Check RPN settings or keep training.')
                if len(rpn_2_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
                    mean_overlapping_bboxes_2 = float(sum(rpn_2_accuracy_rpn_monitor)) / len(rpn_2_accuracy_rpn_monitor)
                    rpn_2_accuracy_rpn_monitor = []
                    print('Average number of overlapping bounding boxes from RPN for image 2 = {} for {} previous iterations'.format(mean_overlapping_bboxes_2, epoch_length))
                    if mean_overlapping_bboxes_2 == 0:
                        print('RPN is not producing bounding boxes that overlap for the ground truth boxes (spect 2). Check RPN settings or keep training.')


                X_1, X_2, Y, img_data = next(data_gen_train)

                loss_rpn = model_rpn.train_on_batch([X_1, X_2], Y)

                P_rpn = model_rpn.predict_on_batch([X_1, X_2])

                result_1 = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
                                                  overlap_thresh = RPN_overlap_thresh,
                                                  max_boxes = RPN_max_Bbox)
                result_2 = roi_helpers.rpn_to_roi(P_rpn[2], P_rpn[3], cfg, K.image_dim_ordering(), use_regr=True,
                                                  overlap_thresh = RPN_overlap_thresh,
                                                  max_boxes = RPN_max_Bbox)


                X2_1, Y1_1, Y2_1, IouS_1 = roi_helpers.calc_iou(result_1, img_data, cfg, class_mapping)
                X2_2, Y1_2, Y2_2, IouS_2 = roi_helpers.calc_iou(result_2, img_data, cfg, class_mapping)


                if X2_1 is None:
                    rpn_1_accuracy_rpn_monitor.append(0)
                    rpn_1_accuracy_for_epoch.append(0)
                    continue
                if X2_2 is None:
                    rpn_2_accuracy_rpn_monitor.append(0)
                    rpn_2_accuracy_for_epoch.append(0)
                    continue

                neg_samples_1 = np.where(Y1_1[0, :, -1] == 1)
                pos_samples_1 = np.where(Y1_1[0, :, -1] == 0)

                if len(neg_samples_1) > 0:
                    neg_samples_1 = neg_samples_1[0]
                else:
                    neg_samples_1 = []

                if len(pos_samples_1) > 0:
                    pos_samples_1 = pos_samples_1[0]
                else:
                    pos_samples_1 = []


                neg_samples_2 = np.where(Y1_2[0, :, -1] == 1)
                pos_samples_2 = np.where(Y1_2[0, :, -1] == 0)
                if len(neg_samples_2) > 0:
                    neg_samples_2 = neg_samples_2[0]
                else:
                    neg_samples_2 = []

                if len(pos_samples_2) > 0:
                    pos_samples_2 = pos_samples_2[0]
                else:
                    pos_samples_2 = []


                rpn_1_accuracy_rpn_monitor.append(len(pos_samples_1))
                rpn_1_accuracy_for_epoch.append((len(pos_samples_1)))
                rpn_2_accuracy_rpn_monitor.append(len(pos_samples_2))
                rpn_2_accuracy_for_epoch.append((len(pos_samples_2)))


                sel_samples_1 = select_Pos_Neg_Sample(cfg, pos_samples_1, neg_samples_1)
                sel_samples_2 = select_Pos_Neg_Sample(cfg, pos_samples_2, neg_samples_2)


                loss_class = model_classifier.train_on_batch([X_1, X_2,
                                                              X2_1[:, sel_samples_1, :], X2_2[:, sel_samples_2, :]],
                                                              [Y1_1[:, sel_samples_1, :], Y2_1[:, sel_samples_1, :],
                                                              Y1_2[:, sel_samples_2, :], Y2_2[:, sel_samples_2, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]
                losses[iter_num, 2] = loss_rpn[3]
                losses[iter_num, 3] = loss_rpn[4]
                losses[iter_num, 4] = loss_class[1]
                losses[iter_num, 5] = loss_class[2]
                losses[iter_num, 6] = loss_class[3]
                losses[iter_num, 7] = loss_class[4]

                iter_num += 1

                progbar.update(iter_num,
                               [('rpn_cls_1', np.mean(losses[:iter_num, 0])), ('rpn_regr_1', np.mean(losses[:iter_num, 1])),
                                ('rpn_cls_2', np.mean(losses[:iter_num, 2])), ('rpn_regr_2', np.mean(losses[:iter_num, 3])),
                                ('detector_cls_1', np.mean(losses[:iter_num, 4])), ('detector_regr_1', np.mean(losses[:iter_num, 5])),
                                ('detector_cls_2', np.mean(losses[:iter_num, 6])), ('detector_regr_2', np.mean(losses[:iter_num, 7]))])


                if iter_num == epoch_length:
                    loss_rpn_cls_1 = np.mean(losses[:, 0])
                    loss_rpn_regr_1 = np.mean(losses[:, 1])
                    loss_rpn_cls_2 = np.mean(losses[:, 2])
                    loss_rpn_regr_2 = np.mean(losses[:, 3])
                    loss_class_cls_1 = np.mean(losses[:, 4])
                    loss_class_regr_1 = np.mean(losses[:, 5])
                    loss_class_cls_2 = np.mean(losses[:, 6])
                    loss_class_regr_2 = np.mean(losses[:, 7])

                    mean_overlapping_bboxes_1 = float(sum(rpn_1_accuracy_for_epoch)) / len(rpn_1_accuracy_for_epoch)
                    rpn_1_accuracy_for_epoch = []
                    mean_overlapping_bboxes_2 = float(sum(rpn_2_accuracy_for_epoch)) / len(rpn_2_accuracy_for_epoch)
                    rpn_2_accuracy_for_epoch = []

                    if cfg.verbose:
                        print('Mean number of bounding boxes from RPN (Band 1) overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes_1))
                        print('Mean number of bounding boxes from RPN (Band 2) overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes_2))
                        print('Loss RPN classifier_1: {}'.format(loss_rpn_cls_1))
                        print('Loss RPN regression_1: {}'.format(loss_rpn_regr_1))
                        print('Loss RPN classifier_2: {}'.format(loss_rpn_cls_2))
                        print('Loss RPN regression_2: {}'.format(loss_rpn_regr_2))
                        print('Loss Detector classifier_1: {}'.format(loss_class_cls_1))
                        print('Loss Detector regression_1: {}'.format(loss_class_regr_1))
                        print('Loss Detector classifier_2: {}'.format(loss_class_cls_2))
                        print('Loss Detector regression_2: {}'.format(loss_class_regr_2))

                        print('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss_RPN = loss_rpn_cls_1 + loss_rpn_regr_1 + loss_rpn_cls_2 + loss_rpn_regr_2
                    curr_loss_Det = loss_class_cls_1 + loss_class_regr_1 + loss_class_cls_2 + loss_class_regr_2
                    curr_loss = curr_loss_RPN + curr_loss_Det

                    iter_num = 0
                    start_time = time.time()

                    model_info = str(epoch_num+1)+'_TL_'+str(round(curr_loss_RPN, 4))+'_R1_'+str(round(loss_rpn_regr_1, 4))+'_C1_'+str(round(loss_rpn_cls_1, 4))+'_R2_'+str(round(loss_rpn_regr_2, 4))+'_C2_'+str(round(loss_rpn_cls_2, 4))
                    model_saving_path = saving_dir + model_info


                    if curr_loss_RPN < best_loss_RPN :
                        if cfg.verbose:
                            print('RPN loss decreased, saving weights.')
                        best_loss_RPN = curr_loss_RPN
                        model_rpn.save_weights(model_saving_path + '_RPN.h5')
                    if curr_loss_Det < best_loss_Det:
                        if cfg.verbose:
                            print('Det loss decreased, saving weights.')
                        best_loss_Det = curr_loss_Det
                        model_classifier.save_weights(model_saving_path + '_DET.h5')
                    if curr_loss < best_loss :
                        if cfg.verbose:
                            print('total loss decreased. saving models.')
                        best_loss = curr_loss
                        model_rpn.save(model_saving_path + '_RPN-M.h5')
                        model_classifier.save(model_saving_path + '_DET-M.h5')
                        
                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                continue
    print('Training complete, exiting.')


if __name__ == '__main__':
    train_kitti()
