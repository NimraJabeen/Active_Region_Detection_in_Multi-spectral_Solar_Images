from keras import backend as K

#new changes
class Config:
    def __init__(self):

        self.im_size = 600

        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False

        self.saving_path = './model/'

        self.spect_1 = '284'
        self.spect_2 = '171'
        #for the local put image folder in same directory as active region but for cluster put it inside the active refion"
        self.training_images_dir ='images'
        #self.simple_label_file_A= '/baie/nfs-cluster-1/mundus/njabeen332/Solar/UAD/labels/my_datasimple_label_TIFF_284-fixed_aug.txt'
        #self.simple_label_file_B= '/baie/nfs-cluster-1/mundus/njabeen332/Solar/UAD/labels/my_datasimple_label_TIFF_171-fixed_aug.txt'
        self.simple_label_file_A= '/baie/nfs-cluster-1/mundus/njabeen332/Solar_GPU/UAD/labels/my_datasimple_label_TIFF_284-fixed_aug.txt'
        self.simple_label_file_B= '/baie/nfs-cluster-1/mundus/njabeen332/Solar_GPU/UAD/labels/my_datasimple_label_TIFF_171-fixed_aug.txt'

        

        
        #self.training_images_dir = 'E:/Data/train/images'
        #self.simple_label_file_A = 'E:/Data/train/labels/label_195.txt'
        #self.simple_label_file_B = 'E:/Data/train/labels/label_171.txt'

        self.config_save_file = 'config.pickle'

        self.gpu_fraction = 0.65

        self.num_epochs = 6000

        self.verbose = True

        self.num_rois = 64

        self.anchor_box_scales = [32, 64, 128, 256]

        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]

        self.finetune = False

        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        self.rpn_stride = 16

        self.balanced_classes = False

        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        self.class_mapping = None





