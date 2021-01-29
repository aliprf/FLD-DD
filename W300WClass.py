from Config import W300WConf, DatasetName, InputDataSize
from ImageModification import ImageModification
# from pose_detection.code.PoseDetector import PoseDetector
from pca_utility import PCAUtility
from tf_utility import TfUtility
from Evaluation import Evaluation
import os, sys
import numpy as np
from numpy import load, save
from tqdm import tqdm
from PIL import Image
import random
import tensorflow as tf
import efficientnet.tfkeras
import matplotlib.pyplot as plt
import csv


class W300WClass:
    """PUBLIC"""

    def create_pca_obj(self, accuracy, normalize):
        pca_utils = PCAUtility()
        pca_utils.create_pca_from_npy(annotation_path=W300WConf.augmented_train_annotation,
                                      pca_accuracy=accuracy, pca_file_name='ibug',  # pca_file_name=DatasetName.ds300W,
                                      normalize=normalize)

    def create_mean_face(self):
        img_mod = ImageModification()
        anno_arr = []
        for file in tqdm(sorted(os.listdir(W300WConf.no_aug_train_annotation))):
            if file.endswith(".npy"):
                anno_path = os.path.join(W300WConf.no_aug_train_annotation, file)
                annotation = load(anno_path)
                anno_arr.append(annotation)

        mean_lbl_arr = np.mean(anno_arr, axis=0)
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = img_mod.create_landmarks(mean_lbl_arr, 1, 1)
        img_mod.test_image_print('mean_300w', np.ones([224, 224, 3]), landmark_arr_xy)

    def create_train_set(self, need_pose=False, need_hm=False, accuracy=100):
        # pose_detector = PoseDetector()
        try:
            i = 0
            for file in tqdm(os.listdir(W300WConf.orig_300W_train)):
                if file.endswith(".png") or file.endswith(".jpg"):
                    i += 1
                    imgs, annotations, bboxs = self._load_data_train(file, W300WConf.orig_300W_train)
                    if imgs is not None:
                        self._do_random_augment(index=i, img=imgs[0], annotation=annotations[0], _bbox=bboxs[0]
                                                , need_hm=need_hm, need_pose=need_pose)
                i += 1
        except Exception as e:
            print(str(e))

        print("create_train_set DONE!!")

    def create_heatmap(self):
        img_mod = ImageModification()
        for i, anno_file in tqdm(enumerate(os.listdir(W300WConf.augmented_train_annotation))):
            hm = img_mod.generate_hm(width=InputDataSize.hm_size, height=InputDataSize.hm_size,
                                     landmark_path=W300WConf.augmented_train_annotation, landmark_filename=anno_file,
                                     s=W300WConf.hm_sigma, de_normalize=False)
            np.save(W300WConf.augmented_train_hm + anno_file, hm)
            # img_mod.print_image_arr_heat(k=i, image=hm, print_single=False)
            # img_mod.print_heatmap_distribution(k=i, image=hm)

    def batch_test(self, weight_files_path, csv_file_path):
        with open(csv_file_path, "w") as csv_file:
            header = ['wight_file_name', 'nme_f', 'nme_ch', 'nme_co', 'fr_f', 'fr_ch', 'fr_co', 'AUC_f', 'AUC_ch',
                      'AUC_co']
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)

            for file in tqdm(os.listdir(weight_files_path)):
                if file.endswith(".h5"):
                    nme_arr, fr_arr, AUC_arr = self.evaluate_on_300w(model_name='---',
                                                                     model_file=os.path.join(weight_files_path, file),
                                                                     print_result=False)
                    line = [str(file)] + nme_arr + fr_arr + AUC_arr
                    writer.writerow(line)

    def create_test_set(self, need_pose=False, need_tf_ref=False):
        """
        create test set from original test data
        :return:
        """
        # tf_utility = TfUtility()
        # pose_detector = PoseDetector()
        img_mod = ImageModification()

        ds_types = ['challenging/', 'common/', 'full/']
        for ds_type in ds_types:
            imgs, annotations, bboxs = self._load_data(W300WConf.orig_300W_test + ds_type)

            for i in tqdm(range(len(imgs))):
                img, annotation = self._crop(img=imgs[i], annotation=annotations[i], bbox=bboxs[i])

                # annotation = img_mod.normalize_annotations(annotation=annotation)

                # pose = None
                # if need_pose:
                #     pose = tf_utility.detect_pose([img], pose_detector)
                self._save(img=img, annotation=annotation, file_name=str(i),
                           image_save_path=W300WConf.test_image_path + ds_type,
                           annotation_save_path=W300WConf.test_annotation_path + ds_type,
                           pose_save_path=W300WConf.test_pose_path + ds_type)
                # img_mod.test_image_print('zzz_final-'+str(i), img, annotation)

        '''tf_record'''
        # if need_tf_ref:
        #     self.wflw_create_tf_record(ds_type=1, need_pose=need_pose)  # we don't need hm for test
        print("300W -> create_test_set DONE!!")

    def create_point_imgpath_map(self):
        """
        only used for KD:
        """
        tf_utility = TfUtility()

        img_file_paths = [W300WConf.no_aug_train_image, W300WConf.augmented_train_image]
        annotation_file_paths = [W300WConf.no_aug_train_annotation, W300WConf.augmented_train_annotation]
        map_name = ['map_orig' + DatasetName.ds300W, 'map_aug' + DatasetName.ds300W]
        tf_utility.create_point_imgpath_map(img_file_paths=img_file_paths,
                                            annotation_file_paths=annotation_file_paths, map_name=map_name)

    def hm_evaluate_on_300w(self, model_name, model_file):
        '''create model using the h.5 model and its wights'''
        model = tf.keras.models.load_model(model_file)
        '''load test files and categories:'''
        ds_types = ['common/', 'challenging/', 'full/']
        for ds_type in ds_types:
            test_annotation_paths, test_image_paths = self._get_test_set(ds_type)

            """"""
            evaluation = Evaluation(model=model, anno_paths=test_annotation_paths, img_paths=test_image_paths,
                                    ds_name=DatasetName.ds300W, ds_number_of_points=W300WConf.num_of_landmarks,
                                    fr_threshold=0.1, is_normalized=True, ds_type=ds_type, model_name=model_name)
            '''predict labels:'''
            '''predict labels:'''
            # nme, fr, AUC = evaluation.predict_hm()
            # nme, fr, AUC = evaluation.predict_hm()

            nme, fr, AUC = evaluation.predict_annotation_hm()
            print('Dataset: ' + DatasetName.ds300W
                  + '{ ds_type: ' + ds_type + '} \n\r'
                  + '{ nme: ' + str(nme) + '}\n\r'
                  + '{ fr: ' + str(fr) + '}\n\r'
                  + '{ AUC: ' + str(AUC) + '}\n\r'
                  )
            print('=========================================')
        '''evaluate with meta data: best to worst'''

    def evaluate_on_300w(self, model_name, model_file, print_result=True):
        '''create model using the h.5 model and its wights'''
        model = tf.keras.models.load_model(model_file)
        '''load test files and categories:'''
        ds_types = ['full/', 'challenging/', 'common/']

        nme_arr = []
        fr_arr = []
        AUC_arr = []
        for ds_type in ds_types:
            test_annotation_paths, test_image_paths = self._get_test_set(ds_type)

            """"""
            evaluation = Evaluation(model=model, anno_paths=test_annotation_paths, img_paths=test_image_paths,
                                    ds_name=DatasetName.ds300W, ds_number_of_points=W300WConf.num_of_landmarks,
                                    fr_threshold=0.1, is_normalized=True, ds_type=ds_type, model_name=model_name)
            '''predict labels:'''
            '''predict labels:'''
            # nme, fr, AUC = evaluation.predict_hm()

            nme, fr, AUC = evaluation.predict_annotation()
            nme_arr.append(nme)
            fr_arr.append(fr)
            AUC_arr.append(AUC)
            if print_result:
                print('Dataset: ' + DatasetName.ds300W
                      + '{ ds_type: ' + ds_type + '} \n\r'
                      + '{ nme: ' + str(nme) + '}\n\r'
                      + '{ fr: ' + str(fr) + '}\n\r'
                      + '{ AUC: ' + str(AUC) + '}\n\r'
                      )
                print('=========================================')
        '''evaluate with meta data: best to worst'''
        return nme_arr, fr_arr, AUC_arr

    def create_sample(self, ds_type):
        img_mod = ImageModification()
        model = tf.keras.models.load_model('./models/300w/KD_main/ds_300w_mn_base.h5')
        if ds_type == 0:
            img_file_path = W300WConf.no_aug_train_image
            annotation_file_path = W300WConf.no_aug_train_annotation
        else:
            img_file_path = W300WConf.test_image_path + 'full'
            annotation_file_path = W300WConf.test_annotation_path + 'full'

        for i, file in tqdm(enumerate(sorted(os.listdir(annotation_file_path)))):
            if file.endswith(".npy"):
                anno_GT = np.load(os.path.join(annotation_file_path, str(file)))
                img_adrs = os.path.join(img_file_path, str(file)[:-3] + "jpg")
                img = np.expand_dims(np.array(Image.open(img_adrs)) / 255.0, axis=0)
                anno_Pre = model.predict(img)[0]
                anno_Pre_asm = img_mod.get_asm(input=anno_Pre, dataset_name='300W', accuracy=80)
                anno_Pre = img_mod.de_normalized(annotation_norm=anno_Pre)
                anno_Pre_asm = img_mod.de_normalized(annotation_norm=anno_Pre_asm)

                img_mod.test_image_print(img_name='z_' + str(i) + '_pr' + str(i) + '__', img=np.ones([224, 224, 3]),
                                         landmarks=anno_Pre)
                img_mod.test_image_print(img_name='z_' + str(i) + '_gt' + str(i) + '__',
                                         img=np.array(Image.open(img_adrs)) / 255.0,
                                         landmarks=anno_GT)
                img_mod.test_image_print(img_name='z_' + str(i) + '_gt' + str(i) + '_',
                                         img=np.ones([224, 224, 3]),
                                         landmarks=anno_GT)
                img_mod.test_image_print(img_name='z_' + str(i) + '_pr_asm' + str(i) + '__', img=np.ones([224, 224, 3]),
                                         landmarks=anno_Pre_asm)

    def create_inter_face_web_distance(self, ds_type):
        img_mod = ImageModification()
        if ds_type == 0:
            img_file_path = W300WConf.no_aug_train_image
            annotation_file_path = W300WConf.no_aug_train_annotation
        else:
            img_file_path = W300WConf.test_image_path + 'full'
            annotation_file_path = W300WConf.test_annotation_path + 'full'
        w300w_inter_fwd_pnt = [(0, 3), (0, 17), (0, 36), (3, 36), (3, 48), (3, 8), (8, 57), (8, 13), (13, 54), (13, 16),
                               (13, 45), (16, 45), (16, 26), (51, 33), (39, 33), (42, 33), (39, 42), (21, 39), (22, 42),
                               (21, 22), (26, 45), (17, 36)]
        w300w_intra_fwd_pnt = [(37, 41), (38, 40,), (43, 47), (44, 46), (30, 33), (31, 33), (35, 33), (57, 66),
                               (51, 62), (62, 66), (27, 31), (27, 35), (36, 39), (42, 45), (17, 21), (22, 26), (19, 17),
                               (19, 21),
                               (24, 22), (24, 26), (0, 1), (0, 2), (4, 3), (4, 5), (8, 6), (8, 7), (8, 9), (8, 10),
                               (12, 11), (12, 13), (16, 15), (16, 14), (48, 57), (48, 51), (54, 51), (54, 57)]
        img_mod.create_normalized_web_facial_distance(inter_points=w300w_inter_fwd_pnt,
                                                      intera_points=w300w_intra_fwd_pnt,
                                                      annotation_file_path=annotation_file_path,
                                                      ds_name=DatasetName.ds300W, img_file_path=img_file_path)

    """PRIVATE"""

    def _get_test_set(self, ds_type):
        test_annotation_paths = []
        test_image_paths = []
        for file in tqdm(os.listdir(W300WConf.test_image_path + ds_type)):
            if file.endswith(".png") or file.endswith(".jpg"):
                test_annotation_paths.append(
                    os.path.join(W300WConf.test_annotation_path + ds_type, str(file)[:-3] + "npy"))
                test_image_paths.append(os.path.join(W300WConf.test_image_path + ds_type, str(file)))
        return test_annotation_paths, test_image_paths

    def w300w_create_tf_record(self, need_pose, accuracy=100, ds_type=0):
        tf_utility = TfUtility()

        # if ds_type == 0:  # train
        tf_file_paths = [W300WConf.no_aug_train_tf_path, W300WConf.augmented_train_tf_path]
        img_file_paths = [W300WConf.no_aug_train_image, W300WConf.augmented_train_image]
        annotation_file_paths = [W300WConf.no_aug_train_annotation, W300WConf.augmented_train_annotation]
        pose_file_paths = [W300WConf.no_aug_train_pose, W300WConf.augmented_train_pose]
        num_train_samples = [W300WConf.num_train_samples_orig, W300WConf.num_train_samples_aug]
        num_eval_samples = [W300WConf.num_eval_samples_orig, W300WConf.num_eval_samples_aug]
        is_test = False
        # else:
        #     tf_file_paths = [W300W.test_tf_path]
        #     img_file_paths = [W300W.test_image_path]
        #     annotation_file_paths = [W300W.test_annotation_path]
        #     pose_file_paths = [W300W.test_pose_path]
        #     num_train_samples = [W300W.orig_number_of_test]
        #     num_eval_samples = [0]
        #     is_test = True

        tf_utility.create_tf_ref(tf_file_paths=tf_file_paths, img_file_paths=img_file_paths,
                                 annotation_file_paths=annotation_file_paths, pose_file_paths=pose_file_paths,
                                 need_pose=need_pose, accuracy=accuracy, is_test=is_test, ds_name=DatasetName.ds300W,
                                 num_train_samples=num_train_samples, num_eval_samples=num_eval_samples)

    def _do_random_augment(self, index, img, annotation, _bbox, need_hm, need_pose, pose_detector=None):
        tf_utility = TfUtility()

        img_mod = ImageModification()

        xmin = _bbox[0]
        ymin = _bbox[1]
        xmax = _bbox[6]
        ymax = _bbox[7]
        '''create 4-point bounding box'''
        bbox_me = [xmin, ymin, xmin, ymax, xmax, ymin, xmax, ymax]

        imgs, annotations = img_mod.random_augment(index=index, img_orig=img, landmark_orig=annotation,
                                                   num_of_landmarks=W300WConf.num_of_landmarks,
                                                   augmentation_factor=W300WConf.augmentation_factor,
                                                   ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax,
                                                   ds_name=DatasetName.ds300W, bbox_me_orig=bbox_me)

        '''create pose'''
        # poses = None
        # if need_pose:
        #     poses = tf_utility.detect_pose(images=imgs, pose_detector=pose_detector)

        '''this is the original image we save in the original path for ablation study'''
        if imgs is not None:
            self._save(img=imgs[0], annotation=annotations[0], file_name=str(index),
                       image_save_path=W300WConf.no_aug_train_image,
                       annotation_save_path=W300WConf.no_aug_train_annotation,
                       pose_save_path=W300WConf.no_aug_train_pose)

            '''this is the augmented images+original one'''
            for i in range(len(imgs)):
                self._save(img=imgs[i], annotation=annotations[i], file_name=str(index) + '_' + str(i),
                           image_save_path=W300WConf.augmented_train_image,
                           annotation_save_path=W300WConf.augmented_train_annotation,
                           pose_save_path=W300WConf.augmented_train_pose)
                # img_mod.test_image_print('zzz_final'+str(index)+'-'+str(i), imgs[i], annotations[i])

        return imgs, annotations

    def _crop(self, img, annotation, bbox):
        img_mod = ImageModification()
        ann_xy, ann_x, ann_y = img_mod.create_landmarks(annotation, 1, 1)
        fix_pad = 10
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[6]
        ymax = bbox[7]

        img, annotation = img_mod.crop_image_test(img, ymin, ymax, xmin, xmax, annotation)
        # img_mod.test_image_print('zz'+str(annotation[0]), img, annotation)
        img, annotation = img_mod.resize_image(img, annotation)
        # img_mod.test_image_print('zz'+str(annotation[0]), img, annotation)

        return img, annotation

    def _save(self, img, annotation, file_name, image_save_path, annotation_save_path, pose_save_path, pose=None):
        im = Image.fromarray(np.round(img * 255).astype(np.uint8))
        im.save(image_save_path + file_name + '.jpg')
        np.save(annotation_save_path + file_name, annotation)
        if pose is not None:
            np.save(pose_save_path + file_name, pose)

    def _load_data_train(self, file, path_folder):
        """
        load all images, annotations and boundingBoxes
        :param annotation_path: path to the folder
        :return: images, annotations, bboxes
        """
        image_arr = []
        annotation_arr = []
        bbox_arr = []
        try:
            images_path = os.path.join(path_folder, file)
            annotations_path = os.path.join(path_folder, str(file)[:-3] + "pts")
            '''load data'''
            image_arr.append(self._load_image(images_path))
            annotation = self._load_annotation(annotations_path)
            annotation_arr.append(annotation)
            bbox_arr.append(self._create_bbox(annotation))
        except Exception as e:
            print('300W: _load_data-Exception' + str(e))
            return None, 0, 0

        # print('300W Loading Done')
        return image_arr, annotation_arr, bbox_arr

    def _load_data(self, path_folder):
        """
        load all images, annotations and boundingBoxes
        :param annotation_path: path to the folder
        :return: images, annotations, bboxes
        """
        print('Loading 300W train set')
        image_arr = []
        annotation_arr = []
        bbox_arr = []

        counter = 0
        for file in tqdm(sorted(os.listdir(path_folder))):
            if (file.endswith(".png") or file.endswith(".jpg")):  # and counter < 10:
                try:
                    images_path = os.path.join(path_folder, file)
                    annotations_path = os.path.join(path_folder, str(file)[:-3] + "pts")
                    '''load data'''
                    image_arr.append(self._load_image(images_path))
                    annotation = self._load_annotation(annotations_path)
                    annotation_arr.append(annotation)
                    bbox_arr.append(self._create_bbox(annotation))
                except Exception as e:
                    print('300W: _load_data-Exception' + str(e))
                counter += 1

        print('300W Loading Done')
        return image_arr, annotation_arr, bbox_arr

    def _load_image(self, path):
        return np.array(Image.open(path))

    def _create_bbox(self, annotation):
        img_mod = ImageModification()
        ann_xy, an_x, an_y = img_mod.create_landmarks(annotation, 1, 1)

        fix_padd = 5
        xmin = int(max(0, min(an_x) - fix_padd))
        ymin = int(max(0, min(an_y) - fix_padd))
        xmax = int(max(an_x) + fix_padd)
        ymax = int(max(an_y) + fix_padd)
        bbox_me = [xmin, ymin, xmin, ymax, xmax, ymin, xmax, ymax]

        return bbox_me

    def _load_annotation(self, file_name):
        try:
            annotation_arr = []

            with open(file_name) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    if 3 < cnt < 72:
                        x_y_pnt = line.strip()
                        x = float(x_y_pnt.split(" ")[0])
                        y = float(x_y_pnt.split(" ")[1])
                        annotation_arr.append(x)
                        annotation_arr.append(y)
                    line = fp.readline()
                    cnt += 1

            return np.round(annotation_arr, 3)
        except Exception as e:
            print(str(e))
            pass
