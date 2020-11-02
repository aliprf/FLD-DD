from Config import WflwConf, DatasetName, InputDataSize
from ImageModification import ImageModification
from pose_detection.code.PoseDetector import PoseDetector
from pca_utility import PCAUtility
from tf_utility import TfUtility

import os, sys
import numpy as np
from numpy import load, save
from tqdm import tqdm
from PIL import Image
import random


class WflwClass:
    """PUBLIC"""

    def create_pca_obj(self, accuracy):
        pca_utils = PCAUtility()
        pca_utils.create_pca_from_npy(annotation_path=WflwConf.augmented_train_annotation,
                                      pca_accuracy=accuracy, pca_file_name='wflw')

    def create_train_set(self, need_pose=False, need_hm=False, need_tf_ref=False, accuracy=100):
        images_path, annotations_path, bboxes_path = self._load_data(WflwConf.orig_WFLW_train)

        for i in tqdm(range(len(images_path))):
            img = self._load_image(images_path[i])
            bbox = self._load_bbox(bboxes_path[i])
            annotation = self._load_annotation(annotations_path[i])

            self._do_random_augment(i, img, annotation, bbox, need_hm=need_hm,
                                    need_pose=need_pose)
        '''create tf_record here: tf need to be created when all samples has been created'''
        if need_tf_ref:
            self._wflw_create_tf_record(ds_type=0, need_pose=need_pose, need_hm=need_hm, accuracy=accuracy)

        print("create_train_set DONE!!")

    def create_test_set(self, need_pose=False, need_tf_ref=False):
        """
        create test set from original test data
        :return:
        """
        tf_utility = TfUtility()
        pose_detector = PoseDetector()
        img_mod = ImageModification()

        imgs, annotations, bboxs, atrs = self._load_data(WflwConf.orig_WFLW_test)

        for i in range(len(imgs)):
            # img_mod.test_image_print('wflw_tst_' + str(i), imgs[i], annotations[i], bboxs[i])

            img, annotation = self._crop(img=imgs[i], annotation=annotations[i], bbox=bboxs[i])
            pose = None
            if need_pose:
                pose = tf_utility.detect_pose([img], pose_detector)
            self._save(img=img, annotation=annotation, file_name=str(i), pose=pose,
                       image_save_path=WflwConf.test_image_path,
                       annotation_save_path=WflwConf.test_annotation_path, pose_save_path=WflwConf.test_pose_path)

        '''tf_record'''
        if need_tf_ref:
            self._wflw_create_tf_record(ds_type=1, need_pose=need_pose)  # we don't need hm for test
        print("create_test_set DONE!!")

    """PRIVATE"""

    def _wflw_create_tf_record(self, ds_type, need_pose, need_hm=False, accuracy=100):
        tf_utility = TfUtility()

        if ds_type == 0:  # train
            tf_file_paths = [WflwConf.no_aug_train_tf_path, WflwConf.augmented_train_tf_path]
            img_file_paths = [WflwConf.no_aug_train_image, WflwConf.augmented_train_image]
            annotation_file_paths = [WflwConf.no_aug_train_annotation, WflwConf.augmented_train_annotation]
            pose_file_paths = [WflwConf.no_aug_train_pose, WflwConf.augmented_train_pose]
            is_test = False
        else:
            tf_file_paths = [WflwConf.test_tf_path]
            img_file_paths = [WflwConf.test_image_path]
            annotation_file_paths = [WflwConf.test_annotation_path]
            pose_file_paths = [WflwConf.test_pose_path]
            is_test = True

        tf_utility.create_tf_ref(tf_file_paths=tf_file_paths, img_file_paths=img_file_paths,
                                 annotation_file_paths=annotation_file_paths, pose_file_paths=pose_file_paths,
                                 need_pose=need_pose, need_hm=need_hm, accuracy=accuracy, is_test=is_test)

    def _do_random_augment(self, index, img, annotation, _bbox, need_hm, need_pose):
        tf_utility = TfUtility()

        img_mod = ImageModification()
        xmin = _bbox[0]
        ymin = _bbox[1]
        xmax = xmin + _bbox[2]
        ymax = ymin + _bbox[2]

        '''create 4-point bounding box'''
        rand_padd = random.randint(0, 10)

        ann_xy, ann_x, ann_y = img_mod.create_landmarks(annotation, 1, 1)
        xmin = min(min(ann_x) - rand_padd, xmin)
        xmax = max(max(ann_x) + rand_padd, xmax)
        ymin = min(min(ann_y) - rand_padd, ymin)
        ymax = max(max(ann_y) + rand_padd, ymax)

        bbox_me = [xmin, ymin, xmin, ymax, xmax, ymin, xmax, ymax]

        imgs, annotations = img_mod.random_augment(index=index, img_orig=img, landmark_orig=annotation,
                                                   num_of_landmarks=WflwConf.num_of_landmarks,
                                                   augmentation_factor=WflwConf.augmentation_factor,
                                                   ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax,
                                                   ds_name=DatasetName.dsCofw, bbox_me_orig=bbox_me)
        '''create pose'''
        poses = None
        if need_pose:
            poses = tf_utility.detect_pose(images=imgs)

        '''this is the original image we save in the original path for ablation study'''
        self._save(img=imgs[0], annotation=annotations[0], file_name=str(index), pose=poses[0],
                   image_save_path=WflwConf.no_aug_train_image,
                   annotation_save_path=WflwConf.no_aug_train_annotation,
                   pose_save_path=WflwConf.no_aug_train_pose)

        '''this is the augmented images+original one'''
        for i in range(len(imgs)):
            self._save(img=imgs[i], annotation=annotations[i], file_name=str(index) + '_' + str(i), pose=poses[i],
                       image_save_path=WflwConf.augmented_train_image,
                       annotation_save_path=WflwConf.augmented_train_annotation,
                       pose_save_path=WflwConf.augmented_train_pose)
            # img_mod.test_image_print('zzz_final'+str(index)+'-'+str(i), imgs[i], annotations[i])

        return imgs, annotations

    def _crop(self, img, annotation, bbox):
        img_mod = ImageModification()

        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[2]

        img, annotation = img_mod.crop_image_test(img, ymin, ymax, xmin, xmax, annotation, padding_percentage=0.05)
        # img_mod.test_image_print('zz'+str(annotation[0]), img, landmarks)
        img, annotation = img_mod.resize_image(img, annotation)
        img_mod.test_image_print('zz'+str(annotation[0]), img, annotation)

        return img, annotation

    def _save(self, img, annotation, pose, file_name, image_save_path, annotation_save_path, pose_save_path):
        im = Image.fromarray(np.round(img * 255).astype(np.uint8))
        im.save(image_save_path + file_name + '.jpg')
        np.save(annotation_save_path + file_name, annotation)
        if pose is not None:
            np.save(pose_save_path + file_name, pose)

    def _load_data(self, annotation_path):
        """
        load all images, annotations and boundingBoxes
        :param annotation_path: path to the folder
        :return: images, annotations, bboxes
        """
        image_arr = []
        annotation_arr = []
        bbox_arr = []
        atr_arr = []

        counter = 0
        with open(annotation_path) as fp:
            line = fp.readline()
            while line and counter < 10:
                sys.stdout.write('\r \r line --> \033[92m' + str(counter))

                total_data = line.strip().split(' ')
                annotation_arr.append(list(map(float, total_data[0:WflwConf.num_of_landmarks * 2])))
                bbox_arr.append(self._create_bbox(list(map(int, total_data[WflwConf.num_of_landmarks * 2:WflwConf.num_of_landmarks * 2 + 4]))))
                atr_arr.append(
                    list(map(int, total_data[WflwConf.num_of_landmarks * 2 + 4:WflwConf.num_of_landmarks * 2 + 10])))
                image_arr.append(self._load_image(WflwConf.orig_WFLW_image + total_data[-1]))
                line = fp.readline()
                counter += 1
        print('Loading Done')
        return image_arr, annotation_arr, bbox_arr, atr_arr

    def _load_image(self, path):
        return np.array(Image.open(path))

    def _create_bbox(self, _bbox):
        bbox_me = []

        xmin = _bbox[0]
        ymin = _bbox[1]
        xmax = _bbox[2]
        ymax = _bbox[3]
        bbox_me = [xmin, ymin, xmin, ymax, xmax, ymin, xmax, ymax]

        return bbox_me

    def _load_annotation(self, path):
        annotation_arr = []
        with open(path) as fp:
            line = fp.readline()
            while line:
                annotation_arr = line.strip().split('\t')
                line = fp.readline()
        annotation_arr = list(map(float, annotation_arr[0:WflwConf.num_of_landmarks * 2]))
        annotation_arr_correct = []

        for i in range(0, len(annotation_arr) // 2, 1):
            annotation_arr_correct.append(annotation_arr[i])
            annotation_arr_correct.append(annotation_arr[i + WflwConf.num_of_landmarks])

        return annotation_arr_correct
