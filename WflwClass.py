from Config import WflwConf, DatasetName, InputDataSize
from ImageModification import ImageModification
# from pose_detection.code.PoseDetector import PoseDetector
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
                                      pca_accuracy=accuracy, pca_file_name=DatasetName.dsWflw)

    def create_train_set(self, need_pose=False, need_hm=False, accuracy=100):
        # pose_detector = PoseDetector()

        imgs, annotations, bboxs, atrs = self._load_data(WflwConf.orig_WFLW_train)

        for i in tqdm(range(len(imgs))):
            self._do_random_augment(index=i, img=imgs[i], annotation=annotations[i], _bbox=bboxs[i],
                                    atr=atrs[i], need_hm=need_hm, need_pose=need_pose)
        print("create_train_set DONE!!")

    def create_test_set(self, need_pose=False, need_tf_ref=False):
        """
        create test set from original test data
        :return:
        """
        tf_utility = TfUtility()
        # pose_detector = PoseDetector()
        img_mod = ImageModification()

        imgs, annotations, bboxs, atrs = self._load_data(WflwConf.orig_WFLW_test)

        for i in tqdm(range(len(imgs))):
            img, annotation = self._crop(img=imgs[i], annotation=annotations[i], bbox=bboxs[i])
            # pose = None
            # if need_pose:
            #     pose = tf_utility.detect_pose([img], pose_detector)
            self._save(img=img, annotation=annotation, atr=atrs[i], file_name=str(i),
                       image_save_path=WflwConf.test_image_path,
                       annotation_save_path=WflwConf.test_annotation_path, pose_save_path=WflwConf.test_pose_path,
                       atr_save_path=WflwConf.test_atr_path)

        '''tf_record'''
        # if need_tf_ref:
        #     self.wflw_create_tf_record(ds_type=1, need_pose=need_pose)  # we don't need hm for test
        print("create_test_set DONE!!")

    """PRIVATE"""

    def wflw_create_tf_record(self, need_pose, accuracy=100, ds_type=0):
        tf_utility = TfUtility()

        # if ds_type == 0:  # train
        tf_file_paths = [WflwConf.no_aug_train_tf_path, WflwConf.augmented_train_tf_path]
        img_file_paths = [WflwConf.no_aug_train_image, WflwConf.augmented_train_image]
        annotation_file_paths = [WflwConf.no_aug_train_annotation, WflwConf.augmented_train_annotation]
        pose_file_paths = [WflwConf.no_aug_train_pose, WflwConf.augmented_train_pose]
        num_train_samples = [WflwConf.num_train_samples_orig, WflwConf.num_train_samples_aug]
        num_eval_samples = [WflwConf.num_eval_samples_orig, WflwConf.num_eval_samples_aug]
        is_test = False
        # else:
        #     tf_file_paths = [WflwConf.test_tf_path]
        #     img_file_paths = [WflwConf.test_image_path]
        #     annotation_file_paths = [WflwConf.test_annotation_path]
        #     pose_file_paths = [WflwConf.test_pose_path]
        #     num_train_samples = [WflwConf.orig_number_of_test]
        #     num_eval_samples = [0]
        #     is_test = True

        tf_utility.create_tf_ref(tf_file_paths=tf_file_paths, img_file_paths=img_file_paths,
                                 annotation_file_paths=annotation_file_paths, pose_file_paths=pose_file_paths,
                                 need_pose=need_pose, accuracy=accuracy, is_test=is_test,ds_name=DatasetName.dsWflw,
                                 num_train_samples=num_train_samples, num_eval_samples=num_eval_samples)

    def _do_random_augment(self, index, img, annotation, _bbox, atr, need_hm, need_pose, pose_detector=None):
        tf_utility = TfUtility()

        img_mod = ImageModification()

        xmin = _bbox[0]
        ymin = _bbox[1]
        xmax = _bbox[6]
        ymax = _bbox[7]
        '''create 4-point bounding box'''
        rand_padd = 1
        # rand_padd = random.randint(1, 5)

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
                                                   ds_name=DatasetName.dsWflw, bbox_me_orig=bbox_me, atr=atr)
        '''create pose'''
        poses = None
        if need_pose:
            poses = tf_utility.detect_pose(images=imgs, pose_detector=pose_detector)

        '''this is the original image we save in the original path for ablation study'''
        self._save(img=imgs[0], annotation=annotations[0], file_name=str(index), atr=atr,
                   image_save_path=WflwConf.no_aug_train_image,
                   annotation_save_path=WflwConf.no_aug_train_annotation,
                   atr_save_path=WflwConf.augmented_train_tf_path,
                   pose_save_path=WflwConf.no_aug_train_pose)

        '''this is the augmented images+original one'''
        for i in range(len(imgs)):
            self._save(img=imgs[i], annotation=annotations[i], file_name=str(index) + '_' + str(i),
                       atr=atr,
                       image_save_path=WflwConf.augmented_train_image,
                       annotation_save_path=WflwConf.augmented_train_annotation,
                       atr_save_path=WflwConf.augmented_train_tf_path,
                       pose_save_path=WflwConf.augmented_train_pose)
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

        xmin = min(min(ann_x) - fix_pad, xmin)
        xmax = max(max(ann_x) + fix_pad, xmax)
        ymin = min(min(ann_y) - fix_pad, ymin)
        ymax = max(max(ann_y) + fix_pad, ymax)

        img, annotation = img_mod.crop_image_test(img, ymin, ymax, xmin, xmax, annotation)
        # img_mod.test_image_print('zz'+str(annotation[0]), img, annotation)
        img, annotation = img_mod.resize_image(img, annotation)
        # img_mod.test_image_print('zz'+str(annotation[0]), img, annotation)

        return img, annotation

    def _save(self, img, annotation, atr, file_name, image_save_path, annotation_save_path, pose_save_path,
              atr_save_path, pose=None):
        im = Image.fromarray(np.round(img * 255).astype(np.uint8))
        im.save(image_save_path + file_name + '.jpg')
        np.save(annotation_save_path + file_name, annotation)
        np.save(atr_save_path + file_name, atr)
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
            while line:  # and counter < 10:
                sys.stdout.write('\r \r line --> \033[92m' + str(counter))

                total_data = line.strip().split(' ')
                annotation_arr.append(list(map(float, total_data[0:WflwConf.num_of_landmarks * 2])))
                bbox_arr.append(self._create_bbox(
                    list(map(int, total_data[WflwConf.num_of_landmarks * 2:WflwConf.num_of_landmarks * 2 + 4]))))
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
