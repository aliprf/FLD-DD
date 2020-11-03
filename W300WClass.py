from Config import W300W, DatasetName, InputDataSize
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


class W300WClass:
    """PUBLIC"""

    def create_pca_obj(self, accuracy):
        pca_utils = PCAUtility()
        pca_utils.create_pca_from_npy(annotation_path=W300W.augmented_train_annotation,
                                      pca_accuracy=accuracy, pca_file_name=DatasetName.ds300W)

    def create_train_set(self, need_pose=False, need_hm=False, accuracy=100):
        pose_detector = PoseDetector()

        imgs, annotations, bboxs = self._load_data(W300W.orig_300W_train)

        for i in tqdm(range(len(imgs))):
            self._do_random_augment(index=i, img=imgs[i], annotation=annotations[i], _bbox=bboxs[i]
                                    , need_hm=need_hm, need_pose=need_pose, pose_detector=pose_detector)
        print("create_train_set DONE!!")

    def create_test_set(self, need_pose=False, need_tf_ref=False):
        """
        create test set from original test data
        :return:
        """
        tf_utility = TfUtility()
        pose_detector = PoseDetector()
        img_mod = ImageModification()

        ds_types = ['challenging/', 'common/', 'full/']
        for ds_type in ds_types:
            imgs, annotations, bboxs = self._load_data(W300W.orig_300W_test+ds_type)

            for i in tqdm(range(len(imgs))):
                img, annotation = self._crop(img=imgs[i], annotation=annotations[i], bbox=bboxs[i])
                pose = None
                if need_pose:
                    pose = tf_utility.detect_pose([img], pose_detector)
                self._save(img=img, annotation=annotation, file_name=str(i), pose=pose,
                           image_save_path=W300W.test_image_path+ds_type,
                           annotation_save_path=W300W.test_annotation_path+ds_type,
                           pose_save_path=W300W.test_pose_path+ds_type)
                # img_mod.test_image_print('zzz_final-'+str(i), img, annotation)

        '''tf_record'''
        # if need_tf_ref:
        #     self.wflw_create_tf_record(ds_type=1, need_pose=need_pose)  # we don't need hm for test
        print("create_test_set DONE!!")

    """PRIVATE"""

    def w300w_create_tf_record(self, ds_type, need_pose, accuracy=100):
        tf_utility = TfUtility()

        if ds_type == 0:  # train
            tf_file_paths = [W300W.no_aug_train_tf_path, W300W.augmented_train_tf_path]
            img_file_paths = [W300W.no_aug_train_image, W300W.augmented_train_image]
            annotation_file_paths = [W300W.no_aug_train_annotation, W300W.augmented_train_annotation]
            pose_file_paths = [W300W.no_aug_train_pose, W300W.augmented_train_pose]
            is_test = False
        else:
            tf_file_paths = [W300W.test_tf_path]
            img_file_paths = [W300W.test_image_path]
            annotation_file_paths = [W300W.test_annotation_path]
            pose_file_paths = [W300W.test_pose_path]
            is_test = True

        tf_utility.create_tf_ref(tf_file_paths=tf_file_paths, img_file_paths=img_file_paths,
                                 annotation_file_paths=annotation_file_paths, pose_file_paths=pose_file_paths,
                                 need_pose=need_pose, accuracy=accuracy, is_test=is_test, ds_name=DatasetName.ds300W)

    def _do_random_augment(self, index, img, annotation, _bbox, need_hm, need_pose, pose_detector):
        tf_utility = TfUtility()

        img_mod = ImageModification()

        xmin = _bbox[0]
        ymin = _bbox[1]
        xmax = _bbox[6]
        ymax = _bbox[7]
        '''create 4-point bounding box'''
        bbox_me = [xmin, ymin, xmin, ymax, xmax, ymin, xmax, ymax]

        imgs, annotations = img_mod.random_augment(index=index, img_orig=img, landmark_orig=annotation,
                                                   num_of_landmarks=W300W.num_of_landmarks,
                                                   augmentation_factor=W300W.augmentation_factor,
                                                   ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax,
                                                   ds_name=DatasetName.dsWflw, bbox_me_orig=bbox_me)
        '''create pose'''
        poses = None
        if need_pose:
            poses = tf_utility.detect_pose(images=imgs, pose_detector=pose_detector)

        '''this is the original image we save in the original path for ablation study'''
        self._save(img=imgs[0], annotation=annotations[0], file_name=str(index), pose=poses[0],
                   image_save_path=W300W.no_aug_train_image,
                   annotation_save_path=W300W.no_aug_train_annotation,
                   pose_save_path=W300W.no_aug_train_pose)

        '''this is the augmented images+original one'''
        for i in range(len(imgs)):
            self._save(img=imgs[i], annotation=annotations[i], file_name=str(index) + '_' + str(i), pose=poses[i],
                       image_save_path=W300W.augmented_train_image,
                       annotation_save_path=W300W.augmented_train_annotation,
                       pose_save_path=W300W.augmented_train_pose)
            img_mod.test_image_print('zzz_final'+str(index)+'-'+str(i), imgs[i], annotations[i])

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

    def _save(self, img, annotation, pose, file_name, image_save_path, annotation_save_path, pose_save_path):
        im = Image.fromarray(np.round(img * 255).astype(np.uint8))
        im.save(image_save_path + file_name + '.jpg')
        np.save(annotation_save_path + file_name, annotation)
        if pose is not None:
            np.save(pose_save_path + file_name, pose)

    def _load_data(self, path_folder):
        """
        load all images, annotations and boundingBoxes
        :param annotation_path: path to the folder
        :return: images, annotations, bboxes
        """
        image_arr = []
        annotation_arr = []
        bbox_arr = []

        counter =0
        for file in tqdm(os.listdir(path_folder)):
            if (file.endswith(".png") or file.endswith(".jpg")) and counter < 100:
                try:
                    images_path = os.path.join(path_folder, file)
                    annotations_path = os.path.join(path_folder, str(file)[:-3] + "pts")
                    '''load data'''
                    image_arr.append(self._load_image(images_path))
                    annotation = self._load_annotation(annotations_path)
                    annotation_arr.append(annotation)
                    bbox_arr.append(self._create_bbox(annotation))
                except Exception as e:
                    print('300W: _load_data-Exception')
                counter += 1

        print('300W Loading Done')
        return image_arr, annotation_arr, bbox_arr

    def _load_image(self, path):
        return np.array(Image.open(path))

    def _create_bbox(self, annotation):
        img_mod = ImageModification()
        ann_xy, an_x, an_y = img_mod.create_landmarks(annotation, 1, 1)

        fix_padd = 15
        xmin = int(max(0, min(an_x) - fix_padd))
        ymin = int(max(0, min(an_y) - fix_padd))
        xmax = int(max(an_x) + fix_padd)
        ymax = int(max(an_y) + fix_padd)
        bbox_me = [xmin, ymin, xmin, ymax, xmax, ymin, xmax, ymax]

        return bbox_me

    def _load_annotation(self, file_name):
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

        return annotation_arr
