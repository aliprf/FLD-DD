from Config import CofwConf, DatasetName
from ImageModification import ImageModification

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import random


class CofwClass:
    """PUBLIC"""

    def create_train_set(self):
        images_path, annotations_path, bboxes_path = self._load_data(CofwConf.orig_COFW_train)

        for i in tqdm(range(len(images_path))):
            img = self._load_image(images_path[i])
            bbox = self._load_bbox(bboxes_path[i])
            annotation = self._load_annotation(annotations_path[i])

            self._do_random_augment(i, img, annotation, bbox)
    print("create_train_set DONE!!")

    def create_test_set(self):
        """
        create test set from original test data
        :return:
        """
        images_path, annotations_path, bboxes_path = self._load_data(CofwConf.orig_COFW_test)

        for i in range(len(images_path)):
            img = self._load_image(images_path[i])
            bbox = self._load_bbox(bboxes_path[i])
            annotation = self._load_annotation(annotations_path[i])

            img, annotation = self._crop(img=img, annotation=annotation, bbox=bbox)

            self._save(img=img, annotation=annotation, file_name=str(i), image_save_path=CofwConf.test_image_path,
                       annotation_save_path=CofwConf.test_annotation_path)
        print("create_test_set DONE!!")

    """PRIVATE"""
    def _do_random_augment(self, index, img, annotation, _bbox):
        img_mod = ImageModification()
        xmin = _bbox[0]
        ymin = _bbox[1]
        xmax = xmin + _bbox[2]
        ymax = ymin + _bbox[2]

        '''create 4-point bounding box'''
        rand_padd = random.randint(0, 10)

        ann_xy, ann_x, ann_y = img_mod.create_landmarks(annotation, 1, 1)
        xmin = min(min(ann_x)-rand_padd, xmin)
        xmax = max(max(ann_x)+rand_padd, xmax)
        ymin = min(min(ann_y)-rand_padd, ymin)
        ymax = max(max(ann_y)+rand_padd, ymax)

        bbox_me = [xmin, ymin, xmin, ymax, xmax, ymin, xmax, ymax]

        imgs, annotations = img_mod.random_augment(index=index, img_orig=img, landmark_orig=annotation,
                                                   num_of_landmarks=CofwConf.num_of_landmarks,
                                                   augmentation_factor=CofwConf.augmentation_factor,
                                                   ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax,
                                                   ds_name=DatasetName.dsCofw, bbox_me_orig=bbox_me)
        ''''''
        '''this is the original image we save in the original path for ablation study'''
        self._save(img=imgs[0], annotation=annotations[0], file_name=str(index),
                   image_save_path=CofwConf.no_aug_train_image,
                   annotation_save_path=CofwConf.no_aug_train_annotation)

        '''this is the augmented images+original one'''
        for i in range(len(imgs)):
            self._save(img=imgs[i], annotation=annotations[i], file_name=str(index)+'_'+str(i),
                       image_save_path=CofwConf.augmented_train_image,
                       annotation_save_path=CofwConf.augmented_train_annotation)
            img_mod.test_image_print('zzz_final'+str(index)+'-'+str(i), imgs[i], annotations[i])

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
        # img_mod.test_image_print('zz'+str(annotation[0]), img, annotation)

        return img, annotation

    def _save(self, img, annotation, file_name, image_save_path, annotation_save_path):
        im = Image.fromarray(np.round(img * 255).astype(np.uint8))
        im.save(image_save_path + file_name + '.jpg')
        np.save(annotation_save_path + file_name, annotation)

    def _load_data(self, path_folder):
        """
        load all images, annotations and boundingBoxes
        :param path_folder: path to the folder
        :return: images, annotations, bboxes
        """
        images_path = []
        annotations_path = []
        bboxes_path = []

        for file in os.listdir(path_folder):
            if file.endswith(".png"):
                images_path.append(os.path.join(path_folder, file))
                annotations_path.append(os.path.join(path_folder, "an_" + str(file)[:-3] + "txt"))
                bboxes_path.append(os.path.join(path_folder, "bb_" + str(file)[:-3] + "txt"))
        return images_path, annotations_path, bboxes_path

    def _load_image(self, path):
        return np.array(Image.open(path))

    def _load_bbox(self, path):
        bbox_arr = []
        with open(path) as fp:
            line = fp.readline()
            while line:
                bbox_arr = line.strip().split('\t')
                line = fp.readline()
        assert len(bbox_arr) == 4
        return list(map(int, bbox_arr))

    def _load_annotation(self, path):
        annotation_arr = []
        with open(path) as fp:
            line = fp.readline()
            while line:
                annotation_arr = line.strip().split('\t')
                line = fp.readline()
        annotation_arr = list(map(float, annotation_arr[0:58]))
        annotation_arr_correct = []

        for i in range(0, len(annotation_arr)//2, 1):
            annotation_arr_correct.append(annotation_arr[i])
            annotation_arr_correct.append(annotation_arr[i+29])

        return annotation_arr_correct