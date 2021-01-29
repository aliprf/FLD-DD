from Config import CofwConf, DatasetName, InputDataSize
from ImageModification import ImageModification
# from pose_detection.code.PoseDetector import PoseDetector
from pca_utility import PCAUtility
from tf_utility import TfUtility
from Evaluation import Evaluation
import os
import numpy as np
from numpy import load, save
from tqdm import tqdm
from PIL import Image
import random
from tensorflow import keras
import efficientnet.tfkeras
import tensorflow as tf
import csv


class CofwClass:
    """PUBLIC"""

    def batch_test(self, weight_files_path, csv_file_path):
        with open(csv_file_path, "w") as csv_file:
            header = ['wight_file_name', 'nme', 'fr',  'AUC']
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)

            for file in tqdm(os.listdir(weight_files_path)):
                if file.endswith(".h5"):
                    nme, fr, AUC = self.evaluate_on_cofw(model_name='---', model_file=os.path.join(weight_files_path, file),
                                                         print_result=False)
                    line = [str(file), str(nme), str(fr), str(AUC)]
                    writer.writerow(line)

    def create_pca_obj(self, accuracy):
        pca_utils = PCAUtility()
        pca_utils.create_pca_from_npy(annotation_path=CofwConf.augmented_train_annotation,
                                      pca_accuracy=accuracy, pca_file_name=DatasetName.dsCofw, normalize=False)

    def create_train_set(self, need_pose=False, need_hm=False, need_tf_ref=False, accuracy=100):
        # pose_detector = PoseDetector()
        images_path, annotations_path, bboxes_path = self._load_data(CofwConf.orig_COFW_train)

        for i in tqdm(range(len(images_path))):
            img = self._load_image(images_path[i])
            bbox = self._load_bbox(bboxes_path[i])
            annotation = self._load_annotation(annotations_path[i])

            self._do_random_augment(index=i, img=img, annotation=annotation, _bbox=bbox, need_hm=need_hm,
                                    need_pose=need_pose)

        print("create_train_set DONE!!")

    def create_test_set(self, need_pose=False, need_tf_ref=False):
        """
        create test set from original test data
        :return:
        """
        img_mod = ImageModification()
        tf_utility = TfUtility()
        # pose_detector = PoseDetector()

        images_path, annotations_path, bboxes_path = self._load_data(CofwConf.orig_COFW_test)

        for i in range(len(images_path)):
            img = self._load_image(images_path[i])
            bbox = self._load_bbox(bboxes_path[i])
            annotation = self._load_annotation(annotations_path[i])

            img, annotation = self._crop(img=img, annotation=annotation, bbox=bbox)

            # annotation = img_mod.normalize_annotations(annotation=annotation)

            # pose = None
            # if need_pose:
            #     pose = tf_utility.detect_pose([img], pose_detector)
            self._save(img=img, annotation=annotation, file_name=str(i),
                       image_save_path=CofwConf.test_image_path,
                       annotation_save_path=CofwConf.test_annotation_path, pose_save_path=CofwConf.test_pose_path)

        '''tf_record'''
        # if need_tf_ref:
        #     self.cofw_create_tf_record(ds_type=1, need_pose=need_pose)  # we don't need hm for test
        print("COFW: create_test_set DONE!!")

    def hm_evaluate_on_cofw(self, model_name, model_file):
        '''create model using the h.5 model and its wights'''
        model = tf.keras.models.load_model(model_file)
        '''load test files and categories:'''
        test_annotation_paths, test_image_paths = self._get_test_set()

        """"""
        evaluation = Evaluation(model=model, anno_paths=test_annotation_paths, img_paths=test_image_paths,
                                ds_name=DatasetName.dsCofw, ds_number_of_points=CofwConf.num_of_landmarks,
                                fr_threshold=0.1, is_normalized=True, ds_type='full', model_name=model_name)
        '''predict labels:'''
        '''predict labels:'''
        nme, fr, AUC = evaluation.predict_hm()
        # nme, fr, AUC = evaluation.predict_hm()

        nme, fr, AUC = evaluation.predict_annotation_hm()
        print('Dataset: ' + DatasetName.dsCofw
              + '{ nme: ' + str(nme) + '}\n\r'
              + '{ fr: ' + str(fr) + '}\n\r'
              + '{ AUC: ' + str(AUC) + '}\n\r'
              )
        print('=========================================')

    '''evaluate with meta data: best to worst'''

    def evaluate_on_cofw(self, model_name, model_file, print_result=True):
        '''create model using the h.5 model and its wights'''
        model = keras.models.load_model(model_file)
        '''load test files and categories:'''
        test_annotation_paths, test_image_paths = self._get_test_set()

        """"""
        evaluation = Evaluation(model_name=model_name, model=model, anno_paths=test_annotation_paths,
                                img_paths=test_image_paths,
                                ds_name=DatasetName.dsCofw, ds_number_of_points=CofwConf.num_of_landmarks,
                                fr_threshold=0.1, is_normalized=False, ds_type='full')
        '''predict labels:'''
        nme, fr, AUC = evaluation.predict_annotation()
        if print_result:
            print('Dataset: ' + DatasetName.dsCofw
                  + '{ nme: ' + str(nme) + '}\n\r'
                  + '{ fr: ' + str(fr) + '}\n\r'
                  + '{ AUC: ' + str(AUC) + '}\n\r'
                  )
            print('=========================================')
            '''evaluate with meta data: best to worst'''
        return nme, fr, AUC

    def create_point_imgpath_map(self):
        """
        only used for KD:
        """
        tf_utility = TfUtility()

        img_file_paths = [CofwConf.no_aug_train_image, CofwConf.augmented_train_image]
        annotation_file_paths = [CofwConf.no_aug_train_annotation, CofwConf.augmented_train_annotation]
        map_name = ['map_orig' + DatasetName.dsCofw, 'map_aug' + DatasetName.dsCofw]
        tf_utility.create_point_imgpath_map(img_file_paths=img_file_paths,
                                            annotation_file_paths=annotation_file_paths, map_name=map_name)

    def cofw_create_tf_record(self, need_pose, accuracy=100, ds_type=0):
        tf_utility = TfUtility()

        # if ds_type == 0:  # train
        tf_file_paths = [CofwConf.no_aug_train_tf_path, CofwConf.augmented_train_tf_path]
        img_file_paths = [CofwConf.no_aug_train_image, CofwConf.augmented_train_image]
        annotation_file_paths = [CofwConf.no_aug_train_annotation, CofwConf.augmented_train_annotation]
        pose_file_paths = [CofwConf.no_aug_train_pose, CofwConf.augmented_train_pose]
        num_train_samples = [CofwConf.num_train_samples_orig, CofwConf.num_train_samples_aug]
        num_eval_samples = [CofwConf.num_eval_samples_orig, CofwConf.num_eval_samples_aug]
        is_test = False
        # else:
        #     tf_file_paths = [CofwConf.test_tf_path]
        #     img_file_paths = [CofwConf.test_image_path]
        #     annotation_file_paths = [CofwConf.test_annotation_path]
        #     pose_file_paths = [CofwConf.test_pose_path]
        #     num_train_samples = [CofwConf.orig_number_of_test]
        #     num_eval_samples = [0]
        #     is_test = True

        tf_utility.create_tf_ref(tf_file_paths=tf_file_paths, img_file_paths=img_file_paths,
                                 annotation_file_paths=annotation_file_paths, pose_file_paths=pose_file_paths,
                                 need_pose=need_pose, accuracy=accuracy, is_test=is_test, ds_name=DatasetName.dsCofw,
                                 num_train_samples=num_train_samples, num_eval_samples=num_eval_samples)

    def create_intera_face_web_distance(self, ds_type):
        img_mod = ImageModification()
        if ds_type == 0:
            img_file_path = CofwConf.no_aug_train_image
            annotation_file_path = CofwConf.no_aug_train_annotation
        else:
            img_file_path = CofwConf.test_image_path
            annotation_file_path = CofwConf.test_annotation_path
        cofw_intera_fwd_pnt = [(0, 8)]

    def create_inter_face_web_distance(self, ds_type):
        img_mod = ImageModification()
        if ds_type == 0:
            img_file_path = CofwConf.no_aug_train_image
            annotation_file_path = CofwConf.no_aug_train_annotation
        else:
            img_file_path = CofwConf.test_image_path
            annotation_file_path = CofwConf.test_annotation_path
        '''between diiferebt elements of the face (eye and nose)'''
        cofw_inter_fwd_pnt = [(0, 8), (1, 9), (2, 10), (3, 11), (2, 3), (10, 11), (10, 21), (11, 21), (21, 24),
                              (27, 28), (8, 22), (9, 23), (22, 28), (23, 28)]
        '''between every single elements (eye points, ..)'''
        cofw_intera_fwd_pnt = [(4, 5), (6, 7), (12, 13), (14, 15), (24, 25), (26, 27), (20, 21),
                               (8, 10), (11, 9), (18, 21), (19, 21), (22, 24), (25, 26), (23, 24), (22, 27), (23, 27), ]
        img_mod.create_normalized_web_facial_distance(inter_points=cofw_inter_fwd_pnt,
                                                      intera_points=cofw_intera_fwd_pnt,
                                                      img_file_path=img_file_path,
                                                      annotation_file_path=annotation_file_path,
                                                      ds_name=DatasetName.dsCofw)

    def create_heatmap(self):
        img_mod = ImageModification()
        for i, anno_file in tqdm(enumerate(os.listdir(CofwConf.augmented_train_annotation))):
            hm = img_mod.generate_hm(width=InputDataSize.hm_size, height=InputDataSize.hm_size,
                                     landmark_path=CofwConf.augmented_train_annotation, landmark_filename=anno_file,
                                     s=CofwConf.hm_sigma, de_normalize=False)
            np.save(CofwConf.augmented_train_hm + anno_file, hm)

    """PRIVATE"""

    def _get_test_set(self):
        test_annotation_paths = []
        test_image_paths = []
        for file in tqdm(os.listdir(CofwConf.test_image_path)):
            if file.endswith(".png") or file.endswith(".jpg"):
                test_annotation_paths.append(os.path.join(CofwConf.test_annotation_path, str(file)[:-3] + "npy"))
                test_image_paths.append(os.path.join(CofwConf.test_image_path, str(file)))
        return test_annotation_paths, test_image_paths

    def _do_random_augment(self, index, img, annotation, _bbox, need_hm, need_pose, pose_detector=None):
        tf_utility = TfUtility()

        img_mod = ImageModification()
        xmin = _bbox[0]
        ymin = _bbox[1]
        xmax = xmin + _bbox[2]
        ymax = ymin + _bbox[3]

        '''create 4-point bounding box'''
        rand_padd = random.randint(0, 1)

        ann_xy, ann_x, ann_y = img_mod.create_landmarks(annotation, 1, 1)
        xmin = min(min(ann_x) - rand_padd, xmin)
        xmax = max(max(ann_x) + rand_padd, xmax)
        ymin = min(min(ann_y) - rand_padd, ymin)
        ymax = max(max(ann_y) + rand_padd, ymax)

        bbox_me = [xmin, ymin, xmin, ymax, xmax, ymin, xmax, ymax]

        imgs, annotations = img_mod.random_augment(index=index, img_orig=img, landmark_orig=annotation,
                                                   num_of_landmarks=CofwConf.num_of_landmarks,
                                                   augmentation_factor=CofwConf.augmentation_factor,
                                                   ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax,
                                                   ds_name=DatasetName.dsCofw, bbox_me_orig=bbox_me)
        '''create pose'''
        # poses = []
        # if need_pose:
        #     poses = tf_utility.detect_pose(images=imgs, pose_detector=pose_detector)

        '''this is the original image we save in the original path for ablation study'''
        self._save(img=imgs[0], annotation=annotations[0], file_name=str(index),
                   image_save_path=CofwConf.no_aug_train_image,
                   annotation_save_path=CofwConf.no_aug_train_annotation,
                   pose_save_path=CofwConf.no_aug_train_pose)

        '''this is the augmented images+original one'''
        for i in range(len(imgs)):
            self._save(img=imgs[i], annotation=annotations[i], file_name=str(index) + '_' + str(i),
                       image_save_path=CofwConf.augmented_train_image,
                       annotation_save_path=CofwConf.augmented_train_annotation,
                       pose_save_path=CofwConf.augmented_train_pose)
            # img_mod.test_image_print('zzz_final'+str(index)+'-'+str(i), imgs[i], annotations[i])
        #
        return imgs, annotations

    def _crop(self, img, annotation, bbox):
        img_mod = ImageModification()

        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]

        img, annotation = img_mod.crop_image_test(img, ymin, ymax, xmin, xmax, annotation, padding_percentage=0.05)
        # img_mod.test_image_print('zz'+str(annotation[0]), img, landmarks)
        img, annotation = img_mod.resize_image(img, annotation)
        # img_mod.test_image_print('zz'+str(annotation[0]), img, annotation)

        return img, annotation

    def _save(self, img, annotation, file_name, image_save_path, annotation_save_path, pose_save_path, pose=None):
        im = Image.fromarray(np.round(img * 255).astype(np.uint8))
        im.save(image_save_path + file_name + '.jpg')
        np.save(annotation_save_path + file_name, annotation)
        if pose is not None:
            np.save(pose_save_path + file_name, pose)

    def _load_data(self, path_folder):
        """
        load all images, annotations and boundingBoxes
        :param path_folder: path to the folder
        :return: images, annotations, bboxes
        """
        images_path = []
        annotations_path = []
        bboxes_path = []

        for file in tqdm(os.listdir(path_folder)):
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
        annotation_arr = list(map(float, annotation_arr[0:CofwConf.num_of_landmarks * 2]))
        annotation_arr_correct = []

        for i in range(0, len(annotation_arr) // 2, 1):
            annotation_arr_correct.append(np.round(annotation_arr[i], 3))
            annotation_arr_correct.append(np.round(annotation_arr[i + CofwConf.num_of_landmarks], 3))

        return annotation_arr_correct
