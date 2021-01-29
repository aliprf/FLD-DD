from Config import WflwConf, DatasetName, InputDataSize
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
import csv

class WflwClass:
    """PUBLIC"""

    def create_pca_obj(self, accuracy, normalize):
        pca_utils = PCAUtility()
        pca_utils.create_pca_from_npy(annotation_path=WflwConf.augmented_train_annotation,
                                      pca_accuracy=accuracy, pca_file_name=DatasetName.dsWflw, normalize=normalize)

    def create_train_set(self, need_pose=False, need_hm=False, accuracy=100):
        # pose_detector = PoseDetector()

        imgs, annotations, bboxs, atrs = self._load_data(WflwConf.orig_WFLW_train)

        for i in tqdm(range(len(imgs))):
            self._do_random_augment(index=i, img=imgs[i], annotation=annotations[i], _bbox=bboxs[i],
                                    atr=atrs[i], need_hm=need_hm, need_pose=need_pose)
        print("create_train_set DONE!!")

    def batch_test(self, weight_files_path, csv_file_path):
        with open(csv_file_path, "w") as csv_file:

            header = ['wight_file_name'] + ['pose', 'expression', 'illumination', 'makeup', 'occlusion', 'blur', 'full']
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)

            for file in tqdm(os.listdir(weight_files_path)):
                if file.endswith(".h5"):
                    nme_arr, fr_arr, AUC_arr = self.evaluate_on_wflw(model_name='---',
                                                                     model_file=os.path.join(weight_files_path, file),
                                                                     print_result=False)
                    line = [str(file)] + nme_arr + fr_arr + AUC_arr
                    writer.writerow(line)


    def create_test_set(self, need_pose=False, need_tf_ref=False):
        """
        create test set from original test data
        :return:
        """
        tf_utility = TfUtility()
        # pose_detector = PoseDetector()
        img_mod = ImageModification()

        # subsets = ['list_98pt_test_largepose.txt', 'list_98pt_test.txt', 'list_98pt_test_blur.txt',
        #            'list_98pt_test_expression.txt', 'list_98pt_test_illumination.txt',
        #            'list_98pt_test_makeup.txt', 'list_98pt_test_occlusion.txt']

        ds_types = ['pose/', 'expression/', 'illumination/', 'makeup/', 'occlusion/', 'blur/', 'full/']

        imgs, annotations, bboxs, atrs = self._load_data(WflwConf.orig_WFLW_test)
        for i in tqdm(range(len(imgs))):
            img, annotation = self._crop(img=imgs[i], annotation=annotations[i], bbox=bboxs[i])

            atr_index = []
            if 1 in list(atrs[i]):
                # atr_index = list(atrs[i]).index(1)
                for kk in range(len(atrs[i])):
                    if atrs[i][kk] != 0:
                        atr_index.append(kk)
            # pose = None
            # if need_pose:
            #     pose = tf_utility.detect_pose([img], pose_detector)
            self._save(img=img, annotation=annotation, atr=atrs[i], file_name=str(i),
                       image_save_path=WflwConf.test_image_path + ds_types[6],
                       annotation_save_path=WflwConf.test_annotation_path + ds_types[6],
                       pose_save_path=WflwConf.test_pose_path + ds_types[6],
                       atr_save_path=WflwConf.test_atr_path + ds_types[6])
            for kk in range(len(atr_index)):
                self._save(img=img, annotation=annotation, atr=atrs[i], file_name=str(i),
                           image_save_path=WflwConf.test_image_path + ds_types[atr_index[kk]],
                           annotation_save_path=WflwConf.test_annotation_path + ds_types[atr_index[kk]],
                           pose_save_path=WflwConf.test_pose_path + ds_types[atr_index[kk]],
                           atr_save_path=WflwConf.test_atr_path + ds_types[atr_index[kk]])

        # imgs, annotations, bboxs, atrs = self._load_data(WflwConf.orig_WFLW_test)
        # for i in tqdm(range(len(imgs))):
        #     img, annotation = self._crop(img=imgs[i], annotation=annotations[i], bbox=bboxs[i])
        #
        #     atr_index = []
        #     if 1 in list(atrs[i]):
        #         # atr_index = list(atrs[i]).index(1)
        #         for kk in range(len(atrs[i])):
        #             if atrs[i][kk] != 0:
        #                 atr_index.append(kk)
        #     # pose = None
        #     # if need_pose:
        #     #     pose = tf_utility.detect_pose([img], pose_detector)
        #     self._save(img=img, annotation=annotation, atr=atrs[i], file_name=str(i),
        #                image_save_path=WflwConf.test_image_path + ds_types[0],
        #                annotation_save_path=WflwConf.test_annotation_path + ds_types[0],
        #                pose_save_path=WflwConf.test_pose_path + ds_types[0],
        #                atr_save_path=WflwConf.test_atr_path + ds_types[0])
        #     for kk in range(len(atr_index)):
        #         self._save(img=img, annotation=annotation, atr=atrs[i], file_name=str(i),
        #                    image_save_path=WflwConf.test_image_path + ds_types[atr_index[kk]],
        #                    annotation_save_path=WflwConf.test_annotation_path + ds_types[atr_index[kk]],
        #                    pose_save_path=WflwConf.test_pose_path+ ds_types[atr_index[kk]],
        #                    atr_save_path=WflwConf.test_atr_path+ ds_types[atr_index[kk]])

        '''tf_record'''
        # if need_tf_ref:
        #     self.wflw_create_tf_record(ds_type=1, need_pose=need_pose)  # we don't need hm for test
        print("create_test_set DONE!!")

    def create_point_imgpath_map(self):
        """
        only used for KD:
        """
        tf_utility = TfUtility()

        img_file_paths = [WflwConf.no_aug_train_image, WflwConf.augmented_train_image]
        annotation_file_paths = [WflwConf.no_aug_train_annotation, WflwConf.augmented_train_annotation]
        map_name = ['map_orig' + DatasetName.dsWflw, 'map_aug' + DatasetName.dsWflw]
        tf_utility.create_point_imgpath_map(img_file_paths=img_file_paths,
                                            annotation_file_paths=annotation_file_paths, map_name=map_name)

    def hm_evaluate_on_wflw(self, model_name, model_file):
        '''create model using the h.5 model and its wights'''
        model = tf.keras.models.load_model(model_file)
        '''load test files and categories:'''
        ds_types = ['pose', 'expression', 'illumination', 'makeup', 'occlusion', 'blur', 'full']
        for ds_type in ds_types:
            test_annotation_paths, test_image_paths = self._get_test_set(ds_type)

            """"""
            evaluation = Evaluation(model=model, anno_paths=test_annotation_paths, img_paths=test_image_paths,
                                    ds_name=DatasetName.dsWflw, ds_number_of_points=WflwConf.num_of_landmarks,
                                    fr_threshold=0.1, is_normalized=True, ds_type=ds_type, model_name=model_name)
            '''predict labels:'''
            nme, fr, AUC = evaluation.predict_annotation_hm()

            print('Dataset: ' + str(DatasetName.dsWflw)
                  + '{ ds_type: ' + str(ds_type) + '} \n\r'
                  + '{ nme: ' + str(nme) + '}\n\r'
                  + '{ fr: ' + str(fr) + '}\n\r'
                  + '{ AUC: ' + str(AUC) + '}\n\r'
                  )
            print('=========================================')

        '''evaluate with meta data: best to worst'''

    def evaluate_on_wflw(self, model_name, model_file, print_result=True):
        """"""
        '''create model using the h.5 model and its wights'''
        model = tf.keras.models.load_model(model_file)
        '''load test files and categories:'''
        ds_types = ['full', 'pose', 'expression', 'illumination', 'makeup', 'occlusion', 'blur']
        nme_arr = []
        fr_arr = []
        AUC_arr = []
        for ds_type in ds_types:
            test_annotation_paths, test_image_paths = self._get_test_set(ds_type)

            """"""
            evaluation = Evaluation(model=model, anno_paths=test_annotation_paths, img_paths=test_image_paths,
                                    ds_name=DatasetName.dsWflw, ds_number_of_points=WflwConf.num_of_landmarks,
                                    fr_threshold=0.1, is_normalized=True, ds_type=ds_type, model_name=model_name)
            '''predict labels:'''
            nme, fr, AUC = evaluation.predict_annotation()
            nme_arr.append(nme)
            fr_arr.append(fr_arr)
            AUC_arr.append(AUC_arr)

            if print_result:
                print('Dataset: ' + str(DatasetName.dsWflw)
                      + '{ ds_type: ' + str(ds_type) + '} \n\r'
                      + '{ nme: ' + str(nme) + '}\n\r'
                      + '{ fr: ' + str(fr) + '}\n\r'
                      + '{ AUC: ' + str(AUC) + '}\n\r'
                      )
                print('=========================================')
        return nme_arr, fr_arr, AUC_arr

    def create_inter_face_web_distance(self, ds_type):
        img_mod = ImageModification()
        if ds_type == 0:
            img_file_path = WflwConf.no_aug_train_image
            annotation_file_path = WflwConf.no_aug_train_annotation
        else:
            img_file_path = WflwConf.test_image_path + 'full'
            annotation_file_path = WflwConf.test_annotation_path + 'full'
        wflw_intra_fwd_pnt = [(6, 7), (0, 1), (32, 31), (7, 8), (0, 2), (32, 30), (25, 24), (25, 26), (16, 15),
                              (16, 17), (85, 94), (79, 90), (90, 94), (76, 79), (79, 82), (82, 85), (76, 85), (55, 57),
                              (57, 59), (57, 54), (51, 55), (51, 59), (61, 67), (63, 65), (60, 64), (68, 72), (69, 75),
                              (71, 73), (35, 40), (44, 48), (37, 38), (42, 50), (10, 9), (10, 11), (22, 21), (22, 23)]

        wflw_inter_fwd_pnt = [(0, 7), (7, 16), (16, 85), (79, 57), (16, 25), (64, 57), (25, 32), (68, 57), (7, 60), (72, 32), (72, 46),
                              (25, 72), (0, 33), (0, 60), (33, 60), (32, 46), (64, 68), (64, 38), (68, 50), (38, 50),  (7, 76), (25, 82)]

        img_mod.create_normalized_web_facial_distance(inter_points=wflw_inter_fwd_pnt,
                                                      intera_points=wflw_intra_fwd_pnt,
                                                      annotation_file_path=annotation_file_path,
                                                      ds_name=DatasetName.dsWflw, img_file_path=img_file_path)

    def create_heatmap(self):
        img_mod = ImageModification()
        for i, anno_file in tqdm(enumerate(os.listdir(WflwConf.augmented_train_annotation))):
            hm = img_mod.generate_hm(width=InputDataSize.hm_size, height=InputDataSize.hm_size,
                                     landmark_path=WflwConf.augmented_train_annotation, landmark_filename=anno_file,
                                     s=WflwConf.hm_sigma, de_normalize=False)
            np.save(WflwConf.augmented_train_hm + anno_file, hm)
    """PRIVATE"""

    def _get_test_set(self, ds_type):
        test_annotation_paths = []
        test_image_paths = []
        for file in tqdm(os.listdir(WflwConf.test_image_path + ds_type)):
            if file.endswith(".png") or file.endswith(".jpg"):
                test_annotation_paths.append(
                    os.path.join(WflwConf.test_annotation_path + ds_type, str(file)[:-3] + "npy"))
                test_image_paths.append(os.path.join(WflwConf.test_image_path + ds_type, str(file)))
        return test_annotation_paths, test_image_paths

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
                                 need_pose=need_pose, accuracy=accuracy, is_test=is_test, ds_name=DatasetName.dsWflw,
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
        xmin = max(0, min(min(ann_x) - rand_padd, xmin))
        xmax = max(max(ann_x) + rand_padd, xmax)
        ymin = max(0, min(min(ann_y) - rand_padd, ymin))
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
                   atr_save_path=WflwConf.no_aug_train_atr,
                   pose_save_path=WflwConf.no_aug_train_pose)

        '''this is the augmented images+original one'''
        for i in range(len(imgs)):
            self._save(img=imgs[i], annotation=annotations[i], file_name=str(index) + '_' + str(i),
                       atr=atr,
                       image_save_path=WflwConf.augmented_train_image,
                       annotation_save_path=WflwConf.augmented_train_annotation,
                       atr_save_path=WflwConf.augmented_train_atr,
                       pose_save_path=WflwConf.augmented_train_pose)
            # img_mod.test_image_print('zzz_final'+str(index)+'-'+str(i), imgs[i], annotations[i])

        return imgs, annotations

    def _crop(self, img, annotation, bbox):
        img_mod = ImageModification()
        ann_xy, ann_x, ann_y = img_mod.create_landmarks(annotation, 1, 1)
        fix_pad = 3
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[6]
        ymax = bbox[7]

        xmin = max(0, min(min(ann_x) - fix_pad, xmin))
        xmax = max(max(ann_x) + fix_pad, xmax)
        ymin = max(0, min(min(ann_y) - fix_pad, ymin))
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
            while line:  # and counter < 200:
                sys.stdout.write('\r \r line --> \033[92m' + str(counter))

                total_data = line.strip().split(' ')
                annotation_arr.append(np.round(list(map(float, total_data[0:WflwConf.num_of_landmarks * 2])), 3))
                bbox_arr.append(self._create_bbox(
                    list(map(int, total_data[WflwConf.num_of_landmarks * 2:WflwConf.num_of_landmarks * 2 + 4])),
                    annotation_arr[counter]))
                atr_arr.append(
                    list(map(int, total_data[WflwConf.num_of_landmarks * 2 + 4:WflwConf.num_of_landmarks * 2 + 10])))
                image_arr.append(self._load_image(WflwConf.orig_WFLW_image + total_data[-1]))
                line = fp.readline()
                counter += 1
        print('Loading Done')
        return image_arr, annotation_arr, bbox_arr, atr_arr

    def _load_image(self, path):
        return np.array(Image.open(path))

    def _create_bbox(self, _bbox, annotation):
        xmin = _bbox[0]
        ymin = _bbox[1]
        xmax = _bbox[2]
        ymax = _bbox[3]
        bbox_me = [xmin, ymin, xmin, ymax, xmax, ymin, xmax, ymax]

        return bbox_me
        # img_mod = ImageModification()
        # ann_xy, an_x, an_y = img_mod.create_landmarks(annotation, 1, 1)
        #
        # fix_padd = 5
        # xmin = int(max(0, min(an_x) - fix_padd))
        # ymin = int(max(0, min(an_y) - fix_padd))
        # xmax = int(max(an_x) + fix_padd)
        # ymax = int(max(an_y) + fix_padd)
        # bbox_me = [xmin, ymin, xmin, ymax, xmax, ymin, xmax, ymax]
        #
        # return bbox_me

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
