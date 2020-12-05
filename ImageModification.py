from Config import InputDataSize, DatasetName, W300WConf
import random
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from pca_utility import PCAUtility
import math
from skimage.transform import warp, AffineTransform
from skimage.transform import rotate
from PIL import Image
from PIL import ImageOps
from skimage.transform import resize
from skimage import transform
from skimage.transform import SimilarityTransform, AffineTransform
from tqdm import tqdm
import os
import cv2 as cv


# from Evaluation import Evaluation


class ImageModification:

    def random_augment(self, index, img_orig, landmark_orig, num_of_landmarks, augmentation_factor, ymin, ymax, xmin,
                       xmax, ds_name, bbox_me_orig, atr=None):
        """"""
        '''keep original'''
        # print('img_orig:' + str(np.array(img_orig).shape))
        # print('landmark_orig: ' + str(np.array(landmark_orig).shape))
        try:

            if len(img_orig.shape) < 3:
                img_orig = np.stack([img_orig, img_orig, img_orig], axis=-1)

            _img, _landmark = self.crop_image_test(img=img_orig, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax,
                                                   landmark=landmark_orig, padding_percentage=0.0)
            _img, _landmark = self.resize_image(_img, _landmark)

            augmented_images = [_img]
            augmented_landmarks = [_landmark]
            '''affine'''
            # scale = (np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1))
            scale = (1, 1)
            translation = (0, 0)
            shear = 0

            aug_num = 0
            '''if atr is not null'''
            if atr is not None:
                # augmentation_factor = 5
                augmentation_factor += atr[0] * 40  # _pose = atr[0]
                augmentation_factor += atr[1] * 7  # _exp = atr[1]
                augmentation_factor += atr[2] * 2  # _illu = atr[2]
                augmentation_factor += atr[3] * 2  # _mkup = atr[3]
                augmentation_factor += atr[4] * 8  # _occl = atr[4]
                augmentation_factor += atr[5] * 2  # _blr = atr[5]

            while aug_num < augmentation_factor - 1:
                try:
                    '''keep original'''
                    img = np.copy(img_orig)
                    landmark = np.copy(landmark_orig)
                    bbox_me = np.copy(np.array(bbox_me_orig))

                    '''let's pad image before'''
                    fix_pad = InputDataSize.image_input_size * 2
                    img = np.pad(img, ((fix_pad, fix_pad), (fix_pad, fix_pad), (0, 0)), 'constant')
                    for jj in range(len(landmark)):
                        landmark[jj] = landmark[jj] + fix_pad
                    for jj in range(len(bbox_me)):
                        bbox_me[jj] = bbox_me[jj] + fix_pad

                    # self.test_image_print(img_name='zzz_affine' + str(index + 1) + '_' + str(aug_num),
                    #                       img=img, landmarks=landmark, bbox_me=bbox_me)

                    '''flipping image'''
                    if aug_num % 2 == 0:
                        img, landmark, bbox_me = self._flip_and_relabel(img, landmark, ds_name, num_of_landmarks,
                                                                        bbox_me)

                    rot = np.random.uniform(-1 * 0.5, 0.5)
                    sx, sy = scale
                    t_matrix = np.array([
                        [sx * math.cos(rot), -sy * math.sin(rot + shear), 0],
                        [sx * math.sin(rot), sy * math.cos(rot + shear), 0],
                        [0, 0, 1]
                    ])
                    t_matrix_2d = np.array([
                        [sx * math.cos(rot), -sy * math.sin(rot + shear)],
                        [sx * math.sin(rot), sy * math.cos(rot + shear)]
                    ])
                    tform = AffineTransform(scale=scale, rotation=rot, translation=translation, shear=np.deg2rad(shear))

                    t_img = transform.warp(img, tform.inverse, mode='edge')
                    # self.test_image_print(img_name='zzz_affine' + str(index + 1) + '_' + str(aug_num), img=t_img, landmarks=landmark)
                    '''affine landmark'''
                    landmark_arr_xy, landmark_arr_x, landmark_arr_y = self.create_landmarks(landmark, 1, 1)
                    label = np.array(landmark_arr_x + landmark_arr_y).reshape([2, num_of_landmarks])
                    margin = np.ones([1, num_of_landmarks])
                    label = np.concatenate((label, margin), axis=0)

                    t_label = self._reorder(np.delete(np.dot(t_matrix, label), 2, axis=0)
                                            .reshape([2 * num_of_landmarks]), num_of_landmarks)
                    '''affine bbox_me'''
                    bbox_xy, bbox_x, bbox_y = self.create_landmarks(bbox_me, 1, 1)
                    bbox_flat = np.array(bbox_x + bbox_y).reshape([2, len(bbox_me) // 2])
                    t_bbox = self._reorder(np.dot(t_matrix_2d, bbox_flat).reshape([len(bbox_me)]), len(bbox_me) // 2)

                    # self.test_image_print(img_name='aa' + str(index + 1) + '_' + str(aug_num), img=t_img, landmarks=t_label,
                    #                       bbox_me=t_bbox)

                    '''now we need to translate image again'''
                    t_bbox_xy, t_bbox_x, t_bbox_y = self.create_landmarks(t_bbox, 1, 1)
                    x_offset = min(t_bbox_x)
                    y_offset = min(t_bbox_y)

                    landmark_new = []
                    for i in range(0, len(t_label), 2):
                        landmark_new.append(t_label[i] - x_offset)
                        landmark_new.append(t_label[i + 1] - y_offset)
                    bbox_new = []
                    for i in range(0, len(t_bbox), 2):
                        bbox_new.append(t_bbox[i] - x_offset)
                        bbox_new.append(t_bbox[i + 1] - y_offset)
                    '''translate image'''
                    tform_1 = AffineTransform(scale=(1, 1), rotation=0, translation=(-x_offset, -y_offset),
                                              shear=np.deg2rad(0))
                    img_new = transform.warp(t_img, tform_1.inverse, mode='edge')

                    '''crop data: we add a small margin to the images'''
                    # c_img = self.crop_image_train(img=t_img, bbox=t_bbox)
                    c_img, landmark_new = self.crop_image_train(img=img_new, bbox=bbox_new, annotation=landmark_new,
                                                                ds_name=ds_name)
                    # self.test_image_print(img_name='bb' + str(index + 1) + '_' + str(aug_num), img=c_img,
                    #                       landmarks=landmark_new, bbox_me=bbox_new)

                    '''resize'''
                    _img, _landmark = self.resize_image(c_img, landmark_new)

                    '''contras and color modification '''
                    _img = self._adjust_gamma(_img)
                    '''noise'''
                    _img = self._noisy(_img)
                    ''''''
                    _img = self._blur(_img)
                    ''''''
                    augmented_images.append(_img)

                    # '''normalized annotations WE DON'T NORMALIZE as the accuracy will reduce'''
                    # _landmark = self.normalize_annotations(annotation=_landmark)

                    augmented_landmarks.append(_landmark)
                    aug_num += 1
                except Exception as e:
                    print('Exception in random augment')

            return augmented_images, augmented_landmarks
        except Exception as e:
            return None, None

    def _flip_and_relabel(self, img, landmark, ds_name, num_of_landmarks, bbox_me):
        t_matrix = np.array([
            [-1, 0],
            [0, 1]
        ])
        '''flip landmark'''
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = self.create_landmarks(landmark, 1, 1)
        label = np.array(landmark_arr_x + landmark_arr_y).reshape([2, num_of_landmarks])
        t_label = self._reorder(np.dot(t_matrix, label).reshape([2 * num_of_landmarks]), num_of_landmarks)
        '''flip bbox'''
        bbox_xy, bbox_x, bbox_y = self.create_landmarks(bbox_me, 1, 1)
        bbox_flat = np.array(bbox_x + bbox_y).reshape([2, len(bbox_me) // 2])
        t_bbox = self._reorder(np.dot(t_matrix, bbox_flat).reshape([2 * len(bbox_me) // 2]), len(bbox_me) // 2)

        '''we need to shift x'''
        for i in range(0, len(t_label), 2):
            t_label[i] = t_label[i] + img.shape[1]

        '''flip bbox'''
        for i in range(0, len(t_bbox), 2):
            t_bbox[i] = t_bbox[i] + img.shape[1]

        '''flip image'''
        img = np.fliplr(img)

        # self.test_image_print(str(t_label[0]), img, t_label, t_bbox)
        '''need to relabel '''
        t_label = self.relabel_ds(ds_name, t_label)

        return img, t_label, t_bbox

    def relabel_ds(self, ds_name, labels):
        new_labels = np.copy(labels)
        if ds_name == DatasetName.ds300W:
            index_src = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 31, 32, 36, 37, 38, 39, 40, 41, 48, 49, 50,
                         60, 61, 67, 59, 58]
            index_dst = [16, 15, 14, 13, 12, 11, 10, 9, 26, 25, 24, 23, 22, 35, 34, 45, 44, 43, 42, 47, 46, 54, 53, 52,
                         64, 63, 65, 55, 56]

        elif ds_name == DatasetName.dsWflw:
            index_src = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 10, 11, 12, 13, 14, 15, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                         60, 61, 62, 63, 64, 65, 66, 67, 96, 55, 56, 76, 77, 78, 88, 89, 95, 87, 86]
            index_dst = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 46, 45, 44, 43, 42, 50, 49, 48,
                         47,
                         72, 71, 70, 69, 68, 75, 74, 73, 97, 59, 58, 82, 81, 80, 92, 91, 93, 83, 84]

        elif ds_name == DatasetName.dsCofw:
            index_src = [0, 2, 4, 5, 8, 10, 12, 13, 16, 18, 22]
            index_dst = [1, 3, 6, 7, 9, 11, 14, 15, 17, 19, 23]

        for i in range(len(index_src)):
            new_labels[index_src[i] * 2] = labels[index_dst[i] * 2]
            new_labels[index_src[i] * 2 + 1] = labels[index_dst[i] * 2 + 1]

            new_labels[index_dst[i] * 2] = labels[index_src[i] * 2]
            new_labels[index_dst[i] * 2 + 1] = labels[index_src[i] * 2 + 1]
        return new_labels

    def create_normalized_face_web_distance(self, inter_points, intera_points, annotation_file_path, img_file_path,
                                            ds_name):
        """"""
        annotations = []
        images = []
        '''load annotations'''
        counter = 0
        for file in tqdm(os.listdir(annotation_file_path)):
            if file.endswith(".npy"):
                annotations.append(np.load(os.path.join(annotation_file_path, str(file))))
                img_adr = os.path.join(img_file_path, str(file)[:-3] + "jpg")
                # self._print_intra_fb(landmark=annotations[counter], points=intera_points,
                #                      img=np.array(Image.open(img_adr)),
                #                      title=ds_name + ' Intra_Face_Web',
                #                      name='zz_' + ds_name + '_intra_fb_' + str(counter))
                # self._print_inter_fb(landmark=annotations[counter], points=inter_points,
                #                      img=np.array(Image.open(img_adr)),
                #                      title=ds_name + ' Inter_Face_Web',
                #                      name='zz_' + ds_name + '_inter_fb_' + str(counter))
                counter += 1
        '''create inter web face per image'''
        inter_fwd = []
        for item in annotations:
            sum_dis = 0
            '''calculate inter ocular distance as a normalizing factor for each face'''
            inter_ocular_dist = self.calculate_interoccular_distance(anno_GT=item, ds_name=ds_name)
            for i in range(len(inter_points)):
                '''now calculate the distance between each pair'''
                x_1 = inter_points[i][0] * 2
                y_1 = inter_points[i][0] * 2 + 1
                x_2 = inter_points[i][1] * 2
                y_2 = inter_points[i][1] * 2 + 1
                dis = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
                sum_dis += dis

            n_dist = sum_dis / inter_ocular_dist
            inter_fwd.append(n_dist)
        # self._print_bar(data=inter_fwd, title='Intra-Face Web Distance Distribution on ' + ds_name, name='fwd_'+ds_name)

        '''create distribution'''
        count_data = []
        inter_fwd = np.array(inter_fwd)
        range_data = np.linspace(np.amin(inter_fwd), np.amax(inter_fwd), 200)
        for i in range(len(range_data)):
            _count = np.count_nonzero(inter_fwd < range_data[i])
            if i - 1 >= 0:
                yy = np.count_nonzero(inter_fwd < range_data[i - 1])
                _count -= yy
            count_data.append(_count)
            # count_data.append(np.count_nonzero(inter_fwd<range_data[i]))

        self._print_fwd_histo(x_data=range_data, y_data=count_data,
                              title='Intra-Face Web Distance Histogram on ' + ds_name,
                              name='fwd_' + ds_name)

        return inter_fwd

    def calculate_interoccular_distance(self, anno_GT, ds_name):
        if ds_name == DatasetName.ds300W:
            left_oc_x = anno_GT[72]
            left_oc_y = anno_GT[73]
            right_oc_x = anno_GT[90]
            right_oc_y = anno_GT[91]
        elif ds_name == DatasetName.dsCofw:
            left_oc_x = anno_GT[16]
            left_oc_y = anno_GT[17]
            right_oc_x = anno_GT[18]
            right_oc_y = anno_GT[19]
        elif ds_name == DatasetName.dsWflw:
            left_oc_x = anno_GT[192]
            left_oc_y = anno_GT[193]
            right_oc_x = anno_GT[194]
            right_oc_y = anno_GT[195]

        distance = math.sqrt(((left_oc_x - right_oc_x) ** 2) + ((left_oc_y - right_oc_y) ** 2))
        return distance

    def _print_intra_fb(self, landmark, points, img, title, name):
        plt.figure()
        plt.imshow(img)
        plt.title(title)
        for i in range(len(points)):
            x_1 = points[i][0] * 2
            y_1 = points[i][0] * 2 + 1
            x_2 = points[i][1] * 2
            y_2 = points[i][1] * 2 + 1
            plt.plot([landmark[x_1], landmark[x_2]], [landmark[y_1], landmark[y_2]], color='#ff5d8f')

        landmarks_x = []
        landmarks_y = []
        for i in range(0, len(landmark), 2):
            landmarks_x.append(landmark[i])
            landmarks_y.append(landmark[i + 1])

        # for i in range(len(landmarks_x)):
        #     plt.annotate(str(i), (landmarks_x[i], landmarks_y[i]), fontsize=6, color='red')

        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#000000', s=15)
        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#fddb3a', s=3)

        # plt.ylabel('Histogram Of Normalized Distances')
        # plt.xlabel('Faces')
        plt.savefig(name)

    def _print_inter_fb(self, landmark, points, img, title, name):
        plt.figure()
        plt.imshow(img)
        plt.title(title)
        _color = ['#f4a261', '#2a9d8f', '#fca311', '#e63946', '#283618', '#72efdd', '#3d405b', '#72efdd', '#f72585',
                  '#40916c', '#283618', '#fca311', '#011627', '#ff5d8f', '#ffea00', '#ff9500', '#f72585', '#40916c',
                  '#f72585', '#72efdd', '#40916c', '#ff5d8f', '#ffbe0b', '#011627', '#f4a261', '#3d405b', '#f72585',
                  '#e63946', '#011627', '#283618', '#ffbe0b', '#2a9d8f']
        for i in range(len(points)):
            x_1 = points[i][0] * 2
            y_1 = points[i][0] * 2 + 1
            x_2 = points[i][1] * 2
            y_2 = points[i][1] * 2 + 1
            plt.plot([landmark[x_1], landmark[x_2]], [landmark[y_1], landmark[y_2]], color=_color[i])

        landmarks_x = []
        landmarks_y = []
        for i in range(0, len(landmark), 2):
            landmarks_x.append(landmark[i])
            landmarks_y.append(landmark[i + 1])

        # for i in range(len(landmarks_x)):
        #     plt.annotate(str(i), (landmarks_x[i], landmarks_y[i]), fontsize=6, color='red')

        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#000000', s=15)
        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#fddb3a', s=3)

        # plt.ylabel('Histogram Of Normalized Distances')
        # plt.xlabel('Faces')
        plt.savefig(name)

    def _print_fwd_histo(self, x_data, y_data, title, name):
        plt.figure()
        plt.title(title)
        # _colors = ['#{:06x}'.format(random.randint(0, 256**3)) for d in data]
        # plt.plot(data, color='#a8dda8')
        plt.bar(x_data, y_data, color='#d63447')
        # plt.bar(np.arange(len(data)), data, color=_colors)
        plt.ylabel('Count')
        plt.xlabel('Normalized Distance')
        plt.savefig(name)

    def crop_image_train(self, img, bbox, annotation, ds_name):
        if ds_name != DatasetName.dsCofw:
            # rand_padd = 0.005 * img.shape[0] + random.randint(0, 5)
            rand_padd = 3  # 0.005 * img.shape[0]
            ann_xy, ann_x, ann_y = self.create_landmarks(annotation, 1, 1)
            xmin = int(max(0, min(ann_x) - rand_padd))
            xmax = int(max(ann_x) + rand_padd)
            ymin = int(max(0, min(ann_y) - rand_padd))
            ymax = int(max(ann_y) + rand_padd)
            annotation_new = []
            for i in range(0, len(annotation), 2):
                annotation_new.append(annotation[i] - xmin)
                annotation_new.append(annotation[i + 1] - ymin)
            croped_img = img[ymin:ymax, xmin:xmax]
            # if croped_img.shape[1] == 0 or croped_img.shape[0] == 0:
            #     print('--')
            return croped_img, annotation_new
        else:
            '''following block just used the bbox'''
            # bb_xy, bb_x, bb_y = self.create_landmarks(bbox, 1, 1)
            # xmin = int(min(bb_x))
            # xmax = int(max(bb_x))
            # ymin = int(min(bb_y))
            # ymax = int(max(bb_y))
            # croped_img = img[ymin:ymax, xmin:xmax]
            # return croped_img, annotation
            '''this block use the landmarks'''
            rand_padd = 0.005 * img.shape[0] + random.randint(5, 10)
            ann_xy, ann_x, ann_y = self.create_landmarks(annotation, 1, 1)
            xmin = int(max(0, min(ann_x) - rand_padd))
            xmax = int(max(ann_x) + rand_padd)
            ymin = int(max(0, min(ann_y) - rand_padd))
            ymax = int(max(ann_y) + rand_padd)
            annotation_new = []
            for i in range(0, len(annotation), 2):
                annotation_new.append(annotation[i] - xmin)
                annotation_new.append(annotation[i + 1] - ymin)
            croped_img = img[ymin:ymax, xmin:xmax]
            # if croped_img.shape[1] == 0 or croped_img.shape[0] == 0:
            #     print('--')
            return croped_img, annotation_new

    def resize_image(self, img, annotation):
        if img.shape[0] == 0 or img.shape[1] == 0:
            print('resize_image  ERRORRR')

        resized_img = resize(img, (InputDataSize.image_input_size, InputDataSize.image_input_size, 3),
                             anti_aliasing=True)
        dims = img.shape
        height = dims[0]
        width = dims[1]
        scale_factor_y = InputDataSize.image_input_size / height
        scale_factor_x = InputDataSize.image_input_size / width

        '''rescale and retrieve landmarks'''
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = self.create_landmarks(landmarks=annotation,
                                                                                scale_factor_x=scale_factor_x,
                                                                                scale_factor_y=scale_factor_y)
        return resized_img, landmark_arr_xy

    def crop_image_test(self, img, ymin, ymax, xmin, xmax, landmark, padding_percentage=0.0):
        # print(np.array(landmark).shape)
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = self.create_landmarks(landmark, 1, 1)

        if ymin < 0: ymin = 0
        if xmin < 0: xmin = 0

        x_land_min = int(min(landmark_arr_x)) - padding_percentage * int(min(landmark_arr_x))
        if x_land_min < 0: x_land_min = int(min(landmark_arr_x))
        x_land_max = int(max(landmark_arr_x)) + padding_percentage * int(max(landmark_arr_x))
        y_land_min = int(min(landmark_arr_y)) - padding_percentage * int(min(landmark_arr_y))
        if y_land_min < 0: y_land_min = int(min(landmark_arr_y))
        y_land_max = int(max(landmark_arr_y)) + padding_percentage * int(max(landmark_arr_y))

        _xmin = max(0, int(min(xmin, x_land_min)))
        _ymin = max(0, int(min(ymin, y_land_min)))
        _xmax = int(max(xmax, x_land_max))
        _ymax = int(max(ymax, y_land_max))

        if _xmax - _xmin <= 0 or _ymax - _ymin <= 0:
            print('crop_image_test: ERRORRR xmax - xmin <= 0 or ymax - ymin <= 0')
            croped_img = img[ymin:ymax, xmin:xmax]
        else:
            croped_img = img[_ymin:_ymax, _xmin:_xmax]
        '''crop_image_test:grayscale to color'''
        if len(croped_img.shape) < 3:
            croped_img = np.stack([croped_img, croped_img, croped_img], axis=-1)
            print('crop_image_test: grayscale to color')

        landmarks_new = []
        for i in range(0, len(landmark), 2):
            landmarks_new.append(landmark[i] - xmin)
            landmarks_new.append(landmark[i + 1] - ymin)

        return croped_img, landmarks_new

    def test_image_print(self, img_name, img, landmarks, bbox_me=None):
        plt.figure()
        plt.imshow(img)
        ''''''
        if bbox_me is not None:
            bb_x = [bbox_me[0], bbox_me[2], bbox_me[4], bbox_me[6]]
            bb_y = [bbox_me[1], bbox_me[3], bbox_me[5], bbox_me[7]]
            plt.scatter(x=bb_x[:], y=bb_y[:], c='red', s=15)

        ''''''
        landmarks_x = []
        landmarks_y = []
        for i in range(0, len(landmarks), 2):
            landmarks_x.append(landmarks[i])
            landmarks_y.append(landmarks[i + 1])

        for i in range(len(landmarks_x)):
            plt.annotate(str(i), (landmarks_x[i], landmarks_y[i]), fontsize=9, color='red')

        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#000000', s=15)
        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#fddb3a', s=8)
        plt.savefig(img_name + '.png')
        # plt.show()
        plt.clf()

    def create_landmarks(self, landmarks, scale_factor_x, scale_factor_y):
        # landmarks_splited = _landmarks.split(';')
        landmark_arr_xy = []
        landmark_arr_x = []
        landmark_arr_y = []
        for j in range(0, len(landmarks), 2):
            x = float(landmarks[j]) * scale_factor_x
            y = float(landmarks[j + 1]) * scale_factor_y

            landmark_arr_xy.append(x)
            landmark_arr_xy.append(y)  # [ x1, y1, x2,y2 ]

            landmark_arr_x.append(x)  # [x1, x2]
            landmark_arr_y.append(y)  # [y1, y2]

        return landmark_arr_xy, landmark_arr_x, landmark_arr_y

    def normalize_annotations(self, annotation):
        """for training we dont normalize COFW"""

        '''normalize landmarks based on hyperface method'''
        width = InputDataSize.image_input_size
        height = InputDataSize.image_input_size
        x_center = width / 2
        y_center = height / 2
        annotation_norm = []
        for p in range(0, len(annotation), 2):
            annotation_norm.append((x_center - annotation[p]) / width)
            annotation_norm.append((y_center - annotation[p + 1]) / height)
        return annotation_norm

    def de_normalized(self, annotation_norm):
        """for training we dont normalize"""
        width = InputDataSize.image_input_size
        height = InputDataSize.image_input_size
        x_center = width / 2
        y_center = height / 2
        annotation = []
        for p in range(0, len(annotation_norm), 2):
            annotation.append(x_center - annotation_norm[p] * width)
            annotation.append(y_center - annotation_norm[p + 1] * height)
        return annotation

    def create_landmarks(self, landmarks, scale_factor_x, scale_factor_y):
        landmark_arr_xy = []
        landmark_arr_x = []
        landmark_arr_y = []
        for j in range(0, len(landmarks), 2):
            x = float(landmarks[j]) * scale_factor_x
            y = float(landmarks[j + 1]) * scale_factor_y

            landmark_arr_xy.append(x)
            landmark_arr_xy.append(y)  # [ x1, y1, x2,y2 ]

            landmark_arr_x.append(x)  # [x1, x2]
            landmark_arr_y.append(y)  # [y1, y2]

        return landmark_arr_xy, landmark_arr_x, landmark_arr_y

    def _blur(self, image):
        do_or_not = random.randint(0, 100)
        if do_or_not % 2 == 0:
            try:
                image = image * 255.0
                image = np.float32(image)
                image = cv.medianBlur(image, 5)
                image = image / 255.0
            except Exception as e:
                print(str(e))
                pass
            return image

        return image

    def _adjust_gamma(self, image):
        do_or_not = random.randint(0, 100)
        if do_or_not % 2 == 0 or do_or_not % 3 == 0:
            try:
                image = image * 255
                image = np.int8(image)

                dark_or_light = random.randint(0, 100)
                if dark_or_light % 2 == 0 or dark_or_light % 3 == 0:
                    gamma = np.random.uniform(0.3, 0.8)
                else:
                    gamma = np.random.uniform(1.5, 3.5)
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255
                                  for i in np.arange(0, 256)]).astype("uint8")
                image = cv.LUT(image, table)
                image = image / 255.0
                return image
            except Exception as e:
                print(str(e))
                pass
        return image

    def _modify_color(self, image):
        """noise is alpha*pixel_value + beta"""
        image = image * 255

        do_or_not = random.randint(0, 2)
        if do_or_not >= 1:
            beta = random.randint(-127, 127)
            alpha = np.random.uniform(0.1, 1.9)

            min_offset = random.randint(0, 20)
            max_offset = random.randint(230, 250)
            for row in image[:, :, 2]:
                for pixel in row:
                    new_pixel = alpha * pixel + beta
                    if new_pixel <= 3:
                        new_pixel = min_offset
                    elif new_pixel >= 250:
                        new_pixel = max_offset
                    image[:, :, 2] = new_pixel
        image = image / 255.0
        return image

    def _noisy(self, image):
        noise_typ = random.randint(0, 8)
        if noise_typ == 0:
            s_vs_p = 0.1
            amount = 0.1
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out
        if 1 <= noise_typ <= 2:  # "s&p":
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = 0.04
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out
        else:
            return image

    def _reorder(self, input_arr, num_of_landmarks):
        out_arr = []
        for i in range(num_of_landmarks):
            out_arr.append(input_arr[i])
            k = num_of_landmarks + i
            out_arr.append(input_arr[k])
        return np.array(out_arr)

    def calc_teacher_weight_loss(self, x_pr, x_gt, x_t, alpha, alpha_mi, beta, beta_mi):
        weight_loss_t = 0
        if x_t > x_gt:
            if x_pr >= x_t:
                weight_loss_t = alpha
            elif beta <= x_pr < x_t:
                weight_loss_t = alpha_mi
            elif x_gt <= x_pr < beta:
                weight_loss_t = (alpha_mi / (beta - x_gt)) * (x_pr - x_gt)
            elif beta_mi < x_pr < x_gt:
                weight_loss_t = (alpha / (beta_mi - x_gt)) * (x_pr - x_gt)
            elif x_pr <= beta_mi:
                weight_loss_t = alpha
        elif x_t < x_gt:
            if x_pr <= x_t:
                weight_loss_t = alpha
            elif x_t < x_pr <= beta_mi:
                weight_loss_t = alpha_mi
            elif beta_mi < x_pr <= x_gt:
                weight_loss_t = (-alpha_mi / (x_gt - beta_mi)) * (x_pr - x_gt)
            elif x_gt < x_pr <= beta:
                weight_loss_t = (alpha / (beta - x_gt)) * (x_pr - x_gt)
            elif x_pr > beta:
                weight_loss_t = alpha
        return weight_loss_t

    def weight_loss_depict(self, x_gt, x_tough, beta_tough, beta_mi_tough, alpha_tough,
                           x_tol, beta_tol, beta_mi_tol, alpha_tol):
        x_values = np.linspace(-1.0, 1.0, 1000)
        weight_loss_tough = np.zeros_like(x_values)
        weight_loss_tol = np.zeros_like(x_values)
        loss_Tough = np.zeros_like(x_values)
        loss_Tol = np.zeros_like(x_values)
        loss_M = np.zeros_like(x_values)
        loss_Total = np.zeros_like(x_values)

        '''create tough weight'''
        for i, x in enumerate(x_values):
            weight_loss_tough[i] = self.calc_teacher_weight_loss(x_pr=x, x_gt=x_gt, x_t=x_tough, alpha=alpha_tough,
                                                                 alpha_mi=-0.5 * alpha_tough, beta=beta_tough,
                                                                 beta_mi=beta_mi_tough)
            weight_loss_tol[i] = self.calc_teacher_weight_loss(x_pr=x, x_gt=x_gt, x_t=x_tol, alpha=alpha_tol,
                                                               alpha_mi=-0.5 * alpha_tol, beta=beta_tol,
                                                               beta_mi=beta_mi_tol)

        '''creating loss'''
        for i, x in enumerate(x_values):
            loss_Tough[i] = weight_loss_tough[i] * abs(x_tough - x)
            loss_Tol[i] = weight_loss_tol[i] * abs(x_tol - x)
            loss_M[i] = abs(x_gt - x)
            loss_Total[i] = 2 * loss_M[i] + (0.8 * loss_Tough[i] + 0.6 * loss_Tol[i])

        '''depicting'''
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlim(0.2, 0.45)
        ax.set_ylim(-0.1, 0.2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(which='major', color='#968c83', linestyle='--', linewidth=0.4)
        ax.grid(which='minor', color='#9ba4b4', linestyle=':', linewidth=0.3)
        # sct_wl, = ax.plot(x_values[:], weight_loss_tough[:], '#16697a', linewidth=1.5, label='Weight Loss')
        sct_wl, = ax.plot(x_values[:], weight_loss_tol[:], '#efa6f0', linewidth=0.5, label='W_L_tol')
        sct_l, = ax.plot(x_values[:], loss_Tough[:], '#222831', linewidth=1.5, label='Teacher Loss')
        sct_l, = ax.plot(x_values[:], loss_Tol[:], '#190f2f', linewidth=1.0, label='L_Tol')
        sct_l, = ax.plot(x_values[:], loss_M[:], '#3f37c9', linewidth=1.0, label='L_m')
        sct_l, = ax.plot(x_values[:], loss_Total[:], '#fb8b24', linewidth=1.5, label='loss_Total')
        ax.legend()
        sct_wl = plt.scatter(x=[x_gt], y=[0], c='#f05454', s=55, label='x')
        sct_wl = plt.scatter(x=[x_tough], y=[0], c='#ffa62b', s=55, label='x')
        sct_wl = plt.scatter(x=[beta_tough, beta_mi_tough], y=[0, 0], c='#cad315', s=10, label='x')
        # sct_wl = plt.scatter(x=[x_gt, x_gt], y=[alpha_tough, -0.5 * alpha_tough], c='#cad315', s=10, label='x')
        # plt.legend((sct_wl,),
        #            ('x_gt'))

        # for i in range(len(x_values)):
        #     plt.annotate(str(i), (x_values[i], weight_loss[i]), fontsize=9, color='#fd8c04')

        # plt.savefig('loss_weight.png')
        # plt.savefig('loss_teacher.png')
        # plt.savefig('loss_wight_and_teacher.png')
        plt.savefig('total.pdf')

    def loss_function_depict(self, x_gt, x_t_tough, x_t_tol):
        w_max_tough = 0.3
        Beta_tough = (0.9 * x_gt + 0.1 * x_t_tough)

        w_max_tol = 0.6
        Beta_tol = (0.7 * x_gt + 0.3 * x_t_tol)

        # w_max_tough = 0.4
        # Beta = (0.6*x_gt + 0.4*x_t_tough)

        def calc_loss(X_pr):
            if X_pr > x_t_tough:
                loss_tou = -abs(x_t_tough - X_pr)
            elif X_pr >= Beta_tough:
                w_func_tough = w_max_tough
                loss_tou = abs(x_t_tough - X_pr) * w_func_tough
            else:
                slope_tough = w_max_tough / (Beta_tough - x_gt)
                w_func_tough = abs(X_pr - x_gt) * slope_tough
                loss_tou = abs(x_t_tough - X_pr) * w_func_tough

            if X_pr >= Beta_tol:
                w_func_tol = w_max_tol
                loss_tol = abs(x_t_tol - X_pr) * w_func_tol
            else:
                slope_tol = w_max_tol / (Beta_tol - x_gt)
                w_func_tol = (X_pr - x_gt) * slope_tol
                loss_tol = abs(x_t_tol - X_pr) * w_func_tol

            loss_main = abs(x_gt - X_pr)
            # loss_main = np.sqrt(abs(x_gt - X_pr))
            # loss_main = np.square(abs(x_gt - X_pr))
            loss_total = loss_main - 0.5 * (loss_tou + 0.9 * loss_tol)

            # w_func_tough = abs(X_pr - x_gt) * slope_tough
            # loss_tou = abs(x_t_tough - X_pr) * w_func_tough
            '''calc loss total'''
            # loss_main = np.sqrt(abs(x_gt-X_pr))
            return loss_total, loss_main, loss_tou, loss_tol

        x_values = np.linspace(x_gt, x_t_tol, 100)
        loss_total = []
        loss_base = []
        loss_tough = []
        loss_tolerant = []
        for i in range(len(x_values / 2)):
            l_total, l_base, l_tou, l_tol = calc_loss(X_pr=x_values[i])
            loss_total.append(l_total)
            loss_base.append(l_base)
            loss_tough.append(l_tou)
            loss_tolerant.append(l_tol)

        '''finally depict'''
        plt.figure()
        sct_total = plt.scatter(x=x_values[:], y=loss_total[:], c='#fd3a69', s=10)
        sct_base = plt.scatter(x=x_values[:], y=loss_base[:], c='#389393', s=10)
        sct_tough = plt.scatter(x=x_values[:], y=loss_tough[:], c='#fd8c04', s=10)
        sct_tol = plt.scatter(x=x_values[:], y=loss_tolerant[:], c='#81b29a', s=10)
        plt.legend((sct_total, sct_base, sct_tough, sct_tol),
                   ('Total Loss', 'Base Loss', 'Loss Tough', 'Loss Tolerant'))

        # for i in range(len(x_values)):
        #     plt.annotate(str(i), (x_values[i], loss_tough[i]), fontsize=9, color='#fd8c04')
        #     plt.annotate(str(i), (x_values[i], loss_base[i]), fontsize=9, color='#389393')
        #     plt.annotate(str(i), (x_values[i], loss_total[i]), fontsize=9, color='#fd3a69')

        plt.savefig('loss_tough')

    def _loss_function_depict(self, x_gt, x_t_tough, x_t_tol):
        """we assume that """

        '''working with the tough teacher:'''
        '''     creating weight function:'''
        w_max_tough = 0.99
        slope_tough = w_max_tough / (x_t_tough - x_gt)

        def calc_tough_loss(X_pr):
            if X_pr >= abs(x_gt + x_t_tough) / 2:
                w_func_tough = abs(X_pr - x_gt) * slope_tough
                loss_tou = abs(x_t_tough - X_pr) * w_func_tough
                loss_main = abs(x_gt - X_pr)
                loss_total = loss_main - loss_tou

            else:
                w_func_tough = abs(X_pr - x_gt) * slope_tough
                loss_tou = abs(x_t_tough - X_pr) * w_func_tough
                loss_main = abs(x_gt - X_pr)
                loss_total = max(loss_main, loss_tou)

            # w_func_tough = abs(X_pr - x_gt) * slope_tough
            # loss_tou = abs(x_t_tough - X_pr) * w_func_tough
            '''calc loss total'''
            # loss_main = np.sqrt(abs(x_gt-X_pr))
            return loss_total, loss_main, loss_tou

        x_values = np.linspace(x_t_tough, x_gt, 200)
        loss_total = []
        loss_base = []
        loss_tough = []
        for i in range(len(x_values / 2)):
            l_total, l_base, l_tou = calc_tough_loss(X_pr=x_values[i])
            loss_total.append(l_total)
            loss_base.append(l_base)
            loss_tough.append(l_tou)

        '''finally depict'''
        plt.figure()
        sct_total = plt.scatter(x=x_values[:], y=loss_total[:], c='#fd3a69', s=10)
        sct_base = plt.scatter(x=x_values[:], y=loss_base[:], c='#389393', s=10)
        sct_tough = plt.scatter(x=x_values[:], y=loss_tough[:], c='#fd8c04', s=10)
        plt.legend((sct_total, sct_base, sct_tough), ('Total Loss', 'Base Loss', 'Loss Tough'))

        # for i in range(len(x_values)):
        #     plt.annotate(str(i), (x_values[i], loss_tough[i]), fontsize=9, color='#fd8c04')
        #     plt.annotate(str(i), (x_values[i], loss_base[i]), fontsize=9, color='#389393')
        #     plt.annotate(str(i), (x_values[i], loss_total[i]), fontsize=9, color='#fd3a69')

        plt.savefig('loss_tough')

    def depict_face_distribution(self):
        imgs = []
        lbls = []
        lbls_asm = []
        lbls_asm_prim = []
        for i, file in enumerate(os.listdir(W300WConf.no_aug_train_image)):
            if file.endswith(".jpg") or file.endswith(".png"):
                lbl_file = str(file)[:-3] + "npy"  # just name
                # img_filenames.append(str(file))
                lbls.append(self._load_and_normalize(W300WConf.no_aug_train_annotation + lbl_file))
                lbls_asm.append(
                    self.get_asm(input=self._load_and_normalize(W300WConf.no_aug_train_annotation + lbl_file),
                                 dataset_name='ibug', accuracy=90))
                lbls_asm_prim.append(
                    list(lbls[i][j] - np.sign(lbls_asm[i][j] - lbls[i][j]) * abs(lbls_asm[i][j] - lbls[i][j])
                         for j in range(len(lbls[i]))))

        # lbls_asm_prim = np.array(lbls_asm_prim)
        # self.print_arr(0, type='face', landmarks_arr=lbls)
        self.print_histogram_plt(0, type='full', landmarks_arr=lbls[:100])
        # self.print_arr(1, type='face', landmarks_arr=lbls_asm)
        self.print_histogram_plt(1, type='full', landmarks_arr=lbls_asm[:100])
        # self.print_arr(2, type='full', landmarks_arr=lbls_asm_prim)
        self.print_histogram_plt(2, type='full', landmarks_arr=lbls_asm_prim[:100])

    def _load_and_normalize(self, point_path):
        annotation = load(point_path)

        """for training we dont normalize COFW"""

        '''normalize landmarks based on hyperface method'''
        width = InputDataSize.image_input_size
        height = InputDataSize.image_input_size
        x_center = width / 2
        y_center = height / 2
        annotation_norm = []
        for p in range(0, len(annotation), 2):
            annotation_norm.append((x_center - annotation[p]) / width)
            annotation_norm.append((y_center - annotation[p + 1]) / height)
        return annotation_norm

    def get_asm(self, input, dataset_name, accuracy):
        pca_utils = PCAUtility()

        eigenvalues = load('pca_obj/' + dataset_name + pca_utils.eigenvalues_prefix + str(accuracy) + ".npy")
        eigenvectors = load('pca_obj/' + dataset_name + pca_utils.eigenvectors_prefix + str(accuracy) + ".npy")
        meanvector = load('pca_obj/' + dataset_name + pca_utils.meanvector_prefix + str(accuracy) + ".npy")

        b_vector_p = pca_utils.calculate_b_vector(input, True, eigenvalues, eigenvectors, meanvector)
        out = meanvector + np.dot(eigenvectors, b_vector_p)
        return out

    def print_arr(self, k, type, landmarks_arr):
        import random

        plt.figure()
        for lndm in landmarks_arr:
            color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

            landmark_arr_xy, landmark_arr_x, landmark_arr_y = self.create_landmarks(landmarks=lndm,
                                                                                    scale_factor_x=1,
                                                                                    scale_factor_y=1)
            if type == 'full':
                plt.scatter(x=landmark_arr_x[:], y=landmark_arr_y[:], c=color, s=2)
            elif type == 'face':
                plt.scatter(x=landmark_arr_x[0:32], y=landmark_arr_y[0:32], c=color, s=5)
                plt.plot(landmark_arr_x[0:32], landmark_arr_y[0:32], '-ok', c=color)

            elif type == 'eyes':
                plt.scatter(x=landmark_arr_x[60:75], y=landmark_arr_y[60:75], c=color, s=5)
                plt.plot(landmark_arr_x[60:75], landmark_arr_y[60:75], '-ok', c=color)

            elif type == 'nose':
                plt.scatter(x=landmark_arr_x[51:59], y=landmark_arr_y[51:59], c=color, s=5)
                plt.plot(landmark_arr_x[51:59], landmark_arr_y[51:59], '-ok', c=color)

            elif type == 'mouth':
                plt.scatter(x=landmark_arr_x[76:95], y=landmark_arr_y[76:95], c=color, s=5)
                plt.plot(landmark_arr_x[76:95], landmark_arr_y[76:95], '-ok', c=color)

            elif type == 'eyebrow':
                plt.scatter(x=landmark_arr_x[33:50], y=landmark_arr_y[33:50], c=color, s=5)
                plt.plot(landmark_arr_x[33:50], landmark_arr_y[33:50], '-ok', c=color)

        plt.axis('off')
        plt.savefig('name_' + str(type) + '_' + str(k) + '.png', bbox_inches='tight')
        # plt.show()
        plt.clf()

    def print_histogram_plt(self, k, type, landmarks_arr):
        import matplotlib.ticker as ticker
        import random
        # var = np.var(landmarks_arr, axis=0)
        # mean = np.mean(landmarks_arr, axis=0)
        #
        colors = ['#440047', '#158467', '#0f4c75']
        plt.figure()
        for lndm in tqdm(landmarks_arr):
            data = lndm
            color = colors[k]

            if type == 'face':
                data = lndm[0:64]
                color = '#008891'

            # color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

            # plt.plot(data, '-ok', c='#799351')
            plt.hist(data, bins=60, color=color, alpha=0.3, histtype='bar')
            # plt.hist(data, bins=num_of_bins, color=color, edgecolor='green', alpha=0.2)

            # plt.axis.set_major_formatter(ticker.PercentFormatter(xmax=len(landmark_arr_x)))

        # plt.text(0,15, 'mean: '+str(mean)+', var:' + str(var))
        plt.savefig('histo_' + str(type) + '_' + str(k) + '.png', bbox_inches='tight')
        plt.clf()
