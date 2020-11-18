from Config import InputDataSize, DatasetName
import random
import numpy as np
import matplotlib.pyplot as plt
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
                augmentation_factor += atr[0] * augmentation_factor  # _pose = atr[0]
                augmentation_factor += atr[1] * augmentation_factor  # _exp = atr[1]
                augmentation_factor += atr[2] * 2  # _illu = atr[2]
                augmentation_factor += atr[3] * 2  # _mkup = atr[3]
                augmentation_factor += atr[4] * augmentation_factor  # _occl = atr[4]
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

    def create_normalized_face_web_distance(self, points, annotation_file_path, img_file_path, ds_name):
        """"""
        evalu = Evaluation(model='', anno_paths='', img_paths='', ds_name=ds_name, ds_number_of_points=0,
                           fr_threshold=0)
        annotations = []
        images = []
        '''load annotations'''
        counter = 0
        for file in tqdm(os.listdir(annotation_file_path)):
            if file.endswith(".npy"):
                annotations.append(np.load(os.path.join(annotation_file_path, str(file))))
                img_adr = os.path.join(img_file_path, str(file)[:-3] + "jpg")
                self._print_intra_fb(landmark=annotations[counter], points=points, img=np.array(Image.open(img_adr)),
                                     title=ds_name + 'inter Face Web', name='z_' + ds_name + 'ia_fb_' + str(counter))
                counter += 1
        inter_fwd = []
        for item in annotations:
            sum_dis = 0
            inter_ocular_dist = evalu.calculate_interoccular_distance(anno_GT=item, ds_name=ds_name)
            for i in range(len(points)):
                x_1 = points[i][0] * 2
                y_1 = points[i][0] * 2 + 1
                x_2 = points[i][1] * 2
                y_2 = points[i][1] * 2 + 1
                dis = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
                sum_dis += dis
            n_dist = sum_dis / inter_ocular_dist
            inter_fwd.append(n_dist)
        # self._print_bar(data=inter_fwd, title='Intra-Face Web Distance Distribution on ' + ds_name, name='fwd_'+ds_name)

        '''create distribution'''
        count_data = []
        inter_fwd = np.array(inter_fwd)
        range_data = np.linspace(np.amin(inter_fwd), np.amax(inter_fwd), 50)
        for i in range(len(range_data)):
            _count = np.count_nonzero(inter_fwd < range_data[i])
            if i - 1 >= 0:
                yy = np.count_nonzero(inter_fwd < range_data[i - 1])
                _count -= yy
            count_data.append(_count)
            # count_data.append(np.count_nonzero(inter_fwd<range_data[i]))

        self._print_fwd_histo(data=count_data, title='Intra-Face Web Distance Distribution on ' + ds_name,
                              name='fwd_' + ds_name)

        return inter_fwd

    def _print_intra_fb(self, landmark, points, img, title, name):
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

    def _print_fwd_histo(self, data, title, name):
        plt.figure()
        plt.title(title)
        # _colors = ['#{:06x}'.format(random.randint(0, 256**3)) for d in data]
        # plt.plot(data, color='#a8dda8')
        plt.bar(np.arange(len(data)), data, color='#01c5c4')
        # plt.bar(np.arange(len(data)), data, color=_colors)
        plt.ylabel('Histogram Of Normalized Distances')
        plt.xlabel('Faces')
        plt.savefig(name)

    def crop_image_train(self, img, bbox, annotation, ds_name):
        if ds_name != DatasetName.dsCofw:
            rand_padd = 0.005 * img.shape[0] + random.randint(0, 5)
            # rand_padd = 0.005 * img.shape[0]
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
            plt.annotate(str(i), (landmarks_x[i], landmarks_y[i]), fontsize=6, color='red')

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
        if do_or_not % 2 == 0:
            try:
                image = image * 255
                image = np.int8(image)

                dark_or_light = random.randint(0, 100)
                if dark_or_light % 2 == 0:
                    gamma = np.random.uniform(0.25, 0.5)
                else:
                    gamma = np.random.uniform(2, 3.5)
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
        if True or do_or_not >= 1:
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
        image = image/255.0
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
