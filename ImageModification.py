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


class ImageModification:

    def random_augment(self, index, img_orig, landmark_orig, num_of_landmarks, augmentation_factor, ymin, ymax, xmin,
                       xmax, ds_name, bbox_me_orig, atr=None):
        """"""
        '''keep original'''
        if len(img_orig.shape) < 3:
            landmark_orig = np.stack([img_orig, img_orig, img_orig], axis=-1)

        _img, _landmark = self.crop_image_test(img_orig, ymin, ymax, xmin, xmax, landmark_orig, padding_percentage=0.02)
        _img, _landmark = self.resize_image(_img, _landmark)

        augmented_images = [_img]
        augmented_landmarks = [_landmark]
        '''affine'''
        scale = (np.random.uniform(0.75, 1.25), np.random.uniform(0.75, 1.25))
        # scale = (1, 1)
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

        while aug_num < augmentation_factor:
            img = img_orig
            landmark = landmark_orig
            bbox_me = np.array(bbox_me_orig)
            '''flipping image'''
            if aug_num % 2 == 0:
                img, landmark, bbox_me = self._flip_and_relabel(img, landmark, ds_name, num_of_landmarks, bbox_me)
            '''noise'''
            img = self._noisy(img)

            rot = np.random.uniform(-1 * 0.6, 0.6)
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
            tform_1 = AffineTransform(scale=1, rotation=0, translation=(-x_offset, -y_offset), shear=np.deg2rad(0))
            img_new = transform.warp(t_img, tform_1.inverse, mode='edge')

            '''crop data: we add a small margin to the images'''
            # c_img = self.crop_image_train(img=t_img, bbox=t_bbox)
            c_img, landmark_new = self.crop_image_train(img=img_new, bbox=bbox_new, annotation=landmark_new,
                                                        ds_name=ds_name)
            # self.test_image_print(img_name='bb' + str(index + 1) + '_' + str(aug_num), img=c_img,
            #                       landmarks=landmark_new, bbox_me=bbox_new)

            '''resize'''
            _img, _landmark = self.resize_image(c_img, landmark_new)
            augmented_images.append(_img)
            augmented_landmarks.append(_landmark)
            aug_num += 1

        return augmented_images, augmented_landmarks

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

    def crop_image_train(self, img, bbox, annotation, ds_name):
        if ds_name != DatasetName.dsCofw:
            rand_padd = 0.005 * random.randint(1, 5) * img.shape[0]
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
            bb_xy, bb_x, bb_y = self.create_landmarks(bbox, 1, 1)
            xmin = int(min(bb_x))
            xmax = int(max(bb_x))
            ymin = int(min(bb_y))
            ymax = int(max(bb_y))
            croped_img = img[ymin:ymax, xmin:xmax]
            return croped_img, annotation

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
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = self.create_landmarks(landmark, 1, 1)

        if ymin < 0: ymin = 0
        if xmin < 0: xmin = 0

        x_land_min = int(min(landmark_arr_x)) - padding_percentage * int(min(landmark_arr_x))
        if x_land_min < 0: x_land_min = int(min(landmark_arr_x))
        x_land_max = int(max(landmark_arr_x)) + padding_percentage * int(max(landmark_arr_x))
        y_land_min = int(min(landmark_arr_y)) - padding_percentage * int(min(landmark_arr_y))
        if y_land_min < 0: y_land_min = int(min(landmark_arr_y))
        y_land_max = int(max(landmark_arr_y)) + padding_percentage * int(max(landmark_arr_y))

        xmin = int(min(xmin, x_land_min))
        ymin = int(min(ymin, y_land_min))
        xmax = int(max(xmax, x_land_max))
        ymax = int(max(ymax, y_land_max))

        croped_img = img[ymin:ymax, xmin:xmax]

        if xmax - xmin <= 0 or ymax - ymin <= 0:
            print('ERRORRR11111111111')
        if croped_img.shape[0] == 0 or croped_img.shape[1] == 0:
            print('ERRORRR')
        '''grayscale to color'''
        if len(croped_img.shape) < 3:
            croped_img = np.stack([croped_img, croped_img, croped_img], axis=-1)
            print('grayscale to color')

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

        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='b', s=5)
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
