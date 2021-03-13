import numpy as np
from tqdm import tqdm
from PIL import Image
import math, os
import matplotlib.pyplot as plt
from scipy.integrate import trapz

from Config import DatasetName, InputDataSize
from ImageModification import ImageModification
from Config import InputDataSize
import random
"""in evaluation, round all numbers with 3 demical"""


class Evaluation:

    def __init__(self, model_name, model, anno_paths, img_paths, ds_name, ds_number_of_points, fr_threshold,
                 is_normalized=False,
                 ds_type=''):
        self.model = model
        self.model_name = model_name
        self.anno_paths = anno_paths
        self.img_paths = img_paths
        self.ds_name = ds_name
        self.ds_number_of_points = ds_number_of_points
        self.fr_threshold = fr_threshold
        self.is_normalized = is_normalized
        self.ds_type = ds_type

    def predict_hm(self):
        img_mod = ImageModification()
        nme_ar = []
        fail_counter = 0
        sum_loss = 0
        for i in tqdm(range(len(sorted(self.anno_paths)))):
            anno_GT = np.load(self.anno_paths[i])  # the GT are not normalized.
            img = np.expand_dims(np.array(Image.open(self.img_paths[i])) / 255.0, axis=0)
            anno_hms = self.model.predict(img)
            img_mod.test_image_print(img_name=str(i), img=np.array(Image.open(self.img_paths[i])), landmarks=[])
            img_mod.print_image_arr_heat(k=i, image=anno_hms[3][0])

    def predict_annotation_hm(self):
        img_mod = ImageModification()
        nme_ar = []
        fail_counter = 0
        sum_loss = 0
        for i in tqdm(range(len(sorted(self.anno_paths)))):
            anno_GT = np.load(self.anno_paths[i])  # the GT are not normalized.
            # anno_GT_hm = img_mod.generate_hm_from_points(height=64, width=64, lnd_xy=anno_GT, s=7, de_normalize=False)
            img = np.expand_dims(np.array(Image.open(self.img_paths[i])) / 255.0, axis=0)
            anno_Pre_hm = self.model.predict(img)[3][0]
            _, _, anno_Pre = self._hm_to_points(heatmaps=anno_Pre_hm)

            # anno_pre_norm = img_mod.normalize_annotations(annotation=anno_Pre)
            # anno_Pre_asm = img_mod.get_asm(input=anno_pre_norm, dataset_name='300W', accuracy=97)
            #
            # anno_Pre_asm = img_mod.de_normalized(annotation_norm=anno_Pre_asm)

            # anno_Pre_hm = anno_Pre_hm[3][0] # hg

            # anno_Pre_hm = anno_Pre_hm[0][0] # efn
            # _, _, anno_Pre = self._hm_to_points(heatmaps=anno_Pre_hm)
            #
            # anno_Pre_reg = anno_Pre_hm[1][0] # efn
            # anno_Pre = img_mod.de_normalized_hm(annotation_norm=anno_Pre_reg)

            '''print'''
            # img_mod.test_image_print(img_name='z_' + str(i) + '_pr' + str(i) + '__',
            #                          img=np.array(Image.open(self.img_paths[i])) / 255.0, landmarks=anno_Pre)

            # img_mod.test_image_print(img_name='z_' + str(i) + '_prASM' + str(i) + '__',
            #                          img=np.array(Image.open(self.img_paths[i])) / 255.0, landmarks=anno_Pre_asm)
            #
            # img_mod.test_image_print(img_name='z_' + str(i) + '_gt' + str(i) + '__',
            #                          img=np.array(Image.open(self.img_paths[i])) / 255.0, landmarks=anno_GT)

            nme_i, norm_error = self._calculate_nme(anno_GT=anno_GT, anno_Pre=anno_Pre, ds_name=self.ds_name,
                                                    ds_number_of_points=self.ds_number_of_points)
            nme_ar.append(nme_i)
            sum_loss += nme_i
            if nme_i > self.fr_threshold:
                fail_counter += 1

        '''calculate total:'''
        AUC = self.calculate_AUC(nme_arr=nme_ar)
        ''''''
        fr = 100 * fail_counter / len(self.anno_paths)
        nme = 100 * sum_loss / len(self.anno_paths)
        print('fail_counter: ' + str(fail_counter))
        print('fr: ' + str(fr))
        print('nme: ' + str(nme))
        print('AUC: ' + str(AUC))
        return nme, fr, AUC

    def calculate_pw_diff_error(self, student_model):
        img_mod = ImageModification()

        nme_ar = []
        pointwise_nme_ar = []
        fail_counter = 0
        sum_loss = 0

        st_err_all = []
        te_err_all = []
        gt_matrix = []
        pr_matrix = []

        for i in tqdm(range(len(sorted(self.anno_paths)))):
            # if i > 20: break
            anno_GT = np.load(self.anno_paths[i])  # the GT are not normalized.
            img = np.expand_dims(np.array(Image.open(self.img_paths[i])) / 255.0, axis=0)

            '''calculate gt_dif and normalize by 224'''
            anno_Pre_stu = student_model.predict(img)[0]
            if self.is_normalized:
                anno_Pre_stu = img_mod.de_normalized(annotation_norm=anno_Pre_stu)

            gt_dif_gt_st = (anno_GT - anno_Pre_stu)

            '''save'''
            # np.save('./reg_data/'+str(i)+'_pr', anno_Pre_stu)
            # np.save('./reg_data/'+str(i)+'_gt', anno_GT)
            # np.save('./reg_data/'+str(i)+'_dif', gt_dif_gt_st)

            # gt_dif_gt_pt = (anno_GT - anno_Pre_tou)/224.0

            '''diff model: normalized by 224'''
            # dif_pr = dif_model.predict(img)
            # pr_dif_gt_st = dif_pr[0][0]*224.0
            # pr_dif_gt_pt = dif_pr[1][0]

            err_st = gt_dif_gt_st
            # err_st = np.abs(gt_dif_gt_st - pr_dif_gt_st)
            # err_te = np.abs(gt_dif_gt_pt - pr_dif_gt_pt)

            st_err_all.append(err_st)
            pr_matrix.append(anno_Pre_stu)
            gt_matrix.append(anno_GT)
            # te_err_all.append(err_te)

        return np.array(st_err_all), np.array(pr_matrix), np.array(gt_matrix)  # , np.array(te_err_all)

    def calculate_error_with_diff(self, dif_model, student_model, teacher_model, confidence_vector):
        img_mod = ImageModification()

        nme_ar = []
        pointwise_nme_ar = []
        fail_counter = 0
        sum_loss = 0

        for i in tqdm(range(len(sorted(self.anno_paths)))):
            anno_GT = np.load(self.anno_paths[i])  # the GT are not normalized.
            img = np.expand_dims(np.array(Image.open(self.img_paths[i])) / 255.0, axis=0)

            '''diff model: normalized by 224'''
            dif_pr = dif_model.predict(img)
            pr_dif_gt_st = dif_pr[0][0]
            # pr_dif_gt_pt = dif_pr[1][0]

            '''now calculate both st and tou points and denormalize both:'''
            anno_Pre_stu = student_model.predict(img)[0]
            # anno_Pre_tou = teacher_model.predict(img)[0]
            if self.is_normalized:
                anno_Pre_stu = img_mod.de_normalized(annotation_norm=anno_Pre_stu)
            anno_Pre_stu_new = anno_Pre_stu + confidence_vector
            # anno_Pre_stu_new = anno_Pre_stu# + confidence_vector
            # anno_Pre_stu_new = anno_Pre_stu + pr_dif_gt_st*confidence_vector

            # img_mod.test_image_print(img_name='z_' + str(i) + '_GT' + str(i) + '__',
            #                          img=np.array(Image.open(self.img_paths[i])) / 255.0, landmarks=anno_GT)
            # img_mod.test_image_print(img_name='z_' + str(i) + '_st' + str(i) + '__',
            #                          img=np.array(Image.open(self.img_paths[i])) / 255.0, landmarks=anno_Pre_stu)
            # img_mod.test_image_print(img_name='z_' + str(i) + '_tou' + str(i) + '__',
            #                          img=np.array(Image.open(self.img_paths[i])) / 255.0, landmarks=anno_Pre_stu_new)
            # new                       6.139649307861214   3.5736749350703283
            # main: 4.06724982475122	6.13952694633844	3.56227254783015
            nme_i, norm_error = self._calculate_nme(anno_GT=anno_GT, anno_Pre=anno_Pre_stu_new, ds_name=self.ds_name,
                                                    ds_number_of_points=self.ds_number_of_points)
            nme_ar.append(nme_i)
            pointwise_nme_ar.append(norm_error)
            sum_loss += nme_i
            if nme_i > self.fr_threshold:
                fail_counter += 1

        '''calculate total:'''
        AUC = self.calculate_AUC(nme_arr=nme_ar)
        ''''''
        fr = 100 * fail_counter / len(self.anno_paths)
        nme = 100 * sum_loss / len(self.anno_paths)
        print('fr: ' + str(fr))
        print('nme: ' + str(nme))
        print('AUC: ' + str(AUC))
        return nme, fr, AUC, pointwise_nme_ar

    def predict_annotation(self, confidence_vector=None, intercept_vec=None, reg_data=None):
        img_mod = ImageModification()
        nme_ar = []
        pointwise_nme_ar = []
        fail_counter = 0
        sum_loss = 0
        for i in tqdm(range(len(sorted(self.anno_paths)))):
            # if i>20:
            #     break
            anno_GT = np.load(self.anno_paths[i])  # the GT are not normalized.
            img = np.expand_dims(np.array(Image.open(self.img_paths[i])) / 255.0, axis=0)
            # anno_Pre = self.model.predict(img)[0][0]
            anno_Pre = self.model.predict(img)[0]
            if self.is_normalized:
                # anno_Pre_asm = img_mod.get_asm(input=anno_Pre, dataset_name='ibug', accuracy=90)
                # anno_Pre_asm = img_mod.de_normalized(annotation_norm=anno_Pre_asm)
                anno_Pre = img_mod.de_normalized(annotation_norm=anno_Pre)
            if confidence_vector is not None and intercept_vec is not None:
                if reg_data is not None:
                    avg_err_st, sd_err_st = reg_data
                    anno_Pre = confidence_vector[:, 0] * anno_Pre + intercept_vec
                               # confidence_vector[:, 1] * (anno_Pre - avg_err_st) + \
                               # confidence_vector[:, 2] * (anno_Pre - sd_err_st) \
                               # + intercept_vec
                               # +confidence_vector[:, 3] * (avg_err_st - sd_err_st) \
                else:
                    anno_Pre = confidence_vector * anno_Pre + intercept_vec
            elif confidence_vector is not None:
                anno_Pre = confidence_vector * anno_Pre
                anno_Pre = anno_Pre + confidence_vector # this is for AVG
            '''print'''
            # img_mod.test_image_print(img_name='z_' + str(i) + '_pr' + str(i) + '__',
            #                          img=np.array(Image.open(self.img_paths[i])) / 255.0, landmarks=anno_Pre)
            # img_mod.test_image_print(img_name='z_' + str(i) + '_gt' + str(i) + '__',
            #                          img=np.array(Image.open(self.img_paths[i])) / 255.0, landmarks=anno_GT)

            # img_mod.test_image_print(img_name='z_'+str(i)+'_pr'+str(i)+'__', img=np.ones([224,224,3]), landmarks=anno_Pre)
            # img_mod.test_image_print(img_name='z_'+str(i)+'_gt'+str(i)+'__', img=np.ones([224,224,3]), landmarks=anno_GT)
            # img_mod.test_image_print(img_name='z_'+str(i)+'_pr_asm'+str(i)+'__', img=np.ones([224,224,3]), landmarks=anno_Pre_asm)

            nme_i, norm_error = self._calculate_nme(anno_GT=anno_GT, anno_Pre=anno_Pre, ds_name=self.ds_name,
                                                    ds_number_of_points=self.ds_number_of_points)
            nme_ar.append(nme_i)
            pointwise_nme_ar.append(norm_error)
            sum_loss += nme_i
            if nme_i > self.fr_threshold:
                fail_counter += 1

        '''calculate total:'''
        AUC = self.calculate_AUC(nme_arr=nme_ar)
        ''''''
        fr = 100 * fail_counter / len(self.anno_paths)
        nme = 100 * sum_loss / len(self.anno_paths)
        print('fr: ' + str(fr))
        print('nme: ' + str(nme))
        print('AUC: ' + str(AUC))
        return nme, fr, AUC, pointwise_nme_ar

    def predict_pointwise_nme(self):
        img_mod = ImageModification()
        nme_ar = []
        fail_counter = 0
        sum_loss = 0
        for i in tqdm(range(len(sorted(self.anno_paths)))):
            anno_GT = np.load(self.anno_paths[i])  # the GT are not normalized.
            img = np.expand_dims(np.array(Image.open(self.img_paths[i])) / 255.0, axis=0)
            # anno_Pre = self.model.predict(img)[0][0]
            anno_Pre = self.model.predict(img)[0]
            if self.is_normalized:
                anno_Pre = img_mod.de_normalized(annotation_norm=anno_Pre)

            nme_i = self._calculate_nme(anno_GT=anno_GT, anno_Pre=anno_Pre, ds_name=self.ds_name,
                                        ds_number_of_points=self.ds_number_of_points)
            nme_ar.append(nme_i)
            sum_loss += nme_i
            if nme_i > self.fr_threshold:
                fail_counter += 1

        '''calculate total:'''
        AUC = self.calculate_AUC(nme_arr=nme_ar)
        ''''''
        fr = 100 * fail_counter / len(self.anno_paths)
        nme = 100 * sum_loss / len(self.anno_paths)
        print('fr: ' + str(fr))
        print('nme: ' + str(nme))
        print('AUC: ' + str(AUC))
        return nme, fr, AUC

    def _calculate_nme(self, anno_GT, anno_Pre, ds_name, ds_number_of_points):
        normalizing_distance = self.calculate_interoccular_distance(anno_GT=anno_GT, ds_name=ds_name)
        '''here we round all data if needed'''
        sum_errors = 0
        errors_arr = []
        for i in range(0, len(anno_Pre), 2):  # two step each time
            x_pr = anno_Pre[i]
            y_pr = anno_Pre[i + 1]
            x_gt = anno_GT[i]
            y_gt = anno_GT[i + 1]
            error = math.sqrt(((x_pr - x_gt) ** 2) + ((y_pr - y_gt) ** 2))

            # manhattan_error_x = abs(x_pr - x_gt) / 224.0
            # manhattan_error_y = abs(y_pr - y_gt) / 224.0

            sum_errors += error
            # errors_arr.append(manhattan_error_x)
            # errors_arr.append(manhattan_error_y)

        NME = sum_errors / (normalizing_distance * ds_number_of_points)
        norm_error = errors_arr
        return NME, norm_error

    # def calculate_AUC(self, nme_arr):
    #     x, y = self.ecdf(error_arr=nme_arr, threshold=0.1)
    #
    #     x_new = []
    #     y_new = []
    #
    #     x_new.append(0)
    #     y_new.append(0)
    #
    #     index = 1
    #     for i in range(x.size):
    #         if x[i] <= 0.1:
    #             x_new.append(x[i])
    #             y_new.append(y[i])
    #
    #     # print(x_new)
    #     # print(y_new)
    #
    #     plt.scatter(x=x_new, y=y_new)
    #
    #     plt.xlabel('x', fontsize=16)
    #     plt.ylabel('y', fontsize=16)
    #     plt.savefig(self.ds_name+'.png', bbox_inches='tight')
    #     plt.savefig(self.ds_name+'.pdf', bbox_inches='tight')
    #     plt.clf()
    #
    #     #
    #     I2 = trapz(np.array(x), np.array(y))
    #
    #     return I2 # this is AUC
    def calculate_AUC(self, nme_arr):
        '''https://arxiv.org/pdf/1511.05049.pdf'''
        x, y_AUC, y_CED = self.ecdf(error_arr=nme_arr, threshold=0.1)

        # plt.scatter(x=x, y=y)
        #
        # plt.xlabel('Normalized error', fontsize=16)
        # plt.ylabel('Proportion of detected landmarks', fontsize=16)
        # plt.savefig(self.ds_name+'_'+self.ds_type+'.png', bbox_inches='tight')
        # plt.savefig(self.ds_name+'_'+self.ds_type+'.pdf', bbox_inches='tight')
        # plt.clf()

        if self.ds_type == 'full' or self.ds_type == 'full/':
            np.save('./auc_data/' + self.ds_name + '_' + self.model_name + '_x', x)
            np.save('./auc_data/' + self.ds_name + '_' + self.model_name + '_y', y_CED)
        #
        I2 = trapz(np.array(x), np.array(y_AUC))
        # I2 = trapz(np.array(y), np.array(x))

        return I2  # this is AUC

    def ecdf(self, error_arr, threshold=0.1):
        point_to_use = 20
        ce_AUC = []
        ce_CED = []
        # sorted_error_arr = np.sort(error_arr)
        data_range = np.linspace(start=0, stop=threshold, num=point_to_use)
        error_arr = np.array(error_arr)
        sum_errors_AUC = 0
        sum_errors_CED = 0
        for thre in data_range:
            sum_errors_AUC += len(error_arr[error_arr <= thre]) / len(error_arr)
            sum_errors_CED = len(error_arr[error_arr <= thre]) / len(error_arr)
            ce_AUC.append(sum_errors_AUC)
            ce_CED.append(sum_errors_CED)
        return data_range, ce_AUC, ce_CED

    # def ecdf(self, data):
    #     """ Compute ECDF """
    #     x = np.sort(data)
    #     n = x.size
    #     y = np.arange(1, n + 1) / n
    #     return x, y

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

    def __calculate_interpupil_distance(self, labels_true):
        # points: x,y 36--> 41 point for left, and 42->47 for right

        left_pupil_x = (labels_true[72] + labels_true[74] + labels_true[76] + labels_true[78] + labels_true[80] +
                        labels_true[82]) / 6
        left_pupil_y = (labels_true[73] + labels_true[75] + labels_true[77] + labels_true[79] + labels_true[81] +
                        labels_true[83]) / 6

        right_pupil_x = (labels_true[84] + labels_true[86] + labels_true[88] + labels_true[90] + labels_true[92] +
                         labels_true[94]) / 6
        right_pupil_y = (labels_true[85] + labels_true[87] + labels_true[89] + labels_true[91] + labels_true[93] +
                         labels_true[95]) / 6

        dis = math.sqrt(((left_pupil_x - right_pupil_x) ** 2) + ((left_pupil_y - right_pupil_y) ** 2))

        # p1 = [left_pupil_x, left_pupil_y]
        # p2 = [right_pupil_x, right_pupil_y]
        # dis1 = distance.euclidean(p1, p2)
        #
        # print(dis)
        # print(dis1)
        # print('==============') both are equal
        return dis

    def _hm_to_points(self, heatmaps):
        x_points = []
        y_points = []
        xy_points = []
        # print(heatmaps.shape) 56,56,68
        for i in range(heatmaps.shape[2]):
            x, y = self._find_nth_biggest_avg(heatmaps[:, :, i], number_of_selected_points=2,
                                              scalar=4.0)
            x_points.append(x)
            y_points.append(y)
            xy_points.append(x)
            xy_points.append(y)
        return np.array(x_points), np.array(y_points), np.array(xy_points)

    def _find_nth_biggest_avg(self, heatmap, number_of_selected_points, scalar):
        indices = self._top_n_indexes(heatmap, number_of_selected_points)

        x_arr = []
        y_arr = []
        w_arr = []
        x_s = 0.0
        y_s = 0.0
        w_s = 0.0

        for index in indices:
            x_arr.append(index[0])
            y_arr.append(index[1])
            w_arr.append(heatmap[index[0], index[1]])
        #
        for i in range(len(x_arr)):
            x_s += w_arr[i]*x_arr[i]
            y_s += w_arr[i]*y_arr[i]
            w_s += w_arr[i]
        x = (x_s * scalar)/w_s
        y = (y_s * scalar)/w_s

        # x = (0.75 * x_arr[1] + 0.25 * x_arr[0]) * scalar
        # y = (0.75 * y_arr[1] + 0.25 * y_arr[0]) * scalar

        return y, x

        # for index in indices:
        #     x_arr.append(index[0])
        #     y_arr.append(index[1])
        #     w_i = heatmap[index[0], index[1]]
        #
        #     if w_i < 0:
        #         w_i *= -1
        #
        #     if w_i == 0:
        #         w_i = 0.00000000001
        #
        #     w_s += w_i
        #     x_s += (w_i * index[1])
        #     y_s += (w_i * index[0])

        # if w_s > 0:
        #     x_s = (x_s / w_s) * scalar
        #     y_s = (y_s / w_s) * scalar
        #     return x_s, y_s
        # else:
        #     return 0, 0

    def _top_n_indexes(self, arr, n):
        import bottleneck as bn
        idx = bn.argpartition(arr, arr.size - n, axis=None)[-n:]
        width = arr.shape[1]
        xxx = [divmod(i, width) for i in idx]
        # result = np.where(arr == np.amax(arr))
        # return result
        return xxx
    # def equation_test(self, x_gt, x_teach_tough, x_teach_tol, x_pr):
    #     sign_delta_gt_teach_tough = np.sign(x_gt - x_teach_tough)
    #     sign_delta_gt_teach_tol = np.sign(x_gt - x_teach_tol)
    #     sign_delta_gt_pr = np.sign(x_gt - x_pr)
    #
    #     '''considering tough teacher:'''
    #     if sign_delta_gt_pr == sign_delta_gt_teach_tough:
    #         loss_total = abs(x_gt - x_pr) + abs(x_gt - x_teach_tough)
    #     else:
