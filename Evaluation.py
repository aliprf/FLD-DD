import numpy as np
from tqdm import tqdm
from PIL import Image
import math
import matplotlib.pyplot as plt
from scipy.integrate import trapz

from Config import DatasetName, InputDataSize
from ImageModification import ImageModification

"""in evaluation, round all numbers with 3 demical"""
class Evaluation:

    def __init__(self, model_name, model, anno_paths, img_paths, ds_name, ds_number_of_points, fr_threshold, is_normalized=False,
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

    def predict_annotation(self):
        img_mod = ImageModification()
        nme_ar = []
        fail_counter = 0
        sum_loss = 0
        for i in tqdm(range(len(sorted(self.anno_paths)))):
            # if i>20:
            #     break
            anno_GT = np.load(self.anno_paths[i]) # the GT are not normalized.
            img = np.expand_dims(np.array(Image.open(self.img_paths[i])) / 255.0, axis=0)
            anno_Pre = self.model.predict(img)[0]
            if self.is_normalized:
                # anno_Pre_asm = img_mod.get_asm(input=anno_Pre, dataset_name='ibug', accuracy=99)
                # anno_Pre = img_mod.de_normalized(annotation_norm=anno_Pre_asm)
                anno_Pre = img_mod.de_normalized(annotation_norm=anno_Pre)

            '''print'''
            # img_mod.test_image_print(img_name='z_'+str(i)+'_pr'+str(i)+'__', img=np.array(Image.open(self.img_paths[i])) / 255.0, landmarks=anno_Pre)
            # img_mod.test_image_print(img_name='z_'+str(i)+'_gt', img=np.array(Image.open(self.img_paths[i])) / 255.0, landmarks=anno_GT)
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
        for i in range(0, len(anno_Pre), 2):  # two step each time
            x_pr = anno_Pre[i]
            y_pr = anno_Pre[i + 1]
            x_gt = anno_GT[i]
            y_gt = anno_GT[i + 1]
            error = math.sqrt(((x_pr - x_gt) ** 2) + ((y_pr - y_gt) ** 2))
            sum_errors += error
        NME = sum_errors / (normalizing_distance * ds_number_of_points)
        return NME

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
        x, y = self.ecdf(error_arr=nme_arr, threshold=0.1)

        # plt.scatter(x=x, y=y)
        #
        # plt.xlabel('Normalized error', fontsize=16)
        # plt.ylabel('Proportion of detected landmarks', fontsize=16)
        # plt.savefig(self.ds_name+'_'+self.ds_type+'.png', bbox_inches='tight')
        # plt.savefig(self.ds_name+'_'+self.ds_type+'.pdf', bbox_inches='tight')
        # plt.clf()

        if self.ds_type == 'full' or self.ds_type == 'full/':
            np.save('./auc_data/'+self.ds_name + '_' + self.model_name + '_x', x)
            np.save('./auc_data/'+self.ds_name + '_' + self.model_name + '_y', y)
        #
        I2 = trapz(np.array(x), np.array(y))
        # I2 = trapz(np.array(y), np.array(x))

        return I2 # this is AUC

    def ecdf(self, error_arr, threshold=0.1):
        point_to_use = 20
        ce = []
        # sorted_error_arr = np.sort(error_arr)
        data_range = np.linspace(start=0, stop=threshold, num=point_to_use)
        error_arr = np.array(error_arr)
        for thre in data_range:
            sum_errors = len(error_arr[error_arr <= thre])/len(error_arr)
            ce.append(sum_errors)
        return data_range, ce


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

    # def equation_test(self, x_gt, x_teach_tough, x_teach_tol, x_pr):
    #     sign_delta_gt_teach_tough = np.sign(x_gt - x_teach_tough)
    #     sign_delta_gt_teach_tol = np.sign(x_gt - x_teach_tol)
    #     sign_delta_gt_pr = np.sign(x_gt - x_pr)
    #
    #     '''considering tough teacher:'''
    #     if sign_delta_gt_pr == sign_delta_gt_teach_tough:
    #         loss_total = abs(x_gt - x_pr) + abs(x_gt - x_teach_tough)
    #     else:
