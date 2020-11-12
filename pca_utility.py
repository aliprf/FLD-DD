from Config import DatasetName, DatasetType, InputDataSize, CofwConf

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pickle
import os
from tqdm import tqdm
from numpy import save, load
import math
from PIL import Image
from numpy import save, load
from ImageModification import ImageModification

class PCAUtility:
    eigenvalues_prefix = "_eigenvalues_"
    eigenvectors_prefix = "_eigenvectors_"
    meanvector_prefix = "_meanvector_"

    def create_pca_from_npy(self, annotation_path, pca_accuracy, pca_file_name, normalize=False):
        print('PCA calculation started: loading labels')
        img_mod = ImageModification()
        lbl_arr = []
        for file in tqdm(os.listdir(annotation_path)):
            if file.endswith(".npy"):
                npy_file = os.path.join(annotation_path, file)
                if normalize:
                    lbl_arr.append(img_mod.normalize_annotations(load(npy_file)))
                else:
                    lbl_arr.append(load(npy_file))

        # lbl_arr = np.array(lbl_arr)

        ''' no normalization is needed, since we want to generate hm'''
        reduced_lbl_arr, eigenvalues, eigenvectors = self._func_PCA(lbl_arr, pca_accuracy)
        mean_lbl_arr = np.mean(lbl_arr, axis=0)
        eigenvectors = eigenvectors.T

        save('pca_obj/' + pca_file_name + self.eigenvalues_prefix + str(pca_accuracy), eigenvalues)
        save('pca_obj/' + pca_file_name + self.eigenvectors_prefix + str(pca_accuracy), eigenvectors)
        save('pca_obj/' + pca_file_name + self.meanvector_prefix + str(pca_accuracy), mean_lbl_arr)

    def test_pca_validity(self, dataset_name, pca_postfix):

        eigenvalues = load('pca_obj/' + dataset_name + self.eigenvalues_prefix + str(pca_postfix) + ".npy")
        eigenvectors = load('pca_obj/' + dataset_name + self.eigenvectors_prefix + str(pca_postfix) + ".npy")
        meanvector = load('pca_obj/' + dataset_name + self.meanvector_prefix + str(pca_postfix) + ".npy")

        '''load data: '''
        lbl_arr = []
        img_arr = []
        path = None
        if dataset_name == DatasetName.ibug:
            path = IbugConf.rotated_img_path_prefix  # rotated is ok, since advs_aug is the same as rotated
            num_of_landmarks = IbugConf.num_of_landmarks
        elif dataset_name == DatasetName.wflw:
            path = WflwConf.rotated_img_path_prefix  # rotated is ok, since advs_aug is the same as rotated
            num_of_landmarks = WflwConf.num_of_landmarks

        for file in tqdm(os.listdir(path)):
            if file.endswith(".pts"):
                pts_file = os.path.join(path, file)
                img_file = pts_file[:-3] + "jpg"
                if not os.path.exists(img_file):
                    continue

                points_arr = []
                with open(pts_file) as fp:
                    line = fp.readline()
                    cnt = 1
                    while line:
                        if 3 < cnt <= num_of_landmarks + 3:
                            x_y_pnt = line.strip()
                            x = float(x_y_pnt.split(" ")[0])
                            y = float(x_y_pnt.split(" ")[1])
                            points_arr.append(x)
                            points_arr.append(y)
                        line = fp.readline()
                        cnt += 1
                lbl_arr.append(points_arr)
                img_arr.append(Image.open(img_file))

        for i in range(100):
            b_vector_p = self.calculate_b_vector(lbl_arr[i], True, eigenvalues, eigenvectors, meanvector)
            lbl_new = meanvector + np.dot(eigenvectors, b_vector_p)
            lbl_new = lbl_new.tolist()

            labels_true_transformed, landmark_arr_x_t, landmark_arr_y_t = image_utility. \
                create_landmarks(lbl_arr[i], 1, 1)

            labels_true_transformed_pca, landmark_arr_x_pca, landmark_arr_y_pca = image_utility. \
                create_landmarks_from_normalized(lbl_new, 224, 224, 112, 112)

            image_utility.print_image_arr(i + 1, img_arr[i], landmark_arr_x_t, landmark_arr_y_t)
            image_utility.print_image_arr((i + 1) * 1000, img_arr[i], landmark_arr_x_pca, landmark_arr_y_pca)

    def calculate_b_vector(self, predicted_vector, correction, eigenvalues, eigenvectors, meanvector):
        tmp1 = predicted_vector - meanvector
        b_vector = np.dot(eigenvectors.T, tmp1)

        # put b in -3lambda =>
        if correction:
            i = 0
            for b_item in b_vector:
                lambda_i_sqr = 3 * math.sqrt(eigenvalues[i])

                if b_item > 0:
                    b_item = min(b_item, lambda_i_sqr)
                else:
                    b_item = max(b_item, -1 * lambda_i_sqr)
                b_vector[i] = b_item
                i += 1

        return b_vector

    def __save_obj(self, obj, name):
        with open('obj/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_pose_obj(self):
        with open('obj/p_1_min.pkl', 'rb') as f:
            p_1_min = pickle.load(f)
        with open('obj/p_1_max.pkl', 'rb') as f:
            p_1_max = pickle.load(f)

        with open('obj/p_2_min.pkl', 'rb') as f:
            p_2_min = pickle.load(f)
        with open('obj/p_2_max.pkl', 'rb') as f:
            p_2_max = pickle.load(f)

        with open('obj/p_3_min.pkl', 'rb') as f:
            p_3_min = pickle.load(f)
        with open('obj/p_3_max.pkl', 'rb') as f:
            p_3_max = pickle.load(f)

        return p_1_min, p_1_max, p_2_min, p_2_max, p_3_min, p_3_max

    def load_pca_obj(self, dataset_name, pca_postfix=97):
        with open('obj/' + dataset_name + self.eigenvalues_prefix + str(pca_postfix) + '.pkl', 'rb') as f:
            eigenvalues = pickle.load(f)
        with open('obj/' + dataset_name + self.eigenvectors_prefix + str(pca_postfix) + '.pkl', 'rb') as f:
            eigenvectors = pickle.load(f)
        with open('obj/' + dataset_name + self.meanvector_prefix + str(pca_postfix) + '.pkl', 'rb') as f:
            meanvector = pickle.load(f)
        return eigenvalues, eigenvectors, meanvector

    def _func_PCA(self, input_data, pca_postfix):
        input_data = np.array(input_data)
        pca = PCA(n_components=pca_postfix / 100)
        # pca = PCA(n_components=0.98)
        # pca = IncrementalPCA(n_components=50, batch_size=50)
        pca.fit(input_data)
        pca_input_data = pca.transform(input_data)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_
        return pca_input_data, eigenvalues, eigenvectors

    def __svd_func(self, input_data, pca_postfix):
        svd = TruncatedSVD(n_components=50)
        svd.fit(input_data)
        pca_input_data = svd.transform(input_data)
        eigenvalues = svd.explained_variance_
        eigenvectors = svd.components_
        return pca_input_data, eigenvalues, eigenvectors
        # U, S, VT = svd(input_data)
