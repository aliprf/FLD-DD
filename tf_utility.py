from Config import CofwConf, DatasetName, InputDataSize, WflwConf, W300W
from ImageModification import ImageModification
from pca_utility import PCAUtility

import os, sys
import numpy as np
from numpy import load, save
from tqdm import tqdm
from PIL import Image
import random
import tensorflow as tf
import pickle


class TfUtility:
    def create_tf_ref(self, tf_file_paths, img_file_paths, annotation_file_paths, pose_file_paths, need_pose,
                      accuracy, is_test, num_train_samples, num_eval_samples, ds_name):
        img_mod = ImageModification()
        main_tf_name = 'train' + str(accuracy) + '.tfrecords'

        if is_test:
            main_tf_name = 'test.tfrecords'

        for index in range(len(tf_file_paths)):
            tf_main_path = tf_file_paths[index] + main_tf_name  # could be test or train
            tf_evaluation_path = tf_file_paths[index] + 'eval' + str(accuracy) + '.tfrecords'

            counter = 0
            writer_main = tf.python_io.TFRecordWriter(tf_main_path)
            if not is_test:
                writer_evaluate = tf.python_io.TFRecordWriter(tf_evaluation_path)
            for file in os.listdir(img_file_paths[index]):
                if file.endswith(".jpg") or file.endswith(".png"):
                    img_tf_name = self._encode_tf_file_name(file)
                    img_file_name = os.path.join(img_file_paths[index], file)
                    '''load img and normalize it'''
                    img = Image.open(img_file_name)
                    img = np.array(img) / 255.0
                    '''load landmark npy, (has been augmented already)'''
                    landmark_file_name = os.path.join(annotation_file_paths[index], file[:-3] + "npy")
                    landmark = load(landmark_file_name)

                    # '''load pose npy'''
                    # pose_file_name = os.path.join(pose_file_paths[index], file[:-3] + "npy")
                    # if need_pose:
                    #     pose = load(pose_file_name)
                    # else:
                    #     pose = None

                    '''create new landmark using accuracy'''
                    if accuracy != 100:
                        landmark = self._get_asm(landmark, ds_name, accuracy)
                    # img_mod.test_image_print(img_name=str(index), landmarks=landmark, img=img)

                    '''create tf_record:'''
                    writable_img = np.reshape(img,
                                              [InputDataSize.image_input_size * InputDataSize.image_input_size * 3])

                    if is_test:
                        # if need_pose:
                        #     feature = {'landmarks': self._float_feature(landmark),
                        #                'image_raw': self._float_feature(writable_img)
                        #                }
                        # else:
                        feature = {'landmarks': self._float_feature(landmark),
                                   'image_raw': self._float_feature(writable_img)
                                   }
                    else:
                        # if need_pose:
                        #     feature = {'landmarks': self._float_feature(landmark),
                        #                'pose': self._float_feature(pose),
                        #                'image_raw': self._float_feature(writable_img),
                        #                'image_name': self._bytes_feature(img_tf_name.encode('utf-8')),
                        #                }
                        # else:
                        feature = {'landmarks': self._float_feature(landmark),
                                   'image_raw': self._float_feature(writable_img),
                                   'image_name': self._bytes_feature(img_tf_name.encode('utf-8')),
                                   }

                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    if counter <= num_train_samples[index]:
                        writer_main.write(example.SerializeToString())
                        msg = 'train --> \033[92m' + " sample number " + str(counter + 1) + \
                              " created." + '\033[94m' + "remains " + str(num_train_samples[index] - counter - 1)
                        sys.stdout.write('\r' + msg)

                    else:
                        writer_evaluate.write(example.SerializeToString())
                        msg = 'eval --> \033[92m' + " sample number " + str(counter + 1) + \
                              " created." + '\033[94m' + "remains " + str(num_train_samples[index] - counter - 1)
                        sys.stdout.write('\r' + msg)
                    counter += 1

            writer_main.close()
            if tf_evaluation_path is not None:
                writer_evaluate.close()

    def detect_pose(self, images, pose_detector):
        poses = []
        _images = np.copy(np.array(images))
        for i in range(len(_images)):
            yaw_predicted, pitch_predicted, roll_predicted = pose_detector.detect(_images[i], isFile=False, show=False)
            '''normalize pose -1 -> +1 '''
            min_degree = -65
            max_degree = 65
            yaw_normalized = 2 * ((yaw_predicted - min_degree) / (max_degree - min_degree)) - 1
            pitch_normalized = 2 * ((pitch_predicted - min_degree) / (max_degree - min_degree)) - 1
            roll_normalized = 2 * ((roll_predicted - min_degree) / (max_degree - min_degree)) - 1

            pose_array = np.array([yaw_normalized, pitch_normalized, roll_normalized])
            poses.append(pose_array)
        return poses

    def _encode_tf_file_name(self, file_name):
        while len(file_name) < 15:
            file_name = "X" + file_name
        return file_name

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _get_asm(self, _input, pca_file_name, accuracy):
        pca_utils = PCAUtility()
        eigenvalues = load('pca_obj/' + pca_file_name + pca_utils.eigenvalues_prefix + str(accuracy) + ".npy")
        eigenvectors = load('pca_obj/' + pca_file_name + pca_utils.eigenvectors_prefix + str(accuracy) + ".npy")
        meanvector = load('pca_obj/' + pca_file_name + pca_utils.meanvector_prefix + str(accuracy) + ".npy")

        b_vector_p = pca_utils.calculate_b_vector(_input, True, eigenvalues, eigenvectors, meanvector)
        out = meanvector + np.dot(eigenvectors, b_vector_p)
        return out

    # def create_point_imgpath_map_tf_record(self, dataset_name):
    #     print('create_point_imgpath_map_tf_record')
    #     map = {}
    #     if dataset_name == DatasetName.ibug:
    #         tf_path = IbugConf.tf_train_path
    #         sample_counts = IbugConf.number_of_train_sample
    #         landmarks_dir = IbugConf.normalized_points_npy_dir
    #
    #     elif dataset_name == DatasetName.cofw:
    #         tf_path = CofwConf.tf_train_path
    #         sample_counts = CofwConf.number_of_train_sample
    #         landmarks_dir = CofwConf.normalized_points_npy_dir
    #
    #     elif dataset_name == DatasetName.wflw:
    #         tf_path = WflwConf.tf_train_path
    #         sample_counts = WflwConf.number_of_train_sample
    #         landmarks_dir = WflwConf.normalized_points_npy_dir
    #
    #     # sample_counts = 1708
    #     lbl_arr, img_arr, pose_arr, img_name_arr = self.retrieve_tf_record_train(tf_path,
    #                                                                              number_of_records=sample_counts,
    #                                                                              only_label=True)
    #     counter = 0
    #     # f = open("key_"+dataset_name, "a")
    #     for lbl in tqdm(lbl_arr):
    #         img_name = self._decode_tf_file_name(img_name_arr[counter].decode("utf-8"))
    #         landmark_key = lbl.tostring()
    #         # img_name = os.path.join(landmarks_dir, img_name)
    #         map[landmark_key] = img_name
    #
    #         # f.write(str(landmark_key))
    #         counter += 1
    #     # f.close()
    #
    #     pkl_file = open("map_" + dataset_name, 'wb')
    #     pickle.dump(map, pkl_file)
    #     pkl_file.close()
    #
    #     file = open("map_" + dataset_name, 'rb')
    #     load_map = pickle.load(file)
    #     print(load_map)
    #     file.close()