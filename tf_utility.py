from Config import CofwConf, DatasetName, InputDataSize
from ImageModification import ImageModification
from pose_detection.code.PoseDetector import PoseDetector
from pca_utility import PCAUtility

import os, sys
import numpy as np
from numpy import load, save
from tqdm import tqdm
from PIL import Image
import random
import tensorflow as tf


class TfUtility:
    def create_tf_ref(self, tf_file_paths, img_file_paths, annotation_file_paths, pose_file_paths, need_pose,
                      accuracy, is_test):

        main_tf_name = 'train' + str(accuracy) + '.tfrecords'
        num_main_samples = CofwConf.num_train_samples
        num_eval_samples = CofwConf.num_eval_samples
        if is_test:
            main_tf_name = 'test.tfrecords'
            num_main_samples = CofwConf.orig_number_of_test

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

                    '''load pose npy'''
                    pose_file_name = os.path.join(pose_file_paths[index], file[:-3] + "npy")
                    if need_pose:
                        pose = load(pose_file_name)
                    else:
                        pose = None

                    '''create new landmark using accuracy'''
                    if accuracy != 100:
                        landmark = self._get_asm(landmark, DatasetName.dsCofw, accuracy)
                    '''create tf_record:'''
                    writable_img = np.reshape(img,
                                              [InputDataSize.image_input_size * InputDataSize.image_input_size * 3])

                    if is_test:
                        if need_pose:
                            feature = {'landmarks': self._float_feature(landmark),
                                       'image_raw': self._float_feature(writable_img)
                                       }
                        else:
                            feature = {'landmarks': self._float_feature(landmark),
                                       'pose': self._float_feature(pose),
                                       'image_raw': self._float_feature(writable_img)
                                       }
                    else:
                        if need_pose:
                            feature = {'landmarks': self._float_feature(landmark),
                                       'pose': self._float_feature(pose),
                                       'image_raw': self._float_feature(writable_img),
                                       'image_name': self._bytes_feature(img_tf_name.encode('utf-8')),
                                       }
                        else:
                            feature = {'landmarks': self._float_feature(landmark),
                                       'image_raw': self._float_feature(writable_img),
                                       'image_name': self._bytes_feature(img_tf_name.encode('utf-8')),
                                       }

                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    if counter <= num_main_samples:
                        writer_main.write(example.SerializeToString())
                        msg = 'train --> \033[92m' + " sample number " + str(counter + 1) + \
                              " created." + '\033[94m' + "remains " + str(num_main_samples - counter - 1)
                        sys.stdout.write('\r' + msg)

                    elif tf_evaluation_path is not None:
                        writer_evaluate.write(example.SerializeToString())
                        msg = 'eval --> \033[92m' + " sample number " + str(counter + 1) + \
                              " created." + '\033[94m' + "remains " + str(num_main_samples - counter - 1)
                        sys.stdout.write('\r' + msg)
                    counter += 1

            writer_main.close()
            if tf_evaluation_path is not None:
                writer_evaluate.close()

    def detect_pose(self, images):
        pose_detector = PoseDetector()
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
