class DatasetName:
    ds300W = 'w300'
    dsCofw = 'cofw'
    dsWflw = 'wflw'

    dsWflw_test = 'wflw_test'
    dsCofw_test = 'cofw_test'
    ds300w_test = 'ibug_test'


class DatasetType:
    ds300w_challenging = 10
    ds300w_comomn = 11
    ds300w_full = 12

    dsWflw_full = 20
    dsWflw_blur = 21
    dsWflw_expression = 22
    dsWflw_illumination = 23
    dsWflw_largepose = 24
    dsWflw_makeup = 25
    dsWflw_occlusion = 26


class InputDataSize:
    image_input_size = 256
    img_center = image_input_size // 2  # 128

    hm_size = image_input_size // 4  # 64
    hm_center = hm_size // 2  # 64


class WflwConf:
    Wflw_prefix_path = '/media/ali/data/new_data/wflw/'  # --> local

    orig_number_of_training = 7500
    orig_number_of_test = 2500

    orig_of_all_test_blur = 773
    orig_of_all_test_expression = 314
    orig_of_all_test_illumination = 689
    orig_of_all_test_largepose = 326
    orig_of_all_test_makeup = 206
    orig_of_all_test_occlusion = 736

    augmentation_factor = 4  # create . image from 4
    num_of_landmarks = 98
    hm_stride = 3


class CofwConf:
    Cofw_prefix_path = '/media/ali/data/new_data/cofw/'  # --> local

    orig_COFW_test = Cofw_prefix_path + 'orig_COFW_test/'
    test_annotation_path = Cofw_prefix_path + 'testing_set/annotations/'
    test_image_path = Cofw_prefix_path + 'testing_set/images/'

    orig_COFW_train = Cofw_prefix_path + 'orig_COFW_train/'
    augmented_train_annotation = Cofw_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_image = Cofw_prefix_path + 'training_set/augmented/images/'
    no_aug_train_annotation = Cofw_prefix_path + 'training_set/no_aug/annotations/'
    no_aug_train_image = Cofw_prefix_path + 'training_set/no_aug/images/'

    orig_number_of_training = 1345
    orig_number_of_test = 507

    augmentation_factor = 5
    num_of_landmarks = 29
    hm_stride = 3


class W300W:
    _Ibug_prefix_path = '/media/ali/data/new_data/300W/'  # --> local

    orig_number_of_training = 3148
    orig_number_of_test_full = 689
    orig_number_of_test_common = 554
    orig_number_of_test_challenging = 135

    augmentation_factor = 3  # create . image from 1
    num_of_landmarks = 68
    hm_stride = 3
