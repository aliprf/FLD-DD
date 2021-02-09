class DatasetName:
    ds300W = '300W'
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
    image_input_size = 224
    img_center = image_input_size // 2  # 128

    hm_size = image_input_size // 4  # 64
    hm_center = hm_size // 2  # 64


class WflwConf:
    # Wflw_prefix_path = '/media/data3/ali/FL/new_data/wflw/'  # --> zeus
    # Wflw_prefix_path = '/media/data2/alip/FL/new_data/wflw/'  # --> atlas
    Wflw_prefix_path = '/media/ali/data/new_data/wflw/'  # --> local
    '''original ds data'''
    orig_WFLW_test_path = Wflw_prefix_path + 'WFLW_annotations/list_98pt_test/'

    orig_WFLW_test = Wflw_prefix_path + 'WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
    orig_WFLW_train = Wflw_prefix_path + 'WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'
    orig_WFLW_image = Wflw_prefix_path + 'WFLW_images/'
    '''created testset data'''
    test_annotation_path = Wflw_prefix_path + 'testing_set/annotations/'
    test_atr_path = Wflw_prefix_path + 'testing_set/atrs/'
    test_pose_path = Wflw_prefix_path + 'testing_set/pose/'
    test_image_path = Wflw_prefix_path + 'testing_set/images/'
    test_tf_path = Wflw_prefix_path + 'testing_set/tf/'
    '''created trainset data'''
    '''     augmented version'''
    augmented_train_pose = Wflw_prefix_path + 'training_set/augmented/pose/'
    augmented_train_annotation = Wflw_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_hm = Wflw_prefix_path + 'training_set/augmented/hm/'
    augmented_train_atr = Wflw_prefix_path + 'training_set/augmented/atrs/'
    augmented_train_image = Wflw_prefix_path + 'training_set/augmented/images/'
    augmented_train_tf_path = Wflw_prefix_path + 'training_set/augmented/tf/'
    '''     original version'''
    no_aug_train_annotation = Wflw_prefix_path + 'training_set/no_aug/annotations/'
    no_aug_train_hm = Wflw_prefix_path + 'training_set/augmented/hm/'
    no_aug_train_atr = Wflw_prefix_path + 'training_set/no_aug/atrs/'
    no_aug_train_pose = Wflw_prefix_path + 'training_set/no_aug/pose/'
    no_aug_train_image = Wflw_prefix_path + 'training_set/no_aug/images/'
    no_aug_train_tf_path = Wflw_prefix_path + 'training_set/no_aug/tf/'

    orig_number_of_training = 7500
    orig_number_of_test = 2500

    orig_of_all_test_blur = 773
    orig_of_all_test_expression = 314
    orig_of_all_test_illumination = 689
    orig_of_all_test_largepose = 326
    orig_of_all_test_makeup = 206
    orig_of_all_test_occlusion = 736

    augmentation_factor = 5  # create . image from 4
    num_of_landmarks = 98
    hm_sigma = 3
    '''for tf record: 60890'''
    num_eval_samples_aug = int(orig_number_of_training * augmentation_factor * 0.05)  # 75000* 0.05 = 3750
    num_train_samples_aug = orig_number_of_training * augmentation_factor - num_eval_samples_aug  # 75000 - 3750 = 71250

    num_eval_samples_orig = int(orig_number_of_training * 0.05)  # 7500* 0.05 =
    num_train_samples_orig = orig_number_of_training - num_eval_samples_orig  # 7500 - 7500* 0.05 =


class CofwConf:
    # Cofw_prefix_path = '/media/data3/ali/FL/new_data/cofw/'  # --> zeus
    # Cofw_prefix_path = '/media/data2/alip/FL/new_data/cofw/'  # --> atlas
    Cofw_prefix_path = '/media/ali/data/new_data/cofw/'  # --> local
    #
    orig_COFW_test = Cofw_prefix_path + 'orig_COFW_test/'
    test_annotation_path = Cofw_prefix_path + 'testing_set/annotations/'
    test_pose_path = Cofw_prefix_path + 'testing_set/pose/'
    test_image_path = Cofw_prefix_path + 'testing_set/images/'
    test_tf_path = Cofw_prefix_path + 'testing_set/tf/'

    orig_COFW_train = Cofw_prefix_path + 'orig_COFW_train/'
    augmented_train_pose = Cofw_prefix_path + 'training_set/augmented/pose/'
    augmented_train_hm = Cofw_prefix_path + 'training_set/augmented/hm/'
    augmented_train_annotation = Cofw_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_image = Cofw_prefix_path + 'training_set/augmented/images/'
    augmented_train_tf_path = Cofw_prefix_path + 'training_set/augmented/tf/'

    no_aug_train_annotation = Cofw_prefix_path + 'training_set/no_aug/annotations/'
    no_aug_train_pose = Cofw_prefix_path + 'training_set/no_aug/pose/'
    no_aug_train_image = Cofw_prefix_path + 'training_set/no_aug/images/'
    no_aug_train_tf_path = Cofw_prefix_path + 'training_set/no_aug/tf/'

    orig_number_of_training = 1345
    orig_number_of_test = 507

    augmentation_factor = 10
    num_of_landmarks = 29
    hm_sigma = 3.5
    '''for tf record:'''
    num_eval_samples_aug = 2670 #int(orig_number_of_training * augmentation_factor * 0.05)
    num_train_samples_aug = 50720 #orig_number_of_training * augmentation_factor - num_eval_samples_aug  # 13450 - 670 = 12775

    num_eval_samples_orig = int(orig_number_of_training * 0.05)  #
    num_train_samples_orig = orig_number_of_training - num_eval_samples_orig  #


class W300WConf:
    w300w_prefix_path = '/media/data3/ali/FL/new_data/300W/'  # --> zeus/
    # w300w_prefix_path = '/media/data2/alip/FL/new_data/300W/'  # --> atlas
    # w300w_prefix_path = '/media/ali/data/new_data/300W/'  # --> local

    orig_300W_test = w300w_prefix_path + 'orig_300W_test/'
    test_annotation_path = w300w_prefix_path + 'testing_set/annotations/'
    test_pose_path = w300w_prefix_path + 'testing_set/pose/'
    test_image_path = w300w_prefix_path + 'testing_set/images/'
    test_tf_path = w300w_prefix_path + 'testing_set/tf/'

    orig_300W_train = w300w_prefix_path + 'orig_300W_train/'
    augmented_train_pose = w300w_prefix_path + 'training_set/augmented/pose/'
    augmented_train_annotation = w300w_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_hm = w300w_prefix_path + 'training_set/augmented/hm/'
    augmented_train_image = w300w_prefix_path + 'training_set/augmented/images/'
    augmented_train_tf_path = w300w_prefix_path + 'training_set/augmented/tf/'

    no_aug_train_annotation = w300w_prefix_path + 'training_set/no_aug/annotations/'
    no_aug_train_hm = w300w_prefix_path + 'training_set/augmented/hm/'
    no_aug_train_pose = w300w_prefix_path + 'training_set/no_aug/pose/'
    no_aug_train_image = w300w_prefix_path + 'training_set/no_aug/images/'
    no_aug_train_tf_path = w300w_prefix_path + 'training_set/no_aug/tf/'

    orig_number_of_training = 3148
    orig_number_of_test_full = 689
    orig_number_of_test_common = 554
    orig_number_of_test_challenging = 135

    augmentation_factor = 6  # create . image from 1 16155
    num_of_landmarks = 68
    hm_sigma = 3
    '''for tf record:'''
    num_eval_samples_aug = 807 #int(3507 * augmentation_factor * 0.05)  #
    num_train_samples_aug = 15347 #3507 * augmentation_factor - num_eval_samples_aug  #
    num_eval_samples_orig = int(orig_number_of_training * 0.05)  # 13450* 0.05 = 670
    num_train_samples_orig = orig_number_of_training - num_eval_samples_orig  # 13450 - 670 = 12775
