from CofwClass import CofwClass
from WflwClass import WflwClass
from W300WClass import W300WClass

if __name__ == '__main__':
    pca_accuracy = 95
    need_pose = False
    need_hm = True
    '''cofw'''
    cofw = CofwClass()
    # cofw.create_test_set(need_pose=need_pose, need_tf_ref=False)
    # cofw.create_train_set(need_pose=need_pose, need_hm=True, need_tf_ref=False,
    #                       accuracy=pca_accuracy)
    # cofw.create_pca_obj(accuracy=pca_accuracy)
    # cofw.cofw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)
    # cofw.cofw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=100)
    # cofw.create_point_imgpath_map()
    # cofw.create_inter_face_web_distance(ds_type=0)
    # cofw.evaluate_on_cofw(model_file='./models/ds_cofw_stu_340.h5')

    # mn-stu:       ch->{nme: 0.00, fr: 0.00}
    # mn-base:      ch->{nme: 4.27332, fr: fr: 1.9723}
    # efn-100-base: {nme: 3.8127, fr: fr: 1.9723}

    # '''300W'''
    # '''  for this dataset, for evaluation part we DON'T use the Tf record, just we load the data and images'''
    w300w = W300WClass()  # todo DON'T FORGET to remove counter in load data
    # w300w.create_test_set(need_pose=need_pose, need_tf_ref=False)
    # w300w.create_train_set(need_pose=False, need_hm=True, accuracy=pca_accuracy)  #
    w300w.create_pca_obj(accuracy=pca_accuracy, normalize=True)
    w300w.w300w_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)
    w300w.w300w_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=100)
    # w300w.create_point_imgpath_map()
    # w300w.create_inter_face_web_distance(ds_type=1)

    # w300w.evaluate_on_300w(model_file='./models/ds_300w_ef90.h5')

    # mn-stu:       ch->{nme: 6.44, fr: 6.66}      co->{nme: 3.57, fr: 0.18}        full->{nme:4.14, fr: 1.45}
    # mn-base:      ch->{nme: 0.00, fr: 0.00}      co->{nme: 0.00, fr: 0.00}        full->{nme:0.00, fr: 0.00}
    # efn-100-base: ch->{nme: 0.00, fr: 0.00}      co->{nme: 0.00, fr: 0.00}        full->{nme:0.00, fr: 0.00}

    '''wflw'''
    '''for this dataset, for evaluation part we DON'T use the Tf record, just we load the data and images'''
    wflw = WflwClass() # todo DON'T FORGET to remove THE LOAD_DATA LINE LIMIT
    # wflw.create_test_set(need_pose=need_pose, need_tf_ref=False)
    # wflw.create_train_set(need_pose=False, need_hm=True, accuracy=pca_accuracy)  #
    wflw.create_pca_obj(accuracy=pca_accuracy, normalize=True)
    wflw.wflw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)
    wflw.wflw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=100)
    # wflw.create_point_imgpath_map()
    # wflw.evaluate_on_wflw(model_file='./models/ds_wflw_ac_100_stu.h5')
    # #
    #