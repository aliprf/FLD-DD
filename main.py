from CofwClass import CofwClass
from WflwClass import WflwClass
from W300WClass import W300WClass

if __name__ == '__main__':
    pca_accuracy = 90
    need_pose = True
    need_hm = True
    '''cofw'''
    cofw = CofwClass()
    cofw.create_test_set(need_pose=True, need_tf_ref=True)
    cofw.create_train_set(need_pose=True, need_hm=True, need_tf_ref=True,
                          accuracy=pca_accuracy)
    cofw.create_pca_obj(accuracy=pca_accuracy)
    cofw.cofw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)

    '''wflw'''
    '''for this dataset, for evaluation part we DON'T use the Tf record, just we load the data and images'''
    wflw = WflwClass() # todo DON'T FORGET to remove THE LOAD_DATA LINE LIMIT
    wflw.create_test_set(need_pose=True, need_tf_ref=False)
    wflw.create_train_set(need_pose=True, need_hm=True, accuracy=pca_accuracy)  # todo DON'T FORGET to relabel after fli;
    wflw.create_pca_obj(accuracy=pca_accuracy)
    wflw.wflw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)
    #

    '''300W'''
    '''  for this dataset, for evaluation part we DON'T use the Tf record, just we load the data and images'''
    w300w = W300WClass()  # todo DON'T FORGET to remove counter in load data
    w300w.create_test_set(need_pose=True, need_tf_ref=False)
    w300w.create_train_set(need_pose=True, need_hm=True, accuracy=pca_accuracy)  # todo DON'T FORGET to relabel after fli;
    w300w.create_pca_obj(accuracy=pca_accuracy)
    w300w.w300w_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)
    #
