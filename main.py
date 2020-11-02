from CofwClass import CofwClass
from WflwClass import WflwClass

if __name__ == '__main__':
    pca_accuracy = 90
    need_pose = True
    need_hm = True
    '''cofw'''
    cofw = CofwClass()
    cofw.create_test_set(need_pose=True, need_tf_ref=True)
    cofw.create_train_set(need_pose=True, need_hm=True, need_tf_ref=True,
                          accuracy=pca_accuracy)  # dont forget to relabel after fli;
    cofw.create_pca_obj(accuracy=pca_accuracy)
    cofw.cofw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)

    '''wflw'''
    # wflw = WflwClass()
    # wflw.create_pca_obj(accuracy=pca_accuracy)
    # wflw.create_test_set(need_pose=True, need_tf_ref=True)
    # wflw.create_train_set(need_pose=True, need_hm=True, need_tf_ref=True,
    #                       accuracy=pca_accuracy)  # dont forget to relabel after fli;
    #
