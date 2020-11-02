from CofwClass import CofwClass
from pca_utility import PCAUtility

if __name__ == '__main__':
    '''data'''
    cofw = CofwClass()
    # cofw.create_pca_obj(accuracy=95)
    # cofw.create_test_set(need_pose=True, need_tf_ref=True)
    cofw.create_train_set(need_pose=True, need_hm=True, need_tf_ref=True,
                          accuracy=95)  # dont forget to relabel after fli;
