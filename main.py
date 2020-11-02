from CofwClass import CofwClass

if __name__ == '__main__':
    cofw = CofwClass()
    # cofw.create_test_set(need_pose=True, need_tf_ref=True)
    cofw.create_train_set(need_pose=True, need_hm=True, need_tf_ref=True) # dont forget to relabel after fli;