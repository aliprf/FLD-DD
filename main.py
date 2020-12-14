from CofwClass import CofwClass
from WflwClass import WflwClass
from W300WClass import W300WClass
from ImageModification import ImageModification

if __name__ == '__main__':
    img_mod = ImageModification()
    # img_mod.depict_face_distribution()
    #
    # x_gt = 0.3
    # x_tough = 0.312
    # beta_tough = x_gt+0.4*abs(x_gt-x_tough)
    # beta_mi_tough = x_gt-0.4*abs(x_gt-x_tough)
    # alpha_tough = 0.9
    ''''''
    # x_tol = 0.4
    # beta_tol = x_gt + 0.4 * abs(x_gt - x_tol)
    # beta_mi_tol = x_gt - 0.4 * abs(x_gt - x_tol)
    # alpha_tol = 0.9
    # ''''''
    # img_mod.weight_loss_depict(x_gt=x_gt, x_tough=x_tough, beta_tough=beta_tough, beta_mi_tough=beta_mi_tough,
    #                            alpha_tough=alpha_tough,
    #                            x_tol=x_tol, beta_tol=beta_tol, beta_mi_tol=beta_mi_tol, alpha_tol=alpha_tol)

    # img_mod.loss_function_depict(x_gt=-0.5, x_t_tough=0.4, x_t_tol=0.5)
    """
    when we load the data, we round it with 3 float points. Then later we normalized it when we wanna create tfRecord.
    for test, before calculating, we deNormalize it.
    """
    pca_accuracy = 90
    need_pose = False
    need_hm = True
    '''cofw'''
    cofw = CofwClass()
    # cofw.create_test_set(need_pose=need_pose, need_tf_ref=False)
    # cofw.create_train_set(need_pose=need_pose, need_hm=True, need_tf_ref=False,
    #                       accuracy=pca_accuracy)
    # cofw.create_pca_obj(accuracy=80)
    cofw.create_pca_obj(accuracy=80)
    cofw.create_pca_obj(accuracy=85)
    cofw.create_pca_obj(accuracy=90)
    cofw.create_pca_obj(accuracy=95)
    cofw.create_pca_obj(accuracy=97)

    # cofw.cofw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)
    # cofw.cofw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=100)
    # cofw.create_point_imgpath_map()
    # cofw.create_inter_face_web_distance(ds_type=0)
    # cofw.evaluate_on_cofw(model_file='./models/cofw/ds_cofw_mn_base.h5')
    # cofw.evaluate_on_cofw(model_file='./models/cofw/stu_model_cofw_nme4.11.h5')
    # cofw.evaluate_on_cofw(model_file='./models/stu_model_78.h5')

    # mn-base:      ch->{nme: 5.04, fr: fr: 3.74}
    # mn-stu:       ch->{nme: 4.11, fr: 2.36}
    # efn-100-base: {nme: 3.8127, fr: fr: 1.9723}

    # '''300W'''
    # '''  for this dataset, for evaluation part we DON'T use the Tf record, just we load the data and images'''
    w300w = W300WClass()  # todo DON'T FORGET to remove counter in load data
    # w300w.create_test_set(need_pose=need_pose, need_tf_ref=False)
    # w300w.create_train_set(need_pose=False, need_hm=True, accuracy=pca_accuracy)  #
    w300w.create_pca_obj(accuracy=80, normalize=True)
    w300w.create_pca_obj(accuracy=85, normalize=True)
    w300w.create_pca_obj(accuracy=90, normalize=True)
    w300w.create_pca_obj(accuracy=95, normalize=True)
    w300w.create_pca_obj(accuracy=97, normalize=True)
    # w300w.w300w_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)
    # w300w.w300w_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=100)
    # w300w.create_point_imgpath_map()
    # w300w.create_inter_face_web_distance(ds_type=1)

    # w300w.evaluate_on_300w(model_file='./models/stu_300w_better.h5')
    # w300w.evaluate_on_300w(model_file='./models/300w/ds_300w_ef_100.h5')
    # w300w.evaluate_on_300w(model_file='./models/stu_model_590_ibug.h5')

    # mn-stu:       ch->{nme:3.83 , fr: 0.0 }      co->{nme:4.09 , fr:0.18 }        full->{nme:4.04, fr: 0.145 }
    # mn-base:      ch->{nme: 5.75 , fr: 3.70}      co->{nme: 3.83, fr: 0.18}        full->{nme:4.21, fr: 0.72}
    # efn-100-base: ch->{nme: 4.02, fr: 0.0}      co->{nme: 3.36, fr: 0.00}        full->{nme:3.49, fr: 0.0}

    '''wflw'''
    '''for this dataset, for evaluation part we DON'T use the Tf record, just we load the data and images'''
    wflw = WflwClass() # todo DON'T FORGET to remove THE LOAD_DATA LINE LIMIT
    # wflw.create_test_set(need_pose=need_pose, need_tf_ref=False)
    # wflw.create_train_set(need_pose=False, need_hm=True, accuracy=pca_accuracy)  #
    # wflw.create_pca_obj(accuracy=pca_accuracy, normalize=True)
    wflw.create_pca_obj(accuracy=80, normalize=True)
    wflw.create_pca_obj(accuracy=85, normalize=True)
    wflw.create_pca_obj(accuracy=90, normalize=True)
    wflw.create_pca_obj(accuracy=95, normalize=True)
    wflw.create_pca_obj(accuracy=97, normalize=True)
    # wflw.wflw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)
    # wflw.wflw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=100)
    # wflw.create_inter_face_web_distance(ds_type=1)
    # wflw.create_point_imgpath_map()
    # wflw.evaluate_on_wflw(model_file='./models/wflw/ds_wflw_ef100.h5') #ch: 12.96
    # wflw.evaluate_on_wflw(model_file='./models/wflw/ds_wflw_mnbase.h5') #ch:17.74
    # wflw.evaluate_on_wflw(model_file='./models/stu_model_125_wflw.h5')
    # #
    #