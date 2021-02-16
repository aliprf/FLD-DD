from CofwClass import CofwClass
from WflwClass import WflwClass
from W300WClass import W300WClass
from ImageModification import ImageModification

if __name__ == '__main__':
    img_mod = ImageModification()
    # img_mod.depict_face_distribution()

    # x_gt = -0.0
    # x_tough = 0.1
    # beta_tough = x_gt+0.4*abs(x_gt-x_tough)
    # beta_mi_tough = x_gt-0.4*abs(x_gt-x_tough)
    # alpha_tough = 0.9
    # ''''''
    # x_tol = -0.2
    # beta_tol = x_gt + 0.4 * abs(x_gt - x_tol)
    # beta_mi_tol = x_gt - 0.4 * abs(x_gt - x_tol)
    # alpha_tol = 0.9
    # # ''''''
    # img_mod.weight_loss_depict(x_gt=x_gt, x_tough=x_tough, beta_tough=beta_tough, beta_mi_tough=beta_mi_tough,
    #                            alpha_tough=alpha_tough,
    #                            x_tol=x_tol, beta_tol=beta_tol, beta_mi_tol=beta_mi_tol, alpha_tol=alpha_tol)

    """
    when we load the data, we round it with 3 float points. Then later we normalized it when we wanna create tfRecord.
    for test, before calculating, we deNormalize it.
    """
    pca_accuracy = 90
    need_pose = False
    need_hm = True
    models = ['Teacher', 'Student', 'mnv2', 'ASM']
    # img_mod.depict_AUC_CURVE()
    '''cofw'''
    cofw = CofwClass()
    # cofw.depict_prediction_error_distribution(diff_net_w_path='./models/cofw/kd/ds_cofw_stu.h5',
    #                                           student_w_path='./models/cofw/kd/ds_cofw_stu.h5',
    #                                           teacher_w_path='./models/cofw/kd/ds_cofw_efn_100.h5')

    # cofw.create_test_set(need_pose=need_pose, need_tf_ref=False)
    # cofw.batch_test(weight_files_path='/media/data2/alip/kd_weights/cofw/24_jan_2021/', csv_file_path='./cofw_CSV_BATCH_RESULT.csv')

    # cofw.create_test_set(need_pose=need_pose, need_tf_ref=False)
    # cofw.create_train_set(need_pose=need_pose, need_hm=True, need_tf_ref=False,
    #                       accuracy=pca_accuracy)
    # cofw.create_heatmap() #
    # cofw.create_pca_obj(accuracy=80)
    # cofw.create_pca_obj(accuracy=80)
    # cofw.create_pca_obj(accuracy=85)
    # cofw.create_pca_obj(accuracy=90)
    # cofw.create_pca_obj(accuracy=95)
    # cofw.create_pca_obj(accuracy=97)

    # cofw.cofw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)
    # cofw.cofw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=100)
    # cofw.create_point_imgpath_map()
    # cofw.create_inter_face_web_distance(ds_type=0)
    # cofw.evaluate_on_cofw(model_name=models[0], model_file='./models/cofw/ds_cofw_ac_100_teacher.h5')
    # cofw.evaluate_on_cofw(model_name=models[1], model_file='./models/cofw/kd/ds_cofw_stu.h5') #4.11
    # cofw.evaluate_on_cofw(model_name=models[2], model_file='./models/cofw/kd/ds_cofw_mn_base.h5') #5.04

    # cofw.evaluate_on_cofw(model_name=models[3], model_file='./models/cofw/asm/ds_cofw_asm.h5')
    # cofw.evaluate_on_cofw(model_name='', model_file='./models/asm_fw_model_215_cofw_0.h5')

    # cofw.evaluate_on_cofw(model_name='', model_file='./models/stu_model_191_cofw_.h5')

    # cofw.evaluate_on_cofw(model_name='', model_file='./models/last_cofw_efn.h5')
    #
    # mn-base:      ch->{nme: 5.04, fr: 3.74,  AUC: 0.7213}
    # mn-stu:       ch->{nme: 4.11, fr: 2.366  AUC: 0.7907}
    # efn-100:      ch->{nme: 3.81,fr: 1.972   AUC: 0.8098}

    # mn-asm:       ch->{nme: 4.08, fr: 1.97}

    # '''300W'''
    # '''  for this dataset, for evaluation part we DON'T use the Tf record, just we load the data and images'''
    w300w = W300WClass()  # todo DON'T FORGET to remove counter in load data
    #
    # confidence_vector, avg_err_st, var_err_st, sd_err_st, intercept_vector = w300w.point_wise_diff_evaluation(
    #     # diff_net_w_path='./models/z_diff/300w/dif_model_ibug.h5',
    #     student_w_path='./models/300w/KD_main_correct/ds_300w_mn_stu.h5',
    #     # student_w_path='./models/ds_300w_mn_stu.h5',
    #     use_save=True)
    #     # teacher_w_path='./models/300w/KD_main_correct/ds_300w_ef_100.h5')
    #
    # nme, fr, AUC, pointwise_nme_ar = w300w.evaluate_on_300w(model_name=models[0],
    #                                                         # model_file='./models/ds_300w_mn_stu.h5',
    #                                                         model_file='./models/300w/KD_main_correct/ds_300w_mn_stu.h5',
    #                                                         # )
    #                                                         confidence_vector=confidence_vector,
    #                                                         intercept_vec=intercept_vector,
    #                                                         reg_data=[avg_err_st, sd_err_st])
    # mn-stu:           ch->{nme:6.13, fr:3.70: ,AUC:0.6029}    co->{nme:3.56, fr:0.180 ,AUC: 0.8356}    full->{nme:4.067, fr: 0.870, AUC:0.790}
    # coef->mn-stu:     ch->{nme:, fr: ,AUC:}    co->{nme:, fr: ,AUC: 0}    full->{nme:, fr: , AUC:}


    # print('6')
    # w300w.create_test_set(need_pose=need_pose, need_tf_ref=False)
    # w300w.batch_test(weight_files_path='/media/data3/ali/kd_weights/300w/24_jan_2021/', csv_file_path='./300w_CSV_BATCH_RESULT.csv')

    # w300w.create_train_set(need_pose=False, need_hm=True, accuracy=pca_accuracy)  #
    # w300w.create_mean_face()  #
    # w300w.create_heatmap() #
    # w300w.create_pca_obj(accuracy=80, normalize=True)
    # w300w.create_pca_obj(accuracy=85, normalize=True)
    # w300w.create_pca_obj(accuracy=90, normalize=True)
    # w300w.create_pca_obj(accuracy=95, normalize=True)
    # w300w.create_pca_obj(accuracy=97, normalize=True)
    # w300w.w300w_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)
    # w300w.w300w_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=100)
    # w300w.create_point_imgpath_map()
    # w300w.create_inter_face_web_distance(ds_type=1)
    # w300w.create_sample(ds_type=1)
    #
    # w300w.evaluate_on_300w(model_name='stu_tol', model_file='./models/300w/KD_main_correct/stu_tol.h5')
    # w300w.evaluate_on_300w(model_name='stu_tou', model_file='./models/300w/KD_main_correct/stu_tou.h5')
    #
    # w300w.evaluate_on_300w(model_name=models[0], model_file='./models/300w/stu_model_4_ibug.h5')

    # w300w.evaluate_on_300w(model_name=models[0], model_file='./models/300w/KD_main_correct/ds_300w_ef_100.h5')
    # w300w.evaluate_on_300w(model_name=models[1], model_file='./models/300w/KD_main_correct/ds_300w_mn_stu.h5')
    # w300w.evaluate_on_300w(model_name=models[2], model_file='./models/300w/KD_main_correct/ds_300w_mn_base.h5')

    # w300w.evaluate_on_300w(model_name=models[0], model_file='./models/300w/KD_main_correct/ds_300w_3o_stu_.h5')
    # w300w.evaluate_on_300w(model_name=models[0], model_file='./models/stu_model_57_ibug_0.007199302.h5')

    # w300w.evaluate_on_300w(model_name=models[0], model_file='./models/300w/ds_300w_ef_100_better.h5')
    # w300w.evaluate_on_300w(model_name=models[1], model_file='./models/300w/KD_main/kd_300W_stu.h5')
    # w300w.evaluate_on_300w(model_name=models[2], model_file='./models/300w/KD_main/ds_300w_mn_base.h5')
    # w300w.evaluate_on_300w(model_name=models[3], model_file='./models/300w/ASM/ASM_300W_model_6_40.h5')
    # w300w.evaluate_on_300w(model_name='--', model_file='./models/asm_fw_model_6_300W_0.h5')

    # w300w.evaluate_on_300w(model_name='--', model_file='./models/asm_4_300W_efNb0.h5')
    # w300w.evaluate_on_300w(model_name='--', model_file='./models/stu_model_41_ibug_0.h5')

    # mn-base:  ch->{nme:6.84, fr:7.40, AUC:0.5425}  co->{nme:3.93, fr:0.18, AUC:0.8102} full->{nme:4.50, fr:1.59, AUC:0.7578}
    # mn-stu:   ch->{nme:6.13, fr:3.70, AUC:0.6029}  co->{nme:3.56, fr:0.18, AUC:0.8356} full->{nme:4.06, fr:0.87, AUC:0.7900}

    # mn-tough:   ch->{nme:6.59 fr:5.92  AUC:0.5509}      co->{nme:3.85, fr:0.18  AUC:0.8146}  full->{nme:4.39 , fr:1.30 ,AUC:0.7630}
    # mn-tol:     ch->{nme:6.41 fr:4.44  AUC:0.5736}      co->{nme:3.73, fr:0.18  AUC:0.8230}  full->{nme:4.25 , fr:1.01 ,AUC:0.7742}
    # efn-100:    ch->{nme:5.80 fr:3.70  AUC:0.6423}   co->{nme: 3.34 , fr: 0.0,  AUC:0.8512}  full->{nme:3.82, fr: 0.72, AUC:0.8103}

    # mn-asm:         ch->{nme:3.91 , fr: 0.0 }      co->{nme:3.81 , fr:0.0 }        full->{nme:3.83, fr: 0.0 }
    # mn-asm-ALW:     ch->{nme:4.21 , fr: 0.0 }      co->{nme:3.82 , fr:0.18 }        full->{nme:3.90, fr: 0.14 }
    # mn-asm-WFD:     ch->{nme:4.39 , fr: 0.0 }      co->{nme:3.89 , fr:0.18 }        full->{nme:3.99, fr: 0.14 }

    '''wflw'''
    '''for this dataset, for evaluation part we DON'T use the Tf record, just we load the data and images'''
    wflw = WflwClass()  # todo DON'T FORGET to remove THE LOAD_DATA LINE LIMIT
    # wflw.create_test_set(need_pose=need_pose, need_tf_ref=False)
    # wflw.batch_test(weight_files_path='/media/data2/alip/kd_weights/wflw/24_jan_2021/', csv_file_path='./wflw_CSV_BATCH_RESULT.csv')

    # wflw.create_train_set(need_pose=False, need_hm=True, accuracy=pca_accuracy)  #
    # wflw.create_heatmap()
    # wflw.create_pca_obj(accuracy=pca_accuracy, normalize=True)
    # wflw.create_pca_obj(accuracy=80, normalize=True)
    # wflw.create_pca_obj(accuracy=85, normalize=True)
    # wflw.create_pca_obj(accuracy=90, normalize=True)
    # wflw.create_pca_obj(accuracy=95, normalize=True)
    # wflw.create_pca_obj(accuracy=97, normalize=True)
    # wflw.wflw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=pca_accuracy)
    # wflw.wflw_create_tf_record(ds_type=0, need_pose=need_pose, accuracy=100)
    # wflw.create_inter_face_web_distance(ds_type=0)
    # wflw.create_point_imgpath_map()

    # wflw.evaluate_on_wflw(model_name='-', model_file='./models/asm_135_wflw_efNb0.h5') #ch:

    # wflw.evaluate_on_wflw(model_name=models[0], model_file='./models/wflw/kd/ds_wflw_efn_100.h5') #ch: 12.96
    # wflw.evaluate_on_wflw(model_name=models[1], model_file='./models/wflw/kd/ds_wflw_stu_model_16.34.h5') #ch:17.74
    # wflw.evaluate_on_wflw(model_name=models[2], model_file='./models/wflw/kd/ds_wflw_mn.h5') #ch:

    # wflw.evaluate_on_wflw(model_name='mn', model_file='./models/wflw/asm/ds_wflw_mn.h5') #ch:
    # wflw.evaluate_on_wflw(model_name='ASM', model_file='./models/wflw/asm/ASM_main_16_89.h5')
    # wflw.evaluate_on_wflw(model_name='ALW', model_file='./models/wflw/asm/ASM_ALW_17_08.h5')
    # wflw.evaluate_on_wflw(model_name='WFD', model_file='./models/wflw/asm/ASM__WFD_17_58.h5')

    # wflw.hm_evaluate_on_wflw(model_name=models[3], model_file='./models/asm_fw_model_23_wflw_0.h5') #ch:
    # #
    #
