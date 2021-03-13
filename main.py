from CofwClass import CofwClass
from WflwClass import WflwClass
from W300WClass import W300WClass
from ImageModification import ImageModification
import run_presentation_utils as demo_u

if __name__ == '__main__':
    '''create demo'''
    # demo_u.convert_movie_to_imge(video_address='./demo/KD/1.mp4', save_path='./demo/KD/imgs/')
    # demo_u.detect_FLP(img_path='./demo/KD/imgs/', save_path='./demo/KD/annotated_img/',
    #                   # model_path='./models/300w/KD_main_correct/ds_300w_ef_100.h5')
    #                   model_path='./models/wflw/kd/ds_wflw_efn_100.h5')
    #

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

    # img_mod.ASM_weight_depict(gamma=0.2, sigma=50)

    """
    when we load the data, we round it with 3 float points. Then later we normalized it when we wanna create tfRecord.
    for test, before calculating, we deNormalize it.
    """
    pca_accuracy = 97
    need_pose = False
    need_hm = True
    models = ['Teacher', 'Student', 'mnv2', 'ASM']
    # img_mod.depict_AUC_CURVE()
    '''cofw'''
    cofw = CofwClass()
    # confidence_vector, avg_err_st, var_err_st, sd_err_st, intercept_vector = cofw.point_wise_diff_evaluation(
    #     student_w_path='./models/cofw/kd/ds_cofw_stu.h5',
    #     use_save=False)

    # cofw.create_test_set(need_pose=need_pose, need_tf_ref=False)
    # cofw.batch_test(weight_files_path='/media/data2/alip/kd_weights/cofw/24_jan_2021/', csv_file_path='./cofw_CSV_BATCH_RESULT.csv')

    # cofw.create_test_set(need_pose=need_pose, need_tf_ref=False)
    cofw.create_train_set(need_pose=need_pose, need_hm=False, need_tf_ref=False,
                          accuracy=pca_accuracy)
    cofw.create_heatmap() #
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

    # cofw.evaluate_on_cofw(model_name='0', model_file='./models/ASM_10_cofw_nme_10.h5')

    '''asm efn'''
    # cofw.evaluate_on_cofw(model_name='efn_fawl', model_file='./models/ASM_169_cofw_nme_4.235380054596289_fr_3.353057199211045.h5')
    # cofw.evaluate_on_cofw(model_name='efn_base', model_file='./models/ASM_69_cofw_nme_4.948433464921171_fr_3.9447731755424065.h5')


    # cofw.hm_evaluate_on_cofw(model_name='hm', model_file='models/IAL178_cofw_0.h5')
    #
    # mn-base:      ch->{nme: 5.04, fr: 3.74,  AUC: 0.7213}
    # mn-stu:       ch->{nme: 4.11, fr: 2.366  AUC: 0.7907}
    # efn-100:      ch->{nme: 3.81,fr: 1.972   AUC: 0.8098}

    # mn-asm:       ch->{nme: 4.08, fr: 1.97, AUC: 0.7928}

    # '''300W'''
    # '''  for this dataset, for evaluation part we DON'T use the Tf record, just we load the data and images'''
    w300w = W300WClass()  # todo DON'T FORGET to remove counter in load data
    #
    # confidence_vector, avg_err_st, var_err_st, sd_err_st, intercept_vector = w300w.point_wise_diff_evaluation(
    #     # diff_net_w_path='./models/z_diff/300w/dif_model_ibug.h5',
    #     student_w_path='./models/300w/KD_main_correct/ds_300w_mn_stu.h5',
    #     # student_w_path='./models/ds_300w_mn_stu.h5',
    #     use_save=False)
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

    w300w.create_train_set(need_pose=False, need_hm=False, accuracy=pca_accuracy)  #
    # w300w.create_mean_face()  #
    w300w.create_heatmap() #
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
    # w300w.evaluate_on_300w(model_name='', model_file='./models/ASM_1_300W_nme_.h5')

    # w300w.hm_evaluate_on_300w(model_name='hm', model_file='./models/last_hr.h5')

    # w300w.evaluate_on_300w(model_name='stu_tol', model_file='./models/300w/KD_main_correct/stu_tol.h5')
    # w300w.evaluate_on_300w(model_name='stu_tou', model_file='./models/300w/KD_main_correct/stu_tou.h5')
    #
    # w300w.evaluate_on_300w(model_name=models[0], model_file='./models/300w/stu_model_4_ibug.h5')

    # w300w.evaluate_on_300w(model_name=models[0], model_file='./models/300w/KD_main_correct/ds_300w_ef_100.h5')
    # w300w.evaluate_on_300w(model_name=models[1], model_file='./models/300w/KD_main_correct/ds_300w_mn_stu.h5')
    # w300w.evaluate_on_300w(model_name=models[2], model_file='./models/300w/KD_main_correct/ds_300w_mn_base.h5')

    # w300w.evaluate_on_300w(model_name=models[3], model_file='./models/300w/ASM/ASM_300W_model_6_40.h5')
    # w300w.evaluate_on_300w(model_name=models[3], model_file='./models/300w/ASM/ASM_300W_model_6_19.h5')

    # w300w.evaluate_on_300w(model_name='efn_base',
    #                        model_file='./models/ASM_32_300W_nme_8.138384697607048_fr_26.666666666666668.h5')
    # w300w.evaluate_on_300w(model_name='efn_fawl', model_file='./models/ASM_33_300W_nme_7.648649876718097_fr_22.962962962962962.h5')


    # mn-base:  ch->{nme:6.84, fr:7.40, AUC:0.5425}  co->{nme:3.93, fr:0.18, AUC:0.8102} full->{nme:4.50, fr:1.59, AUC:0.7578}
    # mn-stu:   ch->{nme:6.13, fr:3.70, AUC:0.6029}  co->{nme:3.56, fr:0.18, AUC:0.8356} full->{nme:4.06, fr:0.87, AUC:0.7900}

    # mn-tough:   ch->{nme:6.59 fr:5.92  AUC:0.5509}      co->{nme:3.85, fr:0.18  AUC:0.8146}  full->{nme:4.39 , fr:1.30 ,AUC:0.7630}
    # mn-tol:     ch->{nme:6.41 fr:4.44  AUC:0.5736}      co->{nme:3.73, fr:0.18  AUC:0.8230}  full->{nme:4.25 , fr:1.01 ,AUC:0.7742}
    # efn-100:    ch->{nme:5.80 fr:3.70  AUC:0.6423}   co->{nme: 3.34 , fr: 0.0,  AUC:0.8512}  full->{nme:3.82, fr: 0.72, AUC:0.8103}

    # mn-asm:         ch->{nme:6.19, fr: 2.96  AUC:0.6004}      co->{nme:3.70 , fr:0.0 AUC:0.8273}        full->{nme:4.19, fr: 0.58 , AUC: 0.7829}
    # mn-asm-ALW:     ch->{nme: , fr: AUC:}      co->{nme: , fr: AUC:}        full->{nme:3.90, fr:  AUC:}
    # mn-asm-WFD:     ch->{nme: , fr:  AUC:}      co->{nme: , fr: AUC:}        full->{nme:3.99, fr:  AUC:}

    '''wflw'''
    '''for this dataset, for evaluation part we DON'T use the Tf record, just we load the data and images'''
    wflw = WflwClass()  # todo DON'T FORGET to remove THE LOAD_DATA LINE LIMIT
    # wflw.create_test_set(need_pose=need_pose, need_tf_ref=False)
    # wflw.batch_test(weight_files_path='/media/data2/alip/kd_weights/wflw/24_jan_2021/', csv_file_path='./wflw_CSV_BATCH_RESULT.csv')

    wflw.create_train_set(need_pose=False, need_hm=False, accuracy=pca_accuracy)  #
    wflw.create_heatmap()
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

    '''KD'''
    # wflw.evaluate_on_wflw(model_name=models[0], model_file='./models/wflw/kd/ds_wflw_efn_100.h5') #ch: 12.96
    # wflw.evaluate_on_wflw(model_name=models[1], model_file='./models/wflw/kd/ds_wflw_stu_model_16.34.h5') #ch:17.74
    # wflw.evaluate_on_wflw(model_name=models[2], model_file='./models/wflw/kd/ds_wflw_mn.h5') #ch:

    '''ASM'''
    # wflw.evaluate_on_wflw(model_name='mn', model_file='./models/wflw/asm/ds_wflw_mn.h5') #ch: 16.06
    # wflw.evaluate_on_wflw(model_name='ASM', model_file='./models/asm_fw_model_3_wflw.h5')

    # wflw.evaluate_on_wflw(model_name='efn_base', model_file='./models/ASM_0_wflw_nme_16.956057100456377_fr_93.55828220858896.h5')
    # wflw.evaluate_on_wflw(model_name='efn_fawl', model_file='./models/ASM_40_wflw_nme_15.476098076653813_fr_88.34355828220859.h5')

    # wflw.evaluate_on_wflw(model_name='ASM', model_file='./models/wflw/asm/ASM_main.h5') # 14.74
    # wflw.evaluate_on_wflw(model_name='ALW', model_file='./models/wflw/asm/ASM_ALW.h5')  # 14.92
    # wflw.evaluate_on_wflw(model_name='WFD', model_file='./models/wflw/asm/ASM_WFD.h5')  # 15.64

    # wflw.hm_evaluate_on_wflw(model_name='hm', model_file='./models/IAL0_wflw_0.h5') #ch:
    # #
    #
