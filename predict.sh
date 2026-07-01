# CUDA_VISIBLE_DEVICES="5" \
# python predict.py \
# --checkpoint "/data1/Code/zhaoxiaowei/SAM2-UNet-main/results2/0_baseline_1/epoch_300.pth" \
# --test_image_path1 "/data1/Datasets/Seg/LesionChange/Brisc2025change_all/test1/" \
# --test_image_path2 "/data1/Datasets/Seg/LesionChange/Brisc2025change_all/test2/" \
# --test_gt_path "/data1/Datasets/Seg/LesionChange/Brisc2025change_all/masks/" \
# --save_path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/test/1_diffadapter_3_xy/test——300"



# CUDA_VISIBLE_DEVICES="4" \
# python predict.py \
# --path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/results_3/ours_mona_lgca_cat_loss" \
# --test_image_path1 "/data1/Datasets/Seg/LesionChange/Dataset2/test0/" \
# --test_image_path2 "/data1/Datasets/Seg/LesionChange/Dataset2/test1/" \
# --test_gt_path "/data1/Datasets/Seg/LesionChange/Dataset2/mask/" \
 
#3
 
CUDA_VISIBLE_DEVICES="5" \
python predict.py \
--path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/results_3/ours_mona_lgca_cat_loss" \
--test_image_path1 "/data1/Datasets/Seg/LesionChange/Dataset3/test0/" \
--test_image_path2 "/data1/Datasets/Seg/LesionChange/Dataset3/test1/" \
--test_gt_path "/data1/Datasets/Seg/LesionChange/Dataset3/mask/" \
 