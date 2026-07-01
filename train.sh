###################################### DATA1 #############################################
# CUDA_VISIBLE_DEVICES="4" \
# python train.py \
# --hiera_path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/sam2_hiera_large.pt" \
# --train_image_path "/data1/Datasets/Seg/Polyp/TrainDataset/image/" \
# --train_mask_path "/data1/Datasets/Seg/LesionChange/Brisc2025change_all/masks/" \
# --save_path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/results_new_new/1_cross_1" \
# --epoch 300 \
# --lr 0.0001 \
# --batch_size 4
###################################### DATA2 #############################################
# CUDA_VISIBLE_DEVICES="5" \
# python train2.py \
# --hiera_path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/sam2_hiera_large.pt" \
# --train_image_path "/data1/Datasets/Seg/LesionChange/Dataset2/" \
# --train_mask_path "/data1/Datasets/Seg/LesionChange/Dataset2/masks/" \
# --save_path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/test/72.08_add_glcav2_adapter" \
# --epoch 300 \
# --lr 0.0001 \
# --batch_size 4
###################################### DATA3 #############################################
# CUDA_VISIBLE_DEVICES="3" \
# python train3.py \
# --hiera_path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/sam2_hiera_large.pt" \
# --train_image_path "/data1/Datasets/Seg/LesionChange/Dataset3/" \
# --train_mask_path "/data1/Datasets/Seg/LesionChange/Dataset3/masks/" \
# --save_path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/test/ours_mona_lgca_cat_loss" \
# --epoch 300 \
# --lr 0.0001 \
# --batch_size 4

###################################### DATA rib #############################################
# CUDA_VISIBLE_DEVICES="4" \
# python train3.py \
# --hiera_path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/sam2_hiera_large.pt" \
# --train_image_path "/data1/Datasets/Seg/LesionChange/Dataset3/" \
# --train_mask_path "/data1/Datasets/Seg/LesionChange/Dataset3/masks/" \
# --save_path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/result_rib/baseline" \
# --epoch 300 \
# --lr 0.0001 \
# --batch_size 4
###################################### RemoteChange #############################################
CUDA_VISIBLE_DEVICES="5" \
python train_rc.py \
--hiera_path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/sam2_hiera_large.pt" \
--train_image_path "/data1/Datasets/Seg/remotechange/All/" \
--train_mask_path "/data1/Datasets/Seg/remotechange/All/mask/" \
--save_path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/result_RC/72.08_cat_0.002re" \
--epoch 300 \
--lr 0.002 \
--batch_size 4





