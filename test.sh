CUDA_VISIBLE_DEVICES="5" \
python test.py \
--checkpoint "/data1/Code/zhaoxiaowei/SAM2-UNet-main/results/diff_re/SAM2-UNet-20.pth" \
--test_image_path "/data1/Datasets/Seg/Polyp/TestDataset/Kvasir/images/" \
--test_gt_path "/data1/Datasets/Seg/Polyp/TestDataset/Kvasir/masks/" \
--save_path "/data1/Code/zhaoxiaowei/SAM2-UNet-main/results/diff_re/test"