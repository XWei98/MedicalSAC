import argparse
import os
import torch
import imageio
import numpy as np
import cv2
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms

from SAM2UNet import SAM2UNet
from dataset import TestDataset
 

def calmiou(imgdir1, imgdir2):  # miou
    miou = 0
    mdsc = 0
    for img in os.listdir(imgdir1):
        imgpath1 = os.path.join(imgdir1, img)
        label_name = os.path.splitext(img)[0] + '.png'
        imgpath2 = os.path.join(imgdir2, label_name)
        img1 = cv2.imread(imgpath1, 0)
        img2 = cv2.imread(imgpath2, 0)
        img1 = cv2.resize(img1, (448, 448))
        img2 = cv2.resize(img2, (448, 448))
        img1[img1 <= 125] = 0
        img1[img1 > 125] = 1
        img2[img2 <= 125] = 0
        img2[img2 > 125] = 1
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        img3 = cv2.bitwise_and(img1, img2)
        img4 = cv2.bitwise_or(img1, img2)
        iou = img3.ravel().sum() / img4.ravel().sum() if img4.ravel().sum() != 0 else 0
        miou = miou + iou

        dsc = 2 * img3.ravel().sum() / (img4.ravel().sum() + img3.ravel().sum()) if (img4.ravel().sum() + img3.ravel().sum()) != 0 else 0
        mdsc = mdsc + dsc

    return miou / len(os.listdir(imgdir1)), mdsc / len(os.listdir(imgdir1))
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--test_image_path1", type=str, required=True)
parser.add_argument("--test_image_path2", type=str, required=True)
parser.add_argument("--test_gt_path", type=str, required=True)
parser.add_argument("--miou_save", type=str, default="miou.txt")
parser.add_argument("--mdsc_save", type=str, default="mdsc.txt")
parser.add_argument("--img_size", type=int, default=448)
args = parser.parse_args()

# ================== 初始化 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAM2UNet().to(device)
model.load_state_dict(torch.load(args.path+"/best_model.pth"), strict=True)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 适配你的模型训练时的归一化策略
])

test_image_dir1 = args.test_image_path1
test_image_dir2 = args.test_image_path2
save_path = args.path + "/test/best"
os.makedirs(save_path, exist_ok=True)

# ================== 推理 ==================
print('\n=============== Start Inference ===============')
for img_name in tqdm(os.listdir(test_image_dir1)):
    img1_path = os.path.join(test_image_dir1, img_name)
    img2_path = os.path.join(test_image_dir2, img_name)

    if not os.path.exists(img2_path):
        continue

    # 加载图像
    img1 = cv2.imread(img1_path, 0)  # t0
    img2 = cv2.imread(img2_path, 0)  # t1

    # Resize & 3通道 & Normalize
    img1 = cv2.resize(img1, (args.img_size, args.img_size))
    img2 = cv2.resize(img2, (args.img_size, args.img_size))

    img1 = np.expand_dims(img1, -1).repeat(3, axis=-1)
    img2 = np.expand_dims(img2, -1).repeat(3, axis=-1)

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        pred = model(img1, img2)  # [1, 3, H, W]
        pred = torch.sigmoid(pred).cpu().numpy()[0]  # [3, H, W]

    # 每一类分别保存
    for i in range(3):
        class_dir = os.path.join(save_path, str(i))
        os.makedirs(class_dir, exist_ok=True)

        pred_mask = pred[i] * 255
        pred_mask[pred_mask <= 127] = 0
        pred_mask[pred_mask > 127] = 255
        pred_mask = pred_mask.astype(np.uint8)

        save_img_path = os.path.join(class_dir, img_name)
        cv2.imwrite(save_img_path, pred_mask)

print('\n=============== Predicted Masks Saved! ===============')

# ================== 指标计算 ==================
print('\n=============== Compute Metrics ===============')

classes = ['0', '1', '2']
class_names = ['growth', 'reduction', 'unchange']
iou_list, dsc_list = [], []
iou_str_list, dsc_str_list = [], []

for idx, cls in enumerate(tqdm(classes)):
    pred_cls_path = os.path.join(save_path, cls)
    gt_cls_path = os.path.join(args.test_gt_path, cls)  # 注意你 test_gt_path 要是 mask 根目录！

    if not os.path.exists(gt_cls_path):
        print(f"[Warning] GT folder missing: {gt_cls_path}")
        continue

    miou, mdsc = calmiou(pred_cls_path, gt_cls_path)
    iou_list.append(miou)
    dsc_list.append(mdsc)

    iou_str_list.append(f'{class_names[idx]}（标签{cls}）的mIoU: {miou:.4f}')
    dsc_str_list.append(f'{class_names[idx]}（标签{cls}）的mDSC: {mdsc:.4f}')

mean_iou = sum(iou_list) / len(iou_list)
mean_dsc = sum(dsc_list) / len(dsc_list)

# 输出并保存
miou_result = '\n'.join(iou_str_list) + f'\n\n三类平均 mIoU: {mean_iou:.4f}'
mdsc_result = '\n'.join(dsc_str_list) + f'\n\n三类平均 mDSC: {mean_dsc:.4f}'

print('\n===== MIoU =====')
print(miou_result)
print('\n===== DSC =====')
print(mdsc_result)
miou_save = save_path + '/pred_miou.txt'
mdsc_save = save_path + '/pred_mdsc.txt'

with open(miou_save, 'w') as f:
    f.write(miou_result)
with open(mdsc_save, 'w') as f:
    f.write(mdsc_result)
