import argparse
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms

from SAM2UNet import SAM2UNet
from misc.metric_tools import ConfuseMatrixMeter
import core.metrics as Metrics


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--test_image_path1", type=str, required=True)
parser.add_argument("--test_image_path2", type=str, required=True)
parser.add_argument("--test_gt_path", type=str, required=True)
parser.add_argument("--img_size", type=int, default=448)
args = parser.parse_args()

# ================== 初始化 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAM2UNet().to(device)
model.load_state_dict(torch.load(os.path.join(args.model_path, "best_model.pth")), strict=True)
model.eval()

metric = ConfuseMatrixMeter(n_class=2)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

save_path = os.path.join(args.model_path, "test", "best")
os.makedirs(save_path, exist_ok=True)

# ================== 推理 ==================
print('\n=============== Start Inference ===============')
all_preds = []
all_gts = []

for img_name in tqdm(os.listdir(args.test_image_path1)):

    img1_path = os.path.join(args.test_image_path1, img_name)
    img2_path = os.path.join(args.test_image_path2, img_name)
    gt_path = os.path.join(args.test_gt_path, img_name.replace('.png', '.png'))

    if not os.path.exists(img2_path) or not os.path.exists(gt_path):
        continue

    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    gt = cv2.imread(gt_path, 0)

    img1 = cv2.resize(img1, (args.img_size, args.img_size))
    img2 = cv2.resize(img2, (args.img_size, args.img_size))
    gt = cv2.resize(gt, (args.img_size, args.img_size))

    img1 = np.expand_dims(img1, -1).repeat(3, axis=-1)
    img2 = np.expand_dims(img2, -1).repeat(3, axis=-1)
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img1, img2)               # [1,2,H,W]
        pred_label = torch.argmax(pred, dim=1) # [1,H,W]

    pred_label = pred_label.squeeze().cpu().numpy()
    gt = (gt > 0).astype(np.uint8)  # GT 转二值

    metric.update_cm(pred_label, gt)
    all_preds.append(pred_label)
    all_gts.append(gt)

    # ✅ 保存可视化 mask
    save_mask = (pred_label * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path, img_name), save_mask)

print('\n=============== Inference Done ✅ ===============')

# ================== 指标总结 ==================
scores = metric.get_scores()

print("\n========= Final Results =========")
for k, v in scores.items():
    print(f"{k}: {v:.4f}")

# ✅ 保存指标
with open(os.path.join(save_path, "metrics.txt"), 'w') as f:
    for k, v in scores.items():
        f.write(f"{k}: {v:.6f}\n")

print("\n📌 Results saved to:", save_path)
print("✅ Testing Completed 🚀")
