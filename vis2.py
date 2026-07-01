import cv2
import os
import numpy as np
from tqdm import tqdm

base_path = "/data1/Code/zhaoxiaowei/SAM2-UNet-main/results_2/0_baseline_1"
pred_image_root = os.path.join(base_path, "test——best/")  # 3个类别的预测图根目录
test2_img_root = "/data1/Datasets/Seg/LesionChange/Dataset2/test1/"  # 叠加用的原图目录

# 3个类别配置：key=类别名称，value=(类别标签索引, 单个类别可视化文件夹名)
class_config = {
    "growth": (1, "growth_vis"),    # 类别1：标签索引0，文件夹名growth_vis
    "reduction": (0, "reduction_vis"),  # 类别2：标签索引1，文件夹名reduction_vis
    "unchange": (2, "unchange_vis")   # 类别3：标签索引2，文件夹名unchange_vis
}

# 新增：合并结果配置（3个类别合并到一张图）
merge_config = {
    "merge_name": "three_class_merge",  # 合并结果文件夹名
    "merge_title": "all"  # 合并图说明（用于文件名）
}

# 目标文件夹：3个单个类别文件夹 + 1个合并结果文件夹
target_base_dir = os.path.join(base_path, "best/all")
target_dirs = {
    # 3个单个类别文件夹
    **{cls_name: os.path.join(target_base_dir, folder_name) 
       for cls_name, (_, folder_name) in class_config.items()},
    # 1个合并结果文件夹
    "merge": os.path.join(target_base_dir, merge_config["merge_name"])
}

# 颜色配置：3个类别专属浅色调 + 合并时保持一致
class_colors = {
    "growth":  (0, 0, 255),    # 浅红色（growth）
    "reduction": (0, 255, 0),  # 浅绿色（reduction）
    "unchange": (255, 0, 0)    # 浅蓝色（unchange）
}


# ========================= 2. 创建目标文件夹（4个：3单+1合并）=========================
for target_dir in target_dirs.values():
    os.makedirs(target_dir, exist_ok=True)  # 自动创建多级目录
print(f"已创建4个目标文件夹：")
for name, path in target_dirs.items():
    print(f"- {name}: {path}")


# ========================= 3. 核心可视化逻辑（单类别+合并结果）=========================
# 遍历所有测试图像（以test2目录的图像名为基准）
for imgname in tqdm(os.listdir(test2_img_root), desc="处理图像（含合并结果）"):
    # 跳过非图像文件（避免隐藏文件/错误格式）
    if not imgname.endswith((".png", ".jpg", ".jpeg")):
        continue
    
    # 读取待叠加的原图（统一转为RGB格式）
    origin_img_path = os.path.join(test2_img_root, imgname)
    origin_img = cv2.imread(origin_img_path, 0)  # 灰度读取
    origin_img = cv2.resize(origin_img, (448, 448))  # 统一尺寸448x448
    origin_img_rgb = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2BGR)  # 灰度→RGB


    # ---------------------- 3.1 生成3个类别的单独可视化结果 ----------------------
    # 存储每个类别的二值掩码（用于后续合并）
    cls_bin_masks = {}
    for cls_name, (cls_idx, folder_name) in class_config.items():
        # 读取预测图并二值化
        imgname = os.path.splitext(imgname)[0] + '.png'
        pred_img_path = os.path.join(pred_image_root, str(cls_idx), imgname)
        if not os.path.exists(pred_img_path):
            print(f"警告：{cls_name}类预测图 {pred_img_path} 不存在，跳过该图像")
            cls_bin_masks[cls_name] = None  # 标记为缺失
            continue
        
        pred_img = cv2.imread(pred_img_path, 0)
        pred_img = cv2.resize(pred_img, (448, 448))
        pred_bin = np.where(pred_img > 125, 1, 0).astype(np.uint8)  # 二值化
        cls_bin_masks[cls_name] = pred_bin  # 保存二值掩码

        # 生成单个类别的彩色掩码
        cls_color = class_colors[cls_name]
        color_mask = np.zeros((448, 448, 3), dtype=np.uint8)
        color_mask[pred_bin == 1] = cls_color

        # 保存单个类别的2种结果：纯掩码 + 掩码+原图叠加
        target_dir = target_dirs[cls_name]
        # 纯彩色掩码
        cv2.imwrite(os.path.join(target_dir, f"{imgname.split('.')[0]}_pure.png"), color_mask)
        # 掩码+原图叠加（半透明融合）
        overlay_img = cv2.addWeighted(color_mask, 0.5, origin_img_rgb, 0.7, 0)
        cv2.imwrite(os.path.join(target_dir, f"{imgname.split('.')[0]}_overlay.png"), overlay_img)


    # ---------------------- 3.2 新增：生成3个类别的合并可视化结果 ----------------------
    # 检查是否所有类别掩码都存在（避免合并缺失数据）
    if all(mask is not None for mask in cls_bin_masks.values()):
        # 1. 生成纯合并掩码图（3个类别颜色叠加，无原图）
        merge_pure_mask = np.zeros((448, 448, 3), dtype=np.uint8)
        for cls_name, pred_bin in cls_bin_masks.items():
            cls_color = class_colors[cls_name]
            merge_pure_mask[pred_bin == 1] = cls_color  # 不同类别填充对应颜色

        # 2. 生成合并掩码+原图叠加图
        merge_overlay_img = cv2.addWeighted(merge_pure_mask, 0.5, origin_img_rgb, 0.9, 0)

        # 3. 保存合并结果到专属文件夹
        merge_target_dir = target_dirs["merge"]
        # 纯合并掩码（标注类别颜色说明）
        merge_pure_save_name = f"{imgname.split('.')[0]}_merge_pure_{merge_config['merge_title']}.png"
        cv2.imwrite(os.path.join(merge_target_dir, merge_pure_save_name), merge_pure_mask)
        # 合并掩码+原图叠加（标注类别颜色说明）
        merge_overlay_save_name = f"{imgname.split('.')[0]}_merge_overlay_{merge_config['merge_title']}.png"
        cv2.imwrite(os.path.join(merge_target_dir, merge_overlay_save_name), merge_overlay_img)


# ========================= 4. 完成提示=========================
print(f"\n可视化全部完成！结果分布：")
for cls_name in class_config.keys():
    print(f"- {cls_name}类单独结果：{target_dirs[cls_name]}（含纯掩码+原图叠加）")
print(f"- 3类别合并结果：{target_dirs['merge']}（含纯合并掩码+合并掩码+原图叠加）")