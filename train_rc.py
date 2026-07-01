
# import os
# import argparse
# import random
# import csv
# import sys
# import time
# import numpy as np
# import torch
# import torch.optim as opt
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from dataset import UnetDataset  # 确保导入正确的数据集类
# from SAM2UNet import SAM2UNet
# import shutil
# from contextlib import contextmanager
# from tqdm import tqdm

# # -------------------- 命令行参数 --------------------
# parser = argparse.ArgumentParser("SAM2-UNet")
# parser.add_argument("--hiera_path", type=str, default="/data1/Code/zhaoxiaowei/SAM2-UNet-main/sam2_hiera_large.pt", 
#                     required=True, help="path to the sam2 pretrained hiera")
# parser.add_argument("--train_image_path", type=str, required=True,
#                     help="path to the image that used to train the model")
# parser.add_argument("--train_mask_path", type=str, required=True,
#                     help="path to the mask file for training")
# parser.add_argument('--save_path', type=str, required=True,
#                     help="path to store the checkpoint")
# parser.add_argument("--epoch", type=int, default=20,
#                     help="training epochs")
# parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
# parser.add_argument("--batch_size", default=12, type=int)
# parser.add_argument("--weight_decay", default=5e-4, type=float)
# parser.add_argument("--val_interval", type=int, default=1, 
#                     help="validation interval (epochs)")
# args = parser.parse_args()

# # -------------------- 损失函数与评估指标 --------------------
# def structure_loss(pred, mask):
#     """结构损失（结合加权BCE和加权IoU）"""
#     weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    
#     pred_sigmoid = torch.sigmoid(pred)
#     inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
#     union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1) / (union - inter + 1)
    
#     return (wbce + wiou).mean()

# def dice_coeff(pred, target, smooth=1e-5):
#     """二值化Dice（与测试一致）"""
#     pred = torch.sigmoid(pred)
    
#     # === 二值化（同 calmiou 阈值 125/255 ≈ 0.49）===
#     pred = (pred > 0.5).float()
#     target = (target > 0.5).float()

#     intersection = (pred * target).sum(dim=(2, 3))
#     union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

#     dice = (2. * intersection + smooth) / (union + smooth)
#     return dice.mean()  # 所有 batch 样本取均值

# def Dice_loss(inputs, targets, smooth=0.00001):
#     inputs = torch.sigmoid(inputs)

#     a, b, c, d = inputs.size()
#     sums = []

#     for i in range(a):
#         for j in range(b):
#             img = inputs[i, j, :, :]
#             label = targets[i, j, :, :]
#             intersection = (img * label).sum()
#             sums.append(1 - (2. * intersection + smooth) / (img.sum() + label.sum() + smooth))

#     return sum(sums) / len(sums)

# # -------------------- 工具函数 --------------------
# def get_lr(optimizer):
#     """获取当前学习率"""
#     for pg in optimizer.param_groups:
#         return pg["lr"]

# @contextmanager
# def dual_logger(log_path):
#     """同时输出日志到终端和文件"""
#     class Tee(object):
#         def __init__(self, *files):
#             self.files = files
#         def write(self, obj):
#             for f in self.files:
#                 f.write(obj)
#                 f.flush()
#         def flush(self):
#             for f in self.files:
#                 f.flush()

#     orig_stdout = sys.stdout
#     with open(log_path, "a", buffering=1) as f:
#         sys.stdout = Tee(sys.stdout, f)
#         try:
#             yield
#         finally:
#             sys.stdout = orig_stdout

# def prepare_loggers(save_dir):
#     """准备日志文件（文本日志和CSV）"""
#     os.makedirs(save_dir, exist_ok=True)
#     log_txt = os.path.join(save_dir, "train.log")
#     csv_path = os.path.join(save_dir, "metrics_history.csv")
    
#     csv_f = open(csv_path, "a", newline="")
#     csv_w = csv.writer(csv_f)
    
#     # 写入表头（包含训练和验证指标）
#     if os.stat(csv_path).st_size == 0:
#         csv_w.writerow([
#             "epoch", "mode", "loss", "dice", 
#             "dice_0", "dice_1", "dice_2", "lr"
#         ])
#         csv_f.flush()
    
#     return log_txt, csv_f, csv_w

# # -------------------- 训练与验证函数 --------------------
# def train_one_epoch(model, dataloader, optimizer, device, epoch):
#     """训练一个epoch"""
#     model.train()
#     total_loss = 0.0
#     total_dice = 0.0
#     total_dice_per_cls = [0.0, 0.0, 0.0]  # 三个类别的Dice
    
#     with tqdm(total=len(dataloader), desc=f"Train Epoch [{epoch+1}/{args.epoch}]", 
#               ncols=120, ascii=True) as pbar:
#         for batch in dataloader:
#             imgs, imgs1, pngs = batch  # 解包数据（根据实际数据集调整）
#             x1, x2, target = imgs.to(device), imgs1.to(device), pngs.to(device)
            
#             optimizer.zero_grad(set_to_none=True)
#             pred = model(x1, x2)  # 模型输出
            
#             # 计算损失
#             loss = Dice_loss(pred, target)
#             loss.backward()
#             optimizer.step()
            
#             # 计算Dice指标
#             dice = dice_coeff(pred, target).mean().item()
#             total_dice += dice
            
#             # 计算每个类别的Dice
#             for cls in range(3):
#                 dice_cls = dice_coeff(pred[:, cls:cls+1, ...], target[:, cls:cls+1, ...]).mean().item()
#                 total_dice_per_cls[cls] += dice_cls
            
#             total_loss += loss.item()
            
#             # 更新进度条
#             pbar.set_postfix({
#                 "loss": f"{loss.item():.4f}",
#                 "dice": f"{dice:.4f}",
#                 "lr": f"{get_lr(optimizer):.6f}"
#             })
#             pbar.update(1)
    
#     # 计算平均指标
#     avg_loss = total_loss / len(dataloader)
#     avg_dice = total_dice / len(dataloader)
#     avg_dice_per_cls = [d / len(dataloader) for d in total_dice_per_cls]
    
#     return avg_loss, avg_dice, avg_dice_per_cls

# def validate(model, dataloader, device, epoch):
#     """验证一个epoch"""
#     model.eval()
#     total_loss = 0.0
#     total_dice = 0.0
#     total_dice_per_cls = [0.0, 0.0, 0.0]
    
#     with torch.no_grad():  # 关闭梯度计算
#         with tqdm(total=len(dataloader), desc=f"Val Epoch [{epoch+1}/{args.epoch}]", 
#                   ncols=120, ascii=True) as pbar:
#             for batch in dataloader:
#                 imgs, imgs1, pngs = batch
#                 x1, x2, target = imgs.to(device), imgs1.to(device), pngs.to(device)
                
#                 pred = model(x1, x2)
                
#                 # 计算损失和指标
#                 loss = Dice_loss(pred, target).item()
#                 dice = dice_coeff(pred, target).mean().item()
                
#                 total_loss += loss
#                 total_dice += dice
                
#                 # 每个类别的Dice
#                 for cls in range(3):
#                     dice_cls = dice_coeff(pred[:, cls:cls+1, ...], target[:, cls:cls+1, ...]).mean().item()
#                     total_dice_per_cls[cls] += dice_cls
                
#                 pbar.set_postfix({"loss": f"{loss:.4f}", "dice": f"{dice:.4f}"})
#                 pbar.update(1)
    
#     # 计算平均指标
#     avg_loss = total_loss / len(dataloader)
#     avg_dice = total_dice / len(dataloader)
#     avg_dice_per_cls = [d / len(dataloader) for d in total_dice_per_cls]
    
#     return avg_loss, avg_dice, avg_dice_per_cls

# # -------------------- 主函数 --------------------
# def main(args):
#     # 设备配置
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # 数据加载
#     data_path = '/data1/Datasets/Seg/remotechange/All/'
#     with open(os.path.join(data_path, "train.txt"), "r") as f:
#         train_lines = f.readlines()
#     with open(os.path.join(data_path, "test.txt"), "r") as f:
#         val_lines = f.readlines()
    
#     image_path = '/data1/Datasets/Seg/remotechange/All/'
#     train_dataset = UnetDataset(train_lines, (448, 448), 3, True, image_path)
#     val_dataset = UnetDataset(val_lines, (448, 448), 3, False, image_path)
    
#     dataloader = DataLoader(
#         train_dataset, 
#         batch_size=args.batch_size, 
#         shuffle=True, 
#         num_workers=8, 
#         pin_memory=True,
#         drop_last=True  # 防止最后一个批次尺寸不一致
#     )
#     dataloaderval = DataLoader(
#         val_dataset, 
#         batch_size=args.batch_size, 
#         shuffle=False,  # 验证集不打乱
#         num_workers=8, 
#         pin_memory=True
#     )

#     # 模型初始化
#     model = SAM2UNet(args.hiera_path).to(device)
    
#     # 优化器与调度器
#     optimizer = opt.AdamW(
#         [{"params": model.parameters(), "initial_lr": args.lr}],
#         lr=args.lr, 
#         weight_decay=args.weight_decay
#     )
#     scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-7)

#     # 保存目录与日志准备
#     os.makedirs(args.save_path, exist_ok=True)
#     try:
#         # 备份关键代码
#         shutil.copy('SAM2UNet.py', os.path.join(args.save_path, 'SAM2UNet.py'))
#         shutil.copy('train.py', os.path.join(args.save_path, 'train.py'))
#         shutil.copy('train.sh', os.path.join(args.save_path, 'train.sh'))
#     except Exception as e:
#         print(f"[Warning] 拷贝配置文件失败: {e}")
    
#     log_txt, csv_f, csv_w = prepare_loggers(args.save_path)

#     # 训练主循环
#     best_val_dice = 0.0  # 记录最佳验证Dice
#     start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
#     with dual_logger(log_txt):
#         print(f"==== 训练开始 @ {start_time} ====")
#         print(f"保存路径: {args.save_path}")
#         print(f"总轮数: {args.epoch}, 批次大小: {args.batch_size}, 初始学习率: {args.lr}")
#         print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

#         try:
#             for epoch in range(args.epoch):
#                 # 训练阶段
#                 train_loss, train_dice, train_dice_cls = train_one_epoch(
#                     model, dataloader, optimizer, device, epoch
#                 )
                
#                 # 记录训练指标
#                 csv_w.writerow([
#                     epoch + 1, "train",
#                     f"{train_loss:.6f}", f"{train_dice:.6f}",
#                     f"{train_dice_cls[0]:.6f}", f"{train_dice_cls[1]:.6f}", f"{train_dice_cls[2]:.6f}",
#                     f"{get_lr(optimizer):.8f}"
#                 ])
#                 csv_f.flush()

#                 # 验证阶段（每args.val_interval轮执行一次）
#                 if (epoch + 1) % args.val_interval == 0:
#                     val_loss, val_dice, val_dice_cls = validate(
#                         model, dataloaderval, device, epoch
#                     )
                    
#                     # 记录验证指标
#                     csv_w.writerow([
#                         epoch + 1, "val",
#                         f"{val_loss:.6f}", f"{val_dice:.6f}",
#                         f"{val_dice_cls[0]:.6f}", f"{val_dice_cls[1]:.6f}", f"{val_dice_cls[2]:.6f}",
#                         "-"  # 验证阶段无学习率
#                     ])
#                     csv_f.flush()

#                     # 打印 epoch 总结
#                     print(f"\n[Epoch {epoch+1}/{args.epoch}] 训练: "
#                           f"loss={train_loss:.4f}, dice={train_dice:.4f} "
#                           f"(cls0={train_dice_cls[0]:.4f}, cls1={train_dice_cls[1]:.4f}, cls2={train_dice_cls[2]:.4f})")
#                     print(f"[Epoch {epoch+1}/{args.epoch}] 验证: "
#                           f"loss={val_loss:.4f}, dice={val_dice:.4f} "
#                           f"(cls0={val_dice_cls[0]:.4f}, cls1={val_dice_cls[1]:.4f}, cls2={val_dice_cls[2]:.4f})")

#                     # 保存最佳模型
#                     if val_dice > best_val_dice:
#                         best_val_dice = val_dice
#                         best_ckpt_path = os.path.join(args.save_path, "best_model.pth")
#                         torch.save(model.state_dict(), best_ckpt_path)
#                         print(f"[保存最佳模型] Dice: {best_val_dice:.4f} -> {best_ckpt_path}")

#                 else:
#                     # 仅打印训练总结
#                     print(f"\n[Epoch {epoch+1}/{args.epoch}] 训练: "
#                           f"loss={train_loss:.4f}, dice={train_dice:.4f} "
#                           f"(cls0={train_dice_cls[0]:.4f}, cls1={train_dice_cls[1]:.4f}, cls2={train_dice_cls[2]:.4f})")

#                 # 学习率调度
#                 scheduler.step()

#                 # 定期保存模型快照
#                 if (epoch + 1) % 100 == 0 or (epoch + 1) == args.epoch:
#                     ckpt_path = os.path.join(args.save_path, f"epoch_{epoch+1}.pth")
#                     torch.save(model.state_dict(), ckpt_path)
#                     print(f"[保存快照] {ckpt_path}")

#         except KeyboardInterrupt:
#             print("\n[中断] 捕获到键盘中断，正在保存当前模型...")
#             torch.save(model.state_dict(), os.path.join(args.save_path, "interrupted_model.pth"))
#         finally:
#             csv_f.close()
#             end_time = time.strftime("%Y-%m-%d %H:%M:%S")
#             print(f"==== 训练结束 @ {end_time} ====")
#             print(f"最佳验证Dice: {best_val_dice:.4f}")


# if __name__ == "__main__":
#     main(args)
import shutil
import os
import argparse
import csv
import sys
import time
import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from contextlib import contextmanager
from tqdm import tqdm

from dataset import UnetDataset
from SAM2UNet import SAM2UNet

# ✅ CE + Dice
from misc.metric_tools import ConfuseMatrixMeter
from models.loss import ce_dice as loss_fn


# -------------------- Arguments --------------------
parser = argparse.ArgumentParser("SAM2-UNet Train")
parser.add_argument("--hiera_path", type=str, required=True)
parser.add_argument("--train_image_path", type=str, required=True)
parser.add_argument("--train_mask_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)

parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", default=4, type=int)

parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--val_interval", type=int, default=1)

args = parser.parse_args()


# -------------------- Utils --------------------
def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def _prepare_targets(mask):
    """mask: [B,H,W] or [B,1,H,W] or [B,2,H,W] → [B,H,W] long"""
    if mask.dim() == 4 and mask.size(1) == 2:
        mask = mask.argmax(dim=1)
    elif mask.dim() == 4 and mask.size(1) == 1:
        mask = mask[:, 0, ...]

    if mask.max() > 1:
        mask = (mask > 0).long()
    return mask.long()


@contextmanager
def dual_logger(log_path):
    """同时输出打印到文件与控制台"""
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files: f.flush()

    orig = sys.stdout
    with open(log_path, "a", buffering=1) as f:
        sys.stdout = Tee(sys.stdout, f)
        try:
            yield
        finally:
            sys.stdout = orig


def prepare_loggers(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_txt = os.path.join(save_dir, "train.log")
    csv_path = os.path.join(save_dir, "metrics.csv")

    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if os.stat(csv_path).st_size == 0:
        csv_w.writerow(["epoch", "phase", "loss", "mF1", "lr"])
        csv_f.flush()
    return log_txt, csv_f, csv_w


# -------------------- Train & Val --------------------
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    metric = ConfuseMatrixMeter(n_class=2)

    total_loss = total_mF1 = 0.0
    pbar = tqdm(dataloader, desc=f"Train[{epoch+1}/{args.epoch}]", ncols=120)

    for A, B, L in pbar:
        A, B = A.to(device), B.to(device)
        L = _prepare_targets(L.to(device))

        optimizer.zero_grad()
        pred = model(A, B)
      

        loss = loss_fn(pred, L)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_label = pred.argmax(dim=1)
            metric.update_cm(pred_label.cpu().numpy(), L.cpu().numpy())
            mf1 = metric.get_scores()["mf1"]

        total_loss += loss.item()
        total_mF1 += mf1

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         mF1=f"{mf1:.4f}",
                         lr=f"{get_lr(optimizer):.6f}")

    return total_loss / len(dataloader), total_mF1 / len(dataloader)


def validate(model, dataloader, device, epoch):
    model.eval()
    metric = ConfuseMatrixMeter(n_class=2)

    total_loss = total_mF1 = 0.0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Val[{epoch+1}/{args.epoch}]", ncols=120)
        for A, B, L in pbar:
            A, B = A.to(device), B.to(device)
            L = _prepare_targets(L.to(device))

            pred = model(A, B)
       

            loss = loss_fn(pred, L)

            pred_label = pred.argmax(dim=1)
            metric.update_cm(pred_label.cpu().numpy(), L.cpu().numpy())
            mf1 = metric.get_scores()["mf1"]

            total_loss += loss.item()
            total_mF1 += mf1

            pbar.set_postfix(loss=f"{loss:.4f}", mF1=f"{mf1:.4f}")

    return total_loss / len(dataloader), total_mF1 / len(dataloader)


# -------------------- Main --------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ dataset original behavior preserved
    data_path = "/data1/Datasets/Seg/remotechange/All/"
    train_lines = open(os.path.join(data_path, "train.txt")).readlines()
    val_lines = open(os.path.join(data_path, "test.txt")).readlines()

    train_dataset = UnetDataset(train_lines, (448, 448), 3, True, data_path)
    val_dataset = UnetDataset(val_lines, (448, 448), 3, False, data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=8)

    model = SAM2UNet(args.hiera_path).to(device)
    optimizer = opt.AdamW(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-7)

    os.makedirs(args.save_path, exist_ok=True)
    try:
        # 备份关键代码
        shutil.copy('SAM2UNet.py', os.path.join(args.save_path, 'SAM2UNet.py'))
        shutil.copy('train_rc.py', os.path.join(args.save_path, 'train_rc.py'))
        shutil.copy('train.sh', os.path.join(args.save_path, 'train.sh'))
    except Exception as e:
        print(f"[Warning] 拷贝配置文件失败: {e}")
    # ✅ Backup key files
    for src in ["train_rc.py", "SAM2UNet.py"]:
        try:
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.save_path, src))
        except:
            pass

    log_txt, csv_f, csv_w = prepare_loggers(args.save_path)

    best_mF1 = 0.0
    start = time.strftime("%Y-%m-%d %H:%M:%S")

    with dual_logger(log_txt):
        print(f"==== Train Start @ {start} ====")

        for epoch in range(args.epoch):
            train_loss, train_mF1 = train_one_epoch(model, train_loader, optimizer, device, epoch)
            csv_w.writerow([epoch+1, "train", train_loss, train_mF1, get_lr(optimizer)])
            csv_f.flush()

            print(f"[Train] E{epoch+1}: Loss={train_loss:.4f}, mF1={train_mF1:.4f}")

            if (epoch + 1) % args.val_interval == 0:
                val_loss, val_mF1 = validate(model, val_loader, device, epoch)
                csv_w.writerow([epoch+1, "val", val_loss, val_mF1, "-"])
                csv_f.flush()

                print(f"[Val]   E{epoch+1}: Loss={val_loss:.4f}, mF1={val_mF1:.4f}")

                if val_mF1 > best_mF1:
                    best_mF1 = val_mF1
                    torch.save(model.state_dict(), os.path.join(args.save_path, "best_model.pth"))
                    print(f"✅ Best Model Updated: mF1={best_mF1:.4f}")

            scheduler.step()

        csv_f.close()
        print(f"==== Train End | Best mF1: {best_mF1:.4f} ====")


if __name__ == "__main__":
    main(args)
