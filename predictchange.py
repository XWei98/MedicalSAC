import os
import cv2
import torch
import numpy as np
import scipy
from scipy.spatial.distance import directed_hausdorff, cdist

from torch import nn
from tqdm import tqdm
from torchvision import transforms
import argparse

# 导入模型（根据实际项目路径调整）
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import argparse
from networks.vmunet import VMUNet
from torch.utils.data import DataLoader
# from nets.doubelunet import doubleunet1
# from nets.doubelunet import unet
# from nets.doubelunet import doubleunet
from networks.vit_seg_modeling_sam import VisionTransformer as ViT_segsam
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling_os import VisionTransformer as osViT_seg
from networks.vit_seg_modeling_os import CONFIGS as osCONFIGS_ViT_seg
import shutil
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch
import os
from networks.vit_seg_modeling_uctransunet import UCTransNet as UCTransNet
import networks.ucconfig as ucconfig
from networks.vit_seg_modeling_unet import UNet
from networks.vit_seg_modeling_unetplus import UnetPulsPuls as UNetPuls
from networks.vit_seg_modeling_attunet import AttU_Net as AttU_Net
from networks.vit_seg_modeling_uctransunet import UCTransNet as UCTransNet
# from networks.vit_seg_modeling_unext import UNext as UNext
from networks.Swinunet import SwinTransformerSys as swinunet
from networks.swinunetr import SwinUNETR as swinunetr
from networks.swinunetr import SwinUNETR as swinunetr
import networks.ucconfig as ucconfig
 


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


 
if __name__ == "__main__":
    num_classes = 3
    backbone = 'resnet50'
    input_shape = (448, 448)
    # image_dir = '/media/20TB/madexin3/allnewimg'
    img_dir1 = '/data1/Datasets/Seg/LesionChange/Brisc2025change_all/test1/'  # 第一张图目录
    img_dir2 = '/data1/Datasets/Seg/LesionChange/Brisc2025change_all/test2/'  # 第二张图目录
 
    save_pdir = '/data1/Code/zhaoxiaowei/MedChange/result/Transunet/Add_3cls_dice/ModelLabel2025_10_09_11_18'
    modelsort = "/best"
    model_path = save_pdir + modelsort +'_epoch_weights.pth'


    pred_save_path = save_pdir + modelsort + '/pred_image'
    miou_save = save_pdir + modelsort + '/pred_miou.txt'
    mdsc_save = save_pdir + modelsort + '/pred_mdice.txt'
    macc_save = save_pdir + modelsort + '/pred_macc.txt'
    mspec_save = save_pdir + modelsort + '/pred_mspec.txt'
    msen_save = save_pdir + modelsort + '/pred_msen.txt'
    mhd_save = save_pdir + modelsort + '/pred_mhd.txt'
    massd_save = save_pdir + modelsort + '/pred_assd.txt'

    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将图像变成PIL格式    输入为[H, W, C]输出为[H, W, C]
        transforms.ToTensor(),  # 将PIL图像转换为tensor    输入为[H, W, C]输出为[C, H, W]
    ])

    parser = argparse.ArgumentParser()
 
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')  # 突触
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=3, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=448, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    args = parser.parse_args()
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    config_vit.n_patches = int(args.img_size / args.vit_patches_size) * int(args.img_size / args.vit_patches_size)
    config_vit.h = int(args.img_size / args.vit_patches_size)
    config_vit.w = int(args.img_size / args.vit_patches_size)
    ##   UCTRANSUNET   #################################################################################
    unet = ViT_seg(config_vit, img_size=448, num_classes=num_classes)



    #  ##   UCTRANSUNET   #################################################################################
    # ucconfig_vit = ucconfig.get_CTranS_config()
    # unet = UCTransNet(ucconfig_vit, n_channels=ucconfig.n_channels, n_classes=num_classes)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet.load_state_dict(torch.load(model_path, map_location=device), strict=False)  # 加载模型参数
    unet = unet.eval()  # 测试模式
    unet = nn.DataParallel(unet)
    unet = unet.cuda()

    print('==============================================Predicted Image Save!==============================================')
    for img in tqdm(os.listdir(img_dir1)):
        imgpath = os.path.join(img_dir1, img)
        imgpath2 = os.path.join(img_dir2, img)
        image = cv2.imread(imgpath, 0)
 
        image = cv2.resize(image, (448, 448))
        image2 = cv2.imread(imgpath2, 0)
        image2 = cv2.resize(image2, (448, 448))
        # 无归一化使用
        '''
        image = np.expand_dims(image, 0).repeat(3, axis=0)    # [3, 448, 448]
        image = np.expand_dims(image, 0)                      # [b, 3, 448, 448]
        image = torch.from_numpy(image).type(torch.FloatTensor)
        '''

        # 归一化使用
        image = np.expand_dims(image, -1).repeat(3, axis=-1)  # [448, 448, 3]
        image = transform(image)  # [3, 448, 448]
        image = image.unsqueeze(0)  # [b, 3, 448, 448]
        # 归一化使用
        image2 = np.expand_dims(image2, -1).repeat(3, axis=-1)  # [448, 448, 3]
        image2 = transform(image2)  # [3, 448, 448]
        image2 = image2.unsqueeze(0)  # [b, 3, 448, 448]


        pred = unet(image,image2)  # [b, num_classes, h, w]
        pred = torch.sigmoid(pred)
        pred = pred.detach().cpu().numpy()
        # pred = t_crf(image.cpu().numpy(), pred)     # 后处理CRF
        print(pred.shape)
        for i in range(num_classes):
            save_path = os.path.join(pred_save_path, str(i))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            savepath = os.path.join(save_path, img)  # 拼接存储路径

            pred_image = pred[0, i, :, :]
            pred_image = pred_image * 255
            pred_image[pred_image <= 127] = 0
            pred_image[pred_image > 127] = 255
            pred_image = pred_image.astype(np.uint8)
            cv2.imwrite(savepath, pred_image)

print('==================================================Compute Metrics==================================================')
# 三个类别的标签列表（根据实际类别调整标签值）
classes = ['0', '1', '2']  # 假设三个类别分别为0、1、2
class_names = ['growth', 'reduction', 'unchange']  # 三类的名称（可自定义）

# 初始化指标列表
iou_list = []
dsc_list = []
iou_str_list = []
dsc_str_list = []

# 遍历每个类别计算指标
for idx, cls in enumerate(tqdm(classes)):
    imgpath = os.path.join(pred_save_path, cls)
    # 标签路径（根据实际路径修改）
    labelpath = os.path.join('/data1/Datasets/Seg/LesionChange/Brisc2025change_all/masks/', cls)
    
    # 计算miou和mdsc（假设calmiou返回这两个值）
    miou, mdsc = calmiou(imgpath, labelpath)
    
    # 存储单个类别指标
    iou_list.append(miou)
    dsc_list.append(mdsc)
    
    # 生成单个类别指标字符串
    iou_str = f'{class_names[idx]}（标签{cls}）的miou: {miou:.4f}'
    dsc_str = f'{class_names[idx]}（标签{cls}）的mdsc: {mdsc:.4f}'
    iou_str_list.append(iou_str)
    dsc_str_list.append(dsc_str)

# 计算三类的平均指标
mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
mean_dsc = sum(dsc_list) / len(dsc_list) if dsc_list else 0.0

# 构建结果字符串（包含单个指标和平均指标）
miou_result = '\n'.join(iou_str_list) + f'\n\n三类平均miou: {mean_iou:.4f}'
dsc_result = '\n'.join(dsc_str_list) + f'\n\n三类平均mdsc: {mean_dsc:.4f}'

# 打印结果
print('\n===== MIoU 指标 =====')
print(miou_result)
print('\n===== DSC 指标 =====')
print(dsc_result)

# 保存结果到文件
with open(miou_save, 'w') as f:
    f.write(miou_result)
with open(mdsc_save, 'w') as f:
    f.write(dsc_result)



