import os
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy

opt = option().parse_args()



class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Call after every optimizer.step() to update shadow weights."""
        for ema_p, live_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(live_p.data, alpha=1.0 - self.decay)
        
        for ema_b, live_b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(live_b.data)

    def get_model(self):
        return self.ema_model



def fft_loss(output, target):
    
    fft_out = torch.fft.rfft2(output, norm='ortho')
    fft_gt  = torch.fft.rfft2(target, norm='ortho')
    
    loss = torch.mean(torch.abs(fft_out.abs() - fft_gt.abs()))
    return loss



def hvi_chroma_loss(output_hvi, gt_hvi, chroma_weight=2.0):
    
    H_loss = torch.mean(torch.abs(output_hvi[:, 0] - gt_hvi[:, 0]))  
    V_loss = torch.mean(torch.abs(output_hvi[:, 1] - gt_hvi[:, 1]))  
    I_loss = torch.mean(torch.abs(output_hvi[:, 2] - gt_hvi[:, 2]))  
    return chroma_weight * (H_loss + V_loss) + I_loss



def noise_consistency_loss(model, input_rgb, output_rgb, margin=0.1):
    
    noise_on_input  = model.get_noise_map(input_rgb.detach())   
    noise_on_output = model.get_noise_map(output_rgb.detach()) 

    hinge = torch.clamp(noise_on_output - noise_on_input + margin, min=0.0)
    return hinge.mean()



def apply_noise_augmentation(im, sigma_range=(0.01, 0.05)):
    
    sigma = random.uniform(sigma_range[0], sigma_range[1])

    gaussian_noise = torch.randn_like(im) * sigma

    poisson_scale = 255.0
    poisson_noise = (torch.poisson(im * poisson_scale) / poisson_scale) - im

    noisy_im = im + gaussian_noise + poisson_noise
    return torch.clamp(noisy_im, 0.0, 1.0)



def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")



def train(epoch):
    model.train()
    loss_print    = 0
    pic_cnt       = 0
    loss_last_10  = 0
    pic_last_10   = 0
    train_len     = len(training_data_loader)
    iter_count    = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)

    
    warmup_scale = min(1.0, epoch / max(1, opt.warmup_loss_epochs))
    
    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()

        
        if opt.noise_aug and random.random() < 0.5:
            im1_input = apply_noise_augmentation(im1, sigma_range=(0.01, 0.05))
        else:
            im1_input = im1

        
        if opt.gamma:
            gamma     = random.randint(opt.start_gamma, opt.end_gamma) / 100.0
            im1_input = im1_input ** gamma

        output_rgb = model(im1_input)

        gt_rgb = im2

        
        output_hvi = model.HVIT(output_rgb)
        gt_hvi     = model.HVIT(gt_rgb)

        
        loss_hvi = (L1_loss(output_hvi, gt_hvi)
                    + D_loss(output_hvi, gt_hvi)
                    + E_loss(output_hvi, gt_hvi)
                    + opt.P_weight * P_loss(output_hvi, gt_hvi)[0])

        loss_rgb = (L1_loss(output_rgb, gt_rgb)
                    + D_loss(output_rgb, gt_rgb)
                    + E_loss(output_rgb, gt_rgb)
                    + opt.P_weight * P_loss(output_rgb, gt_rgb)[0])

        loss = loss_rgb + opt.HVI_weight * loss_hvi

       
        loss_fft = fft_loss(output_rgb, gt_rgb)
        loss     = loss + warmup_scale * opt.fft_weight * loss_fft
        
        loss_chroma    = hvi_chroma_loss(output_hvi, gt_hvi, opt.chroma_weight)
        loss           = loss + warmup_scale * opt.chroma_loss_weight * loss_chroma
        
        loss_noise_cons = noise_consistency_loss(model, im1_input, output_rgb,
                                                 margin=opt.noise_margin)
        loss            = loss + warmup_scale * opt.noise_cons_weight * loss_noise_cons
        

        iter_count += 1

        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        ema.update(model)

        loss_print   += loss.item()
        loss_last_10 += loss.item()
        pic_cnt      += 1
        pic_last_10  += 1

        if iter_count == train_len:
            print("===> Epoch[{}] WarmupScale:{:.2f} | Loss: {:.4f} | FFT: {:.4f} | "
                  "Chroma: {:.4f} | NoiseCons: {:.4f} | lr={:.2e}".format(
                epoch,
                warmup_scale,
                loss_last_10 / pic_last_10,
                loss_fft.item(),
                loss_chroma.item(),
                loss_noise_cons.item(),
                optimizer.param_groups[0]['lr']))
            loss_last_10 = 0
            pic_last_10  = 0
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0).clamp(0, 1))
            gt_img     = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder + 'training'):
                os.mkdir(opt.val_folder + 'training')
            output_img.save(opt.val_folder + 'training/test.png')
            gt_img.save(opt.val_folder + 'training/gt.png')

    return loss_print, pic_cnt


def checkpoint(epoch):
    if not os.path.exists("./weights"):
        os.mkdir("./weights")
    if not os.path.exists("./weights/train"):
        os.mkdir("./weights/train")
    
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    ema_out_path   = "./weights/train/epoch_{}_ema.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    torch.save(ema.get_model().state_dict(), ema_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    print("EMA checkpoint saved to {}".format(ema_out_path))
    
    return ema_out_path


def load_datasets():
    print(f'===> Loading datasets: {opt.dataset}')
    if opt.dataset == 'lol_v1':
        train_set = get_lol_training_set(opt.data_train_lol_v1, size=opt.cropSize)
        test_set  = get_eval_set(opt.data_val_lol_v1)
    elif opt.dataset == 'lol_blur':
        train_set = get_training_set_blur(opt.data_train_lol_blur, size=opt.cropSize)
        test_set  = get_eval_set(opt.data_val_lol_blur)
    elif opt.dataset == 'lolv2_real':
        train_set = get_lol_v2_training_set(opt.data_train_lolv2_real, size=opt.cropSize)
        test_set  = get_eval_set(opt.data_val_lolv2_real)
    elif opt.dataset == 'lolv2_syn':
        train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn, size=opt.cropSize)
        test_set  = get_eval_set(opt.data_val_lolv2_syn)
    elif opt.dataset == 'SID':
        train_set = get_SID_training_set(opt.data_train_SID, size=opt.cropSize)
        test_set  = get_eval_set(opt.data_val_SID)
    elif opt.dataset == 'SICE_mix':
        train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
        test_set  = get_SICE_eval_set(opt.data_val_SICE_mix)
    elif opt.dataset == 'SICE_grad':
        train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
        test_set  = get_SICE_eval_set(opt.data_val_SICE_grad)
    elif opt.dataset == 'fivek':
        train_set = get_fivek_training_set(opt.data_train_fivek, size=opt.cropSize)
        test_set  = get_fivek_eval_set(opt.data_val_fivek)
    else:
        raise Exception("should choose a dataset")

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                                      batch_size=opt.batchSize, shuffle=opt.shuffle)
    testing_data_loader  = DataLoader(dataset=test_set, num_workers=opt.threads,
                                      batch_size=1, shuffle=False)
    return training_data_loader, testing_data_loader


def build_model():
    print('===> Building model')
    model = CIDNet(max_offset=opt.max_offset).cuda()
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model


def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[(opt.nEpochs // 4) - opt.warmup_epochs, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1], eta_mins=[0.0002, 0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1,
                                               total_epoch=opt.warmup_epochs,
                                               after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[opt.nEpochs // 4, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1], eta_mins=[0.0002, 0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch],
                restart_weights=[1], eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1,
                                               total_epoch=opt.warmup_epochs,
                                               after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[opt.nEpochs - opt.start_epoch],
                restart_weights=[1], eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer, scheduler


def init_loss():
    L1_loss = L1Loss(loss_weight=opt.L1_weight, reduction='mean').cuda()
    D_loss  = SSIM(weight=opt.D_weight).cuda()
    E_loss  = EdgeLoss(loss_weight=opt.E_weight).cuda()
    P_loss  = PerceptualLoss(
        {'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
        perceptual_weight=1.0, criterion='mse').cuda()
    return L1_loss, P_loss, E_loss, D_loss


if __name__ == '__main__':

    
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model    = build_model()
    optimizer, scheduler = make_scheduler()
    L1_loss, P_loss, E_loss, D_loss = init_loss()

    
    ema = EMAModel(model, decay=opt.ema_decay)
    

    psnr  = []
    ssim  = []
    lpips = []
    start_epoch = max(opt.start_epoch, 0)

    if not os.path.exists(opt.val_folder):
        os.mkdir(opt.val_folder)

    for epoch in range(start_epoch + 1, opt.nEpochs + start_epoch + 1):
        epoch_loss, pic_num = train(epoch)
        scheduler.step()

        if epoch % opt.snapshots == 0:
            
            model_out_path = checkpoint(epoch)
            norm_size = True
            npy = False

            if opt.dataset == 'lol_v1':
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            if opt.dataset == 'lolv2_real':
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.dataset == 'lolv2_syn':
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            if opt.dataset == 'lol_blur':
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
            if opt.dataset == 'SID':
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.dataset == 'SICE_mix':
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.dataset == 'SICE_grad':
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
            if opt.dataset == 'fivek':
                output_folder = 'fivek/'
                label_dir = opt.data_valgt_fivek
                norm_size = False

            im_dir      = opt.val_folder + output_folder + '*.png'
            is_lol_v1   = (opt.dataset == 'lol_v1')
            is_lolv2_real = (opt.dataset == 'lolv2_real')

            
            eval(ema.get_model(), testing_data_loader, model_out_path,
                 opt.val_folder + output_folder,
                 norm_size=norm_size, LOL=is_lol_v1, v2=is_lolv2_real, alpha=0.8)
            

            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
            print("===> Avg.PSNR: {:.4f} dB".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f}".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f}".format(avg_lpips))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            print(psnr)
            print(ssim)
            print(lpips)

        torch.cuda.empty_cache()

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    with open(f"./results/training/metrics{now}.md", "w") as f:
        f.write("dataset: " + output_folder + "\n")
        f.write(f"lr: {opt.lr}\n")
        f.write(f"batch size: {opt.batchSize}\n")
        f.write(f"crop size: {opt.cropSize}\n")
        f.write(f"HVI_weight: {opt.HVI_weight}\n")
        f.write(f"L1_weight: {opt.L1_weight}\n")
        f.write(f"D_weight: {opt.D_weight}\n")
        f.write(f"E_weight: {opt.E_weight}\n")
        f.write(f"P_weight: {opt.P_weight}\n")
        f.write(f"fft_weight: {opt.fft_weight}\n")
        f.write(f"warmup_loss_epochs: {opt.warmup_loss_epochs}\n")
        f.write(f"chroma_weight: {opt.chroma_weight}\n")
        f.write(f"chroma_loss_weight: {opt.chroma_loss_weight}\n")
        f.write(f"noise_cons_weight: {opt.noise_cons_weight}\n")
        f.write(f"noise_margin: {opt.noise_margin}\n")
        f.write(f"noise_aug: {opt.noise_aug}\n")
        f.write(f"EMA decay: {opt.ema_decay}\n")
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")
        f.write("|---|---|---|---|\n")
        for i in range(len(psnr)):
            f.write(f"| {opt.start_epoch + (i + 1) * opt.snapshots} "
                    f"| {psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} |\n")
