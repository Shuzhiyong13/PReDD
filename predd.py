import argparse
import torch
from models import DiT_models
from download import find_model
from diffusers.models import AutoencoderKL
from torchvision.transforms import ToPILImage
from torchvision import transforms
from diffusion import create_diffusion
from tqdm import tqdm
from torchvision.utils import save_image
import os
import torch.nn as nn
import torch
from utils_szy import Config
import copy
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
import time

def save_images(args, images, place_to_store):
    if not os.path.exists(os.path.dirname(place_to_store)):
        print(f"路径不存在: {os.path.dirname(place_to_store)}")
        os.makedirs(os.path.dirname(place_to_store))
    for clip_val in [2.5]:
        std = torch.std(images)
        mean = torch.mean(images)
        images = torch.clip(images, min=mean-clip_val*std, max=mean+clip_val*std)
    save_image(images, place_to_store, normalize=True)

def main(args):
    # Setup Pytorch
    torch.manual_seed(args.seed)
    # torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # 拿到蒸馏数据集，得到标签和数据
    
    if args.distill_type == 'dm':
        distill_data = torch.load(args.distill_path, weights_only=False)
        img_distill = distill_data['data'][0][0]
        label_distill = distill_data['data'][0][1]
    elif args.distill_type in ['mtt', 'edf']:
        img_distill = torch.load(os.path.join(args.distill_path, 'images_best.pt'))
        label_distill = torch.load(os.path.join(args.distill_path, 'labels_best.pt'))
        # 防止有些结果是软标签
        # label_distill = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    elif args.distill_type == 'NCFM':
        print("==============NCFM==============")
        print("=we split dataset into 4 times=")
        print("==========origin size===========")
        distill_data = torch.load(args.distill_path)
        img_distill = distill_data[0]
        print(img_distill.shape)
        print("==========after split===========")
        label_distill = distill_data[1]
        img_distill_ori = copy.deepcopy(img_distill.detach().cpu())
        def split_to_subimages(images):
            """将[N, C, 128, 128]拆分为[4*N, C, 64, 64]"""
            subs = []
            for img in images:
                subs.extend([
                    img[:, :64, :64],    # 左上
                    img[:, :64, 64:],    # 右上
                    img[:, 64:, :64],    # 左下
                    img[:, 64:, 64:]     # 右下
                ])
            return torch.stack(subs)  # [4*N, C, 64, 64]
        
        img_distill = split_to_subimages(img_distill)
        label_distill = label_distill.repeat_interleave(4)  # 标签也复制4份
    elif 'glad' in args.distill_type:
        img_distill = torch.load(os.path.join(args.distill_path, 'images_best.pt'))
        # label_distill = torch.load(os.path.join(args.distill_path, 'labels_best.pt'))
        label_distill = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    else:
        print("error input format")
    # img_distill_ori = copy.deepcopy(img_distill.detach().cpu())
    config = Config()
    subset = args.dataset.split("-")[1]
    class_indices = config.dict[subset]

    print("==============data==============")
    print('img_distill.shape:', img_distill.shape)
    print("==============label=============")
    print('label_distill.shape', label_distill.shape)
    print("================================")
    resize_it = False
    # # 判断数据格式
    if args.res != args.image_size:
        import torch.nn.functional as F
        img_distill = F.interpolate(img_distill, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)
        print(img_distill.shape)
    
    # 构建 DataLoader 仅用于非 NCFM 模式, NCFM 模式batch_szie=1
    if args.distill_type in ['dm', 'mtt','edf', 'glad'] or 'glad' in args.distill_type:
        batch_size = args.batch_size
        dataset = TensorDataset(img_distill, label_distill)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        total = len(loader)
    else:
        # NCFM 模式用原来的方式
        loader = zip(label_distill, img_distill)  # 等价于原来的 for-loop
        total = len(label_distill)

    # load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size = latent_size,
        num_classes = args.num_classes
    ).to(device)
    
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval() # mc add
    args.device = 'cuda'

    features = []
    # register forward hook
    # if you need vae features to guide, feel free to use this--------- args.hook_layer = [0,1,2,3]
    # def hook_fn(module, input, output):
    #     features.append(output)
    # handle = vae.encoder.down_blocks[args.hook_layer].register_forward_hook(hook_fn)
    
    criterion_mse = nn.MSELoss().to(args.device)

    all_samples = []
    all_labels = []
    data_save = []
    # 执行sample
    print(f"do sampling in dataset {args.dataset}")
    os.makedirs(os.path.join(args.save_dir, args.dataset, args.exp), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.dataset, args.exp, f"images_origin"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.dataset, args.exp, f"images_diffusion"), exist_ok=True)
    ind = 0

    if args.distill_type == 'NCFM':
        ncfm_samples = []   # 存储的四个
        ncfm_labels = []    # 原始标签
        merged_ind = 0 
        merged_samples = [] # 合并后的
        merged_labels = []  # 合并后的

    epoch_start_time = time.time()
    max_mem = 0
    for batch in tqdm(loader, total=total, desc="Processing batches"):
        if args.distill_type in ['mtt', 'edf', 'dm', 'glad'] or 'glad' in args.distill_type:
            # ================== 非NCFM模式：支持batch操作 ==================
            imgs, labels = batch  # imgs: [B, 3, H, W], labels: [B]
            imgs = imgs.to(device)
            labels = labels.to(device)
            batch_size = imgs.size(0)
            # print('type(imgs):', type(imgs))
            # print(imgs)
            # batch_size = imgs.shape[0]

            if args.save_origin:
                # 保存原图
                img_vis_batch = imgs.detach().cpu()
                for i in range(batch_size):
                    dir_path = os.path.join(
                        args.save_dir, args.dataset, args.exp, "images_origin",
                        f"origin_{labels[i].item()}_{ind + i}.png"
                    )
                    # save_image(img_vis_batch[i].unsqueeze(0), dir_path, normalize=True)
                    save_images(args, img_vis_batch[i].unsqueeze(0), dir_path)

            with torch.no_grad():
                features.clear()
                # start_time = time.time()  # 记录开始时间
                z = vae.encode(imgs).latent_dist.sample() * 0.18215  # [B, 4, 32, 32]

                feature_z0=features.copy()
                syn_feature = z.detach().clone()
                
                # start_time = time.time()  # 记录开始时间
                noise = torch.randn_like(z)

                # if no adding noise, no repeat batches
                t = torch.tensor([args.forward_t], device=device).repeat(batch_size)
                z = diffusion.q_sample(x_start=z, t=t, noise=noise)
                # z_cat
                z_cat = torch.cat([z, z], dim=0)                      # [2B, 4, 32, 32]     
                
                # diffusion 广播机制，batch > 1 会报错，这里对feature同样做cat
                syn_feature = torch.cat([syn_feature, syn_feature], dim=0)

                y_real = torch.tensor([class_indices[int(l.item())] for l in labels], device=device)
                y_null = torch.tensor([1000] * batch_size, device=device)
                y_cat = torch.cat([y_real, y_null], dim=0)  # [2B]

                model_kwargs = dict(
                    y=y_cat,
                    cfg_scale=args.cfg_scale,
                )

                if args.sample_method == 'ddim':
                    samples = diffusion.ddim_sample_loop(
                        model=model.forward_with_cfg,
                        shape=z_cat.shape,
                        noise=z_cat,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        progress=False,
                        device=device,
                        start_t=args.reverse_t
                    )
                else:
                    samples, sample_buffer = diffusion.p_sample_loop(
                        model=model.forward_with_cfg,
                        shape=z_cat.shape,
                        noise=z_cat,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        progress=False,
                        device=device,
                        start_t=args.reverse_t
                    )

                samples, _ = samples.chunk(2, dim=0)  # keep only guided samples: [B, 4, 32, 32]
                images_out = vae.decode(samples / 0.18215).sample  # [B, 3, 256, 256]


            # # 保存每个采样时间步的图
            # for t0, image_t in enumerate(sample_buffer):
            #     image_t = vae.decode(image_t / 0.18215).sample
            #     save_images(args, image_t,
            #                 os.path.join(args.save_dir, args.dataset, args.exp,
            #                             "images_all", str(ind), f"{ind}_t{t0}.png"))

            # 保存每张图
            for i in range(batch_size):
                label_i = labels[i].item()
                image_i = images_out[i].cpu()
                if args.saved:
                    save_images(args, image_i.unsqueeze(0),
                                os.path.join(args.save_dir, args.dataset, args.exp,
                                            "images_diffusion", f"{label_i}_sample{ind % 10000}.png"))
                    print(f"saved image {label_i}_sample{ind % 10000}.png")

                all_samples.append(image_i.unsqueeze(0))  # [1,3,256,256]
                all_labels.append(torch.tensor([label_i]))
                ind += 1
        
        else:
            # ================== NCFM模式：默认batch_size=1，一张一张处理 ==================
            y_idx, img = batch  # [1], [1,3,H,W]
            img = img.to(device)

            img_vis = copy.deepcopy(img_distill_ori[ind // 4].detach().cpu())

            if ind%4 ==0 and args.saved:
                # 添加resize到256x256的逻辑
                img_vis = F.interpolate(
                    img_vis.unsqueeze(0),  # 增加batch维度 [1,3,128,128]
                    size=(256, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # 移除batch维度 [3,256,256]
                dir_path = os.path.join(args.save_dir,args.dataset,args.exp,f"images_origin",
                                                f"origin{y_idx.item()}_{(ind // 4)%10}.png")
                save_images(args, img_vis.unsqueeze(0), dir_path)

            # diffusion流程（单张）
            img = img.unsqueeze(0).to(device)
            img_input = img.to(device)
            current_img = img_input
            for loop_id in range(args.diff_loop):
                features.clear()
                z = vae.encode(current_img).latent_dist.sample() * 0.18215
                feature_z0 = features.copy()
                syn_feature = z.detach().clone()
                noise = torch.randn_like(z).to(device)
                forward_t = torch.tensor([args.forward_t], device=device)
                z = diffusion.q_sample(x_start=z, t=forward_t, noise=noise)

                features.clear()
                z = torch.cat([z, z], 0)
                y_real = torch.tensor([class_indices[y_idx.item()]], device=device)
                y_null = torch.tensor([1000], device=device)
                y_cat = torch.cat([y_real, y_null], 0)
                model_kwargs = dict(
                    y=y_cat,
                    cfg_scale=args.cfg_scale,
                )

                if args.sample_method == 'ddim':
                    samples = diffusion.ddim_sample_loop(
                        model=model.forward_with_cfg, shape=z.shape, noise=z,
                        clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device)
                else:
                    samples, sample_buffer = diffusion.p_sample_loop(
                        model=model.forward_with_cfg, shape=z.shape, noise=z,
                        clip_denoised=False, model_kwargs=model_kwargs, progress=False,
                        device=device, start_t=args.reverse_t)

                samples, _ = samples.chunk(2, dim=0)
                with torch.no_grad():   
                    samples = vae.decode(samples / 0.18215).sample
                current_img = samples.detach()
                features.clear()

            ncfm_samples.append(samples.squeeze(0).cpu())
            ncfm_labels.append(y_idx.item())

            # 每4张合并一次
            if len(ncfm_samples) == 4:
                # 拼接后下采样
                merged_512 = torch.zeros(3, 512, 512)
                merged_512[:, :256, :256] = ncfm_samples[0]   # 左上
                merged_512[:, :256, 256:] = ncfm_samples[1]   # 右上
                merged_512[:, 256:, :256] = ncfm_samples[2]   # 左下
                merged_512[:, 256:, 256:] = ncfm_samples[3]   # 右下

                merged_256 = F.interpolate(
                    merged_512.unsqueeze(0), 
                    size=(256, 256), 
                    mode='bilinear'
                ).squeeze(0)

                if args.saved:
                    save_images(args, merged_256.unsqueeze(0), 
                            os.path.join(args.save_dir, args.dataset, args.exp,
                                        f"images_diffusion/{ncfm_labels[0]}_sample{merged_ind%10}.png"))  # 关键修改
                    print(f"saved {ncfm_labels[0]}_sample{merged_ind%10}.png")
                # 进一步保存
                merged_samples.append(merged_256)
                merged_labels.append(ncfm_labels[0])
                ncfm_samples, ncfm_labels = [], []  # 重置
                merged_ind += 1
            ind += 1 

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    print(f"整个采样过程总耗时: {epoch_time:.2f} 秒 ({epoch_time/60:.2f} 分钟)")

    cur_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"总过程峰值显存占用: {cur_mem:.2f} MB")

    if args.distill_type == 'NCFM':
        # 转换为 [10,3,512,512] 和 [10] 标签
        final_tensor = torch.stack(merged_samples)  # [10,3,512,512]
        final_labels = torch.tensor(merged_labels)   # [10]
        data_save.append([copy.deepcopy(final_tensor.detach().cpu()), copy.deepcopy(final_labels.detach().cpu())])
    else:
        data_tensor = torch.cat(all_samples, dim=0)     # [N, C, H, W]
        label_tensor = torch.cat(all_labels, dim=0)     # [N]
        data_save.append([copy.deepcopy(data_tensor.detach().cpu()), copy.deepcopy(label_tensor.detach().cpu())])
    # 保存为字典结构
    torch.save({'data': data_save, 'res': args.image_size},
           os.path.join(args.save_dir, args.dataset, args.exp, f'imagenet_distill_256_256.pt'))
    # during evaluation, we resize to the original size for fairness (128, 128)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument('--res', type=int, default=128, choices=[128, 256, 512], help='resolution')
    parser.add_argument("--save-dir", type=str, default='./sample_results/dit-distillation', help='the directory to put the generated images')
    parser.add_argument("--exp", type=str, default='test', help='the exp name, save_dir + exp')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--diff-loop", type=int, default=1, help="diffusion loop num.")
    parser.add_argument("--forward-t", type=int, default=5, help="Adding noise like Gaussian Distribute.")
    parser.add_argument("--reverse-t", type=int, default=5, help="How many step to reverse process.")
    parser.add_argument("--distill-path", type=str, default=None, help="distilled data path")
    parser.add_argument("--sample-method", type=str, default='ddpm', help="sample-method")
    parser.add_argument('--distill-type', type=str, default='mtt', help='')
    parser.add_argument('--saved', type=lambda x: x.lower() == 'true', default=True, help='Whether to save or not (default: True)')
    parser.add_argument('--save_origin', type=lambda x: x.lower() == 'true', default=True, help='Whether to save original images or not (default: True)')
    args = parser.parse_args()
    print('args\n',args)
    main(args)
