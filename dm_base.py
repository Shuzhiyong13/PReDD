import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils_szy import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, match_loss, get_time, \
    TensorDataset, epoch, DiffAugment, ParamDiffAug, get_eval_lrs
# import wandb
from tqdm import tqdm
import torchvision
import random
import gc

def get_args(custom_args=None):
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='M',
                        help='eval_mode')  # S: the same to training model, M: multi architectures
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--save_it', type=int, default=None, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')

    parser.add_argument('--mom_img', type=float, default=0.5, help='momentum for updating synthetic images')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_test', type=int, default=128, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='noise', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')

    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    parser.add_argument('--res', type=int, default=128, choices=[128, 256, 512], help='resolution')
    


    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for pixels or f_latents')

    if custom_args is not None:
        return parser.parse_args(custom_args)
    elif 'ipykernel' in sys.modules:
        return parser.parse_args(args=[])  # 默认用 Jupyter 默认值
    else:
        return parser.parse_args()


# we first do dm method to distill data
def main(args):
    args = parser.parse_args()
    args.method = 'DM'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True


    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # run = wandb.init(
    #     project="GLaD",
    #     job_type="DM",
    #     config=args
    # )

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    # run_dir = "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), run.name)

    args.save_path = os.path.join(args.save_path, "dm", args.dataset, f"{args.ipc}_ipc")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    eval_it_pool = [200, 400, 600, 800, 1000]
    print('eval_it_pool', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.res, args=args)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    def build_dataset(ds, class_map, num_classes):
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        print("BUILDING DATASET")
        for i in tqdm(range(len(ds))):
            sample = ds[i]
            images_all.append(torch.unsqueeze(sample[0], dim=0))
            labels_all.append(class_map[torch.tensor(sample[1]).item()])
        for i, lab in tqdm(enumerate(labels_all)):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to("cpu")
        labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

        return images_all, labels_all, indices_class

    images_all, labels_all, indices_class = build_dataset(dst_train, class_map, num_classes)

    # real_train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True,
    #                                                 num_workers=16)

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle].to(args.device)

    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')

    """ training """
    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    print('%s training begins'%get_time())

    for it in range(args.Iteration+1):
        eval_pool_dict = get_eval_lrs(args)

        """ Evaluate synthetic data """
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                print('DSA augmentation strategy: \n', args.dsa_strategy)
                print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                accs = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device)
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                    args.lr_net = eval_pool_dict[model_eval]
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                    accs.append(acc_test)
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                if it == args.Iteration: # record the final results
                    accs_all_exps[model_eval] += accs

            ''' visualize and save '''
            save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, it))
            image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
            for ch in range(channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
            image_syn_vis[image_syn_vis<0] = 0.0
            image_syn_vis[image_syn_vis>1] = 1.0
            print('image_syn_vis.shape:', image_syn_vis.shape)
            save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

        """ Train synthetic data"""
        net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device)
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False

        # 获取嵌入器（适配 GPU 多卡）
        if hasattr(net, 'module'):
            embed = net.module.embed if hasattr(net.module, 'embed') else net.module
        else:
            embed = net.embed if hasattr(net, 'embed') else net

        loss_avg = 0

        ''' update synthetic data '''
        if 'ConvNet' in args.model:
            if 'BN' not in args.model: # for ConvNet
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real).to(args.device)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)

                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            else: # for ConvNetBN
                images_real_all = []
                images_syn_all = []
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)

                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)

                output_real = embed(images_real_all).detach()
                output_syn = embed(images_syn_all)

                loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)
        else: # 新增: 对 ResNet18 或其他嵌入器结构的支持
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real).to(args.device)
                img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                output_real = net.embed(img_real).detach()  # shape: [B, D]
                output_syn = net.embed(img_syn)             # shape: [ipc, D]

                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)
    
        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        loss_avg += loss.item()


        loss_avg /= (num_classes)

        if it%10 == 0:
            print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

        if it == args.Iteration: # only record the final results
            data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
            torch.save({'data': data_save, 'res': args.res, }, os.path.join(args.save_path, 'res_%d_%s_%s_%s_%dipc.pt'%(args.res, args.method, args.dataset, args.model, args.ipc)))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run experiment, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode')  # S: the same to training model, M: multi architectures
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--eval_it', type=int, default=200, help='how often to evaluate')
    parser.add_argument('--save_it', type=int, default=None, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')

    parser.add_argument('--mom_img', type=float, default=0.5, help='momentum for updating synthetic images')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_test', type=int, default=128, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='noise', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')

    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    parser.add_argument('--res', type=int, default=128, choices=[128, 256, 512], help='resolution')
    

    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for pixels or f_latents')
    args = parser.parse_args()
    main(args)