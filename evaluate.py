import os
import torch
import numpy as np
import random
import argparse
from utils_szy import get_eval_pool, get_network, get_eval_lrs, evaluate_synset, get_dataset, ParamDiffAug
import copy
from torchvision import transforms

def main(args):

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    print('model_eval_pool:', model_eval_pool)
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    # 获取数据集
    if args.test_type in ['mtt', 'edf', 'glad']:
        # 测试原始 MTT 蒸馏数据集性能
        image = torch.load(os.path.join(args.distill_path, 'images_best.pt'))
        label = torch.load(os.path.join(args.distill_path, 'labels_best.pt'))
        # label = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    elif args.test_type == 'dm' or args.test_type == 'ours':
        # dm 以及我们的方法是这样的
        diffusion_data = torch.load(args.distill_path)
        image = diffusion_data['data'][0][0]
        label = diffusion_data['data'][0][1]
    elif args.test_type == 'NCFM':
        distill_data = torch.load(args.distill_path)
        image = distill_data[0]
        label = distill_data[1]
    elif args.test_type == 'random':
        image = torch.load(os.path.join(args.distill_path))
        label = torch.tensor([0,1,2,3,4,5,6,7,8,9])

    print('image.shape:', image.shape)
    print('dataset', args.dataset)
    eval_pool_dict = get_eval_lrs(args)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.res, args=args)

    transform = transforms.Resize((128, 128))

    image = transform(image)
    print(image.shape)
    print(im_size)
    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\n model_eval = %s' % model_eval)
        print('DSA augmentation strategy: \n', args.dsa_strategy)
        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

        accs = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device)
            image_eval, label_eval = copy.deepcopy(image.detach()), copy.deepcopy(label.detach()) # avoid any unaware modification
            args.lr_net = eval_pool_dict[model_eval]
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_eval, label_eval, testloader, args)
            accs.append(acc_test)
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

        accs_all_exps[model_eval] += accs
    
    cross_arch_accs = []
    cross_arch_stds = []  # 用来存每个架构的 std

    for key in model_eval_pool:
        accs = accs_all_exps[key]
        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        print('Evaluation results, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' %
            (len(accs), key, mean_acc, std_acc))
        
        if key != 'ConvNet':  # 只统计非原架构
            cross_arch_accs.append(mean_acc)
            cross_arch_stds.append(std_acc)  # 记录 std

    # 计算跨架构 mean 和 average std
    cross_mean = np.mean(cross_arch_accs)
    avg_std = np.mean(cross_arch_stds)

    print('Cross-Architecture average accuracy = %.2f%%  avg std = %.2f%%' % (cross_mean, avg_std))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--eval_mode', type=str, default='M',
                        help='eval_mode')  # S: the same to training model, M: multi architectures
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--mom_img', type=float, default=0.5, help='momentum for updating synthetic images')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_test', type=int, default=128, help='batch size for training networks')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--test_type', type=str, default='mtt', help='')
    parser.add_argument('--distill_path', type=str, default='data', help='distilled dataset path')
    parser.add_argument('--res', type=int, default=128, choices=[128, 256, 512], help='resolution')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for pixels or f_latents')

    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=3)
    args = parser.parse_args()
    main(args)

