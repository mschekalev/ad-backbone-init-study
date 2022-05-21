import os
import random
import time
from random import sample
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets.mvtec as mvtec
import load_model
import utils


def padim(device, use_cuda, arch, to_plot, to_dump):
    save_path = './padim/mvtec_result'
    data_path = 'C:/Users/Mikhail/Documents/Study/diploma/mvtec'

    # load model
    model, t_d, d = load_model.load_model(arch)

    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    result = {}
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []
    def hook(module, input, output):
        outputs.append(output)

    load_model.per_layer_hook(arch, model, hook)

    os.makedirs(os.path.join(save_path, 'temp_%s' % arch), exist_ok=True)
    if to_plot:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig_img_rocauc = ax[0]
        fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES:

        train_dataset = mvtec.MVTecDataset(data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        if arch not in load_model.ONELAYER_ARCHS:
            train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
            test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        else:
            train_outputs = OrderedDict([('layer1', [])])
            test_outputs = OrderedDict([('layer1', [])])

        # extract train set features
        train_feature_filepath = os.path.join(save_path, 'temp_%s' % arch, 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []

            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)

            # Embedding concat
            if arch in load_model.TRANSFORMERS:
                embedding_vectors = train_outputs['layer1'].unsqueeze(-1)
                for layer_name in ['layer2', 'layer3']:
                    embedding_vectors = utils.embedding_concat(
                        embedding_vectors,
                        train_outputs[layer_name].unsqueeze(-1)
                    )
            else:
                embedding_vectors = train_outputs['layer1']
                if arch not in load_model.ONELAYER_ARCHS:
                    for layer_name in ['layer2', 'layer3']:
                        embedding_vectors = utils.embedding_concat(embedding_vectors, train_outputs[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            for i in range(H * W):
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            # save learned distribution
            train_outputs = [mean, cov]
            if to_dump:
                with open(train_feature_filepath, 'wb') as f:
                    pickle.dump(train_outputs, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            times = []

            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                start_time = time.monotonic()  # Calculating average time of inference
                _ = model(x.to(device))
                end_time = time.monotonic() - start_time

                times.append(end_time)
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        result[class_name] = []

        # Embedding concat
        if arch in load_model.TRANSFORMERS:
            embedding_vectors = test_outputs['layer1'].unsqueeze(-1)
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = utils.embedding_concat(
                    embedding_vectors,
                    test_outputs[layer_name].unsqueeze(-1)
                )
        else:
            embedding_vectors = test_outputs['layer1']
            if arch not in load_model.ONELAYER_ARCHS:
                for layer_name in ['layer2', 'layer3']:
                    embedding_vectors = utils.embedding_concat(
                        embedding_vectors,
                        test_outputs[layer_name]
                    )

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        if to_plot:
            fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        if to_plot:
            fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
            save_dir = save_path + '/' + f'pictures_{arch}'
            os.makedirs(save_dir, exist_ok=True)
            utils.plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

        result[class_name] = [img_roc_auc, per_pixel_rocauc, np.mean(np.array(times))]

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))

    if to_plot:
        fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
        fig_img_rocauc.legend(loc="lower right")
        
        fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
        fig_pixel_rocauc.legend(loc="lower right")

        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f'roc_curve_{arch}.png'), dpi=100)

    result['Average'] = [np.mean(total_roc_auc), np.mean(total_pixel_roc_auc), np.mean(np.array(list(result.values()))[:,2])]

    df_result = pd.DataFrame(result).T.reset_index()
    df_result.columns = ['Type', 'Image', 'Pixel', 'Time']

    print(df_result)
    df_result.to_csv(f'{save_path}/result_{arch}.csv', index=False)
