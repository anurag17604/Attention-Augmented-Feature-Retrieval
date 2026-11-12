"""
Reproducible and Rigorous Retrieval Experiment Pipeline for ICICC 2027 Submission 344.
Authors: ICICC 2027 Evaluation Suite
"""

import os
import sys
import time
import json
import hashlib
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


# ==============================================================================
# 1. Standard Triplet Attention (Misra et al., WACV 2021)
# ==============================================================================
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(AttentionGate, self).__init__()
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size=kernel_size, stride=1,
                              padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate(kernel_size=kernel_size)
        self.hc = AttentionGate(kernel_size=kernel_size)
        self.hw = AttentionGate(kernel_size=kernel_size)

    def forward(self, x):
        # 1. Channel-Height branch (permute C and W)
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        # 2. Channel-Width branch (permute C and H)
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        # 3. Spatial Height-Width branch
        x_out = self.hw(x)

        # Average and aggregate
        return 1.0 / 3.0 * (x_out + x_out11 + x_out21)


# ==============================================================================
# 2. Feature Extraction Models
# ==============================================================================
class VisualFeatureExtractor(nn.Module):
    def __init__(self, backbone_name='resnet50', use_attention=False, pretrained=True):
        super(VisualFeatureExtractor, self).__init__()
        self.backbone_name = backbone_name
        self.use_attention = use_attention

        # Load backbone
        weights = 'DEFAULT' if pretrained else None
        if backbone_name == 'resnet18':
            base = models.resnet18(weights=weights)
            in_features = 512
        elif backbone_name == 'resnet34':
            base = models.resnet34(weights=weights)
            in_features = 512
        elif backbone_name == 'resnet50':
            base = models.resnet50(weights=weights)
            in_features = 2048
        elif backbone_name == 'resnet101':
            base = models.resnet101(weights=weights)
            in_features = 2048
        elif backbone_name == 'resnet152':
            base = models.resnet152(weights=weights)
            in_features = 2048
        elif backbone_name == 'resnext50_32x4d':
            base = models.resnext50_32x4d(weights=weights)
            in_features = 2048
        elif backbone_name == 'resnext101_32x8d':
            base = models.resnext101_32x8d(weights=weights)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Extract stages up to conv5_x
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.attention = TripletAttention(kernel_size=7) if use_attention else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_attention:
            x = self.attention(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        # L2 normalization for cosine / Euclidean similarity
        norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x = x / norm
        return x


# ==============================================================================
# 3. Dataset Loader with RGB and Grayscale Modes
# ==============================================================================
class FashionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, color_mode: str = 'rgb'):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.color_mode = color_mode.lower()

        # ImageNet normalization parameters
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id']}.jpg")

        im = Image.open(img_path)
        if self.color_mode == 'grayscale':
            im = im.convert('L').convert('RGB')  # 3-channel grayscale as in paper
        else:
            im = im.convert('RGB')

        im_t = transforms.functional.to_tensor(self.resize(im))
        im_t = self.normalize(im_t)

        return im_t, idx


# ==============================================================================
# 4. Rigorous Retrieval Evaluation Protocol
# ==============================================================================
def compute_retrieval_metrics(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    query_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    relevance_col: str = 'articleType',
    top_k_list: List[int] = [5, 10],
    mask_identical_hashes: bool = True
) -> Dict[str, Any]:
    """
    Computes genuine retrieval metrics:
    - Precision@K
    - Recall@K
    - F1@K
    - Mean Average Precision (mAP)
    - Mean Reciprocal Rank (MRR)
    - Mean Similarity (S = 1 / (1 + D))
    """
    n_queries = len(query_df)
    n_gallery = len(gallery_df)

    # Pre-extract labels and hashes
    query_labels = query_df[relevance_col].values
    gallery_labels = gallery_df[relevance_col].values
    query_hashes = query_df['image_hash'].values if 'image_hash' in query_df else None
    gallery_hashes = gallery_df['image_hash'].values if 'image_hash' in gallery_df else None

    # Precompute gallery label counts for fast Recall denominator
    gallery_label_counts = pd.Series(gallery_labels).value_counts().to_dict()

    p_at_k = {k: [] for k in top_k_list}
    r_at_k = {k: [] for k in top_k_list}
    f1_at_k = {k: [] for k in top_k_list}
    sim_at_k = {k: [] for k in top_k_list}
    ap_list = []
    mrr_list = []

    # Batch compute cosine similarities: Q x G
    # Since features are L2 normalized: Cosine Sim = Q @ G.T, Euclidean Dist = sqrt(2 - 2*Cosine)
    batch_size = 250
    for i in range(0, n_queries, batch_size):
        q_batch = query_feats[i:i + batch_size]
        q_labels_batch = query_labels[i:i + batch_size]

        # Inner product
        sim_matrix = np.dot(q_batch, gallery_feats.T)  # (batch, n_gallery)

        # For each query in batch
        for b_idx in range(len(q_batch)):
            q_idx = i + b_idx
            q_label = q_labels_batch[b_idx]
            sims = sim_matrix[b_idx].copy()

            # Mask identical images to prevent trivial duplicate data leakage
            if mask_identical_hashes and query_hashes is not None and gallery_hashes is not None:
                q_h = query_hashes[q_idx]
                dup_mask = (gallery_hashes == q_h)
                sims[dup_mask] = -1e9

            # Sort descending by similarity (ascending by Euclidean distance)
            ranked_indices = np.argsort(-sims)

            # Get matching binary relevance
            retrieved_labels = gallery_labels[ranked_indices]
            is_relevant = (retrieved_labels == q_label)

            # Total relevant in gallery (excluding any masked duplicates)
            total_relevant = gallery_label_counts.get(q_label, 0)
            if mask_identical_hashes and query_hashes is not None and gallery_hashes is not None:
                total_relevant -= np.sum((gallery_labels == q_label) & (gallery_hashes == q_h))

            if total_relevant <= 0:
                continue

            # Compute Euclidean distances and Similarity S = 1 / (1 + D)
            # D = sqrt(clip(2 - 2*cos, 0, 4))
            cos_ranked = sims[ranked_indices]
            d_ranked = np.sqrt(np.clip(2.0 - 2.0 * cos_ranked, 0.0, 4.0))
            s_ranked = 1.0 / (1.0 + d_ranked)

            # Top-K Metrics
            for k in top_k_list:
                rel_k = is_relevant[:k]
                tp_k = np.sum(rel_k)
                prec = tp_k / k
                rec = tp_k / total_relevant
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                mean_s = np.mean(s_ranked[:k])

                p_at_k[k].append(prec)
                r_at_k[k].append(rec)
                f1_at_k[k].append(f1)
                sim_at_k[k].append(mean_s)

            # Average Precision (AP)
            cumsum_rel = np.cumsum(is_relevant)
            ranks = np.arange(1, len(is_relevant) + 1)
            precisions = cumsum_rel / ranks
            ap = np.sum(precisions * is_relevant) / total_relevant
            ap_list.append(ap)

            # MRR (Mean Reciprocal Rank)
            first_rel_rank = np.where(is_relevant)[0]
            if len(first_rel_rank) > 0:
                mrr_list.append(1.0 / (first_rel_rank[0] + 1))
            else:
                mrr_list.append(0.0)

    results = {
        'n_queries_evaluated': len(ap_list),
        'gallery_size': n_gallery,
        'relevance_criterion': relevance_col,
        'mAP': float(np.mean(ap_list)),
        'MRR': float(np.mean(mrr_list)),
    }
    for k in top_k_list:
        results[f'Precision@{k}'] = float(np.mean(p_at_k[k]))
        results[f'Recall@{k}'] = float(np.mean(r_at_k[k]))
        results[f'F1@{k}'] = float(np.mean(f1_at_k[k]))
        results[f'MeanSim@{k}'] = float(np.mean(sim_at_k[k]))

    return results


# ==============================================================================
# 5. Feature Extraction & Benchmark Utilities
# ==============================================================================
def extract_dataset_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, float]:
    model.eval()
    model.to(device)
    features_list = []
    
    t0 = time.time()
    with torch.no_grad():
        for batch_imgs, _ in dataloader:
            batch_imgs = batch_imgs.to(device)
            feats = model(batch_imgs)
            features_list.append(feats.cpu().numpy())
    elapsed = time.time() - t0
    
    all_features = np.concatenate(features_list, axis=0)
    return all_features, elapsed


def measure_query_latency(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    n_trials: int = 50
) -> float:
    # Single query retrieval latency in milliseconds
    latencies = []
    for _ in range(n_trials):
        idx = np.random.randint(0, len(query_feats))
        q = query_feats[idx:idx+1]
        t0 = time.perf_counter()
        sims = np.dot(q, gallery_feats.T)
        ranked = np.argsort(-sims[0])[:10]
        t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(t_elapsed)
    return float(np.mean(latencies))


def count_parameters(model: nn.Module) -> Tuple[int, float]:
    # Returns (param_count, size_mb)
    params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return params, size_mb


# ==============================================================================
# 6. Main Benchmark Runner
# ==============================================================================
def run_benchmark(
    dataset_dir: str,
    styles_csv: str,
    output_dir: str = 'retrieval_results',
    sample_queries: int = 1000,
    batch_size: int = 128,
    device_name: str = 'auto'
):
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, 'feature_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Determine device
    if device_name == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_name)
    print(f"Using device: {device}")

    # Load and clean metadata
    print("Loading styles.csv and indexing images...")
    df = pd.read_csv(styles_csv, on_bad_lines='skip')
    
    # Check physical existence of images
    img_files = set(os.listdir(dataset_dir))
    df = df[df['id'].apply(lambda x: f"{x}.jpg" in img_files)].copy()
    print(f"Total physically verified images: {len(df)}")

    # Precompute image MD5 hashes for deduplication
    hash_cache_file = os.path.join(output_dir, 'image_hashes.csv')
    if os.path.exists(hash_cache_file):
        print("Loading image hashes from cache...")
        hash_df = pd.read_csv(hash_cache_file)
        df = df.merge(hash_df, on='id', how='left')
    else:
        print("Computing image MD5 hashes...")
        hashes = []
        for iid in df['id'].values:
            p = os.path.join(dataset_dir, f"{iid}.jpg")
            with open(p, 'rb') as fp:
                hashes.append(hashlib.md5(fp.read()).hexdigest())
        df['image_hash'] = hashes
        df[['id', 'image_hash']].to_csv(hash_cache_file, index=False)
        print("Image hashes computed and cached.")

    # Fixed seed train/test (gallery/query) split (80:20)
    # Grouped or stratified to ensure test queries represent diverse categories
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_total = len(df)
    n_train = int(0.8 * n_total)
    
    gallery_df = df.iloc[:n_train].copy().reset_index(drop=True)
    query_df_full = df.iloc[n_train:].copy().reset_index(drop=True)

    # Subsample queries if requested for fast benchmark
    if sample_queries > 0 and sample_queries < len(query_df_full):
        query_df = query_df_full.iloc[:sample_queries].copy().reset_index(drop=True)
    else:
        query_df = query_df_full.copy()

    print(f"Gallery set size: {len(gallery_df)}, Query set evaluated: {len(query_df)}")

    # Save splits for transparency
    gallery_df[['id', 'masterCategory', 'subCategory', 'articleType']].to_csv(
        os.path.join(output_dir, 'gallery_split.csv'), index=False
    )
    query_df[['id', 'masterCategory', 'subCategory', 'articleType']].to_csv(
        os.path.join(output_dir, 'query_split.csv'), index=False
    )

    all_results = {}

    # Define model configurations to benchmark
    models_to_test = [
        ('resnet18', False, 'rgb'),
        ('resnet34', False, 'rgb'),
        ('resnet50', False, 'rgb'),
        ('resnet101', False, 'rgb'),
        ('resnet152', False, 'rgb'),
        ('resnext50_32x4d', False, 'rgb'),
        ('resnet50', True, 'rgb'),         # ResNet50 + Triplet Attention (RGB)
        ('resnet50', False, 'grayscale'),   # ResNet50 (Grayscale)
        ('resnet50', True, 'grayscale'),    # ResNet50 + Triplet Attention (Grayscale)
    ]

    for backbone, use_att, color in models_to_test:
        att_tag = "TA" if use_att else "Base"
        exp_name = f"{backbone}_{att_tag}_{color}"
        print(f"\n========================================================")
        print(f"Running Experiment: {exp_name}")
        print(f"========================================================")

        # 1. Initialize model
        model = VisualFeatureExtractor(backbone_name=backbone, use_attention=use_att, pretrained=True)
        param_count, model_mb = count_parameters(model)
        print(f"Model parameters: {param_count:,} ({model_mb:.2f} MB)")

        # 2. Check feature cache
        gallery_cache = os.path.join(cache_dir, f"gallery_{exp_name}.npy")
        query_cache = os.path.join(cache_dir, f"query_{exp_name}.npy")

        if os.path.exists(gallery_cache) and os.path.exists(query_cache):
            print("Loading cached features...")
            gallery_feats = np.load(gallery_cache)
            query_feats = np.load(query_cache)
            feat_extract_time = 0.0
        else:
            print("Extracting features...")
            gallery_ds = FashionDataset(gallery_df, dataset_dir, color_mode=color)
            query_ds = FashionDataset(query_df, dataset_dir, color_mode=color)

            gallery_loader = DataLoader(gallery_ds, batch_size=batch_size, shuffle=False, num_workers=2)
            query_loader = DataLoader(query_ds, batch_size=batch_size, shuffle=False, num_workers=2)

            g_feats, g_time = extract_dataset_features(model, gallery_loader, device)
            q_feats, q_time = extract_dataset_features(model, query_loader, device)
            feat_extract_time = g_time + q_time

            np.save(gallery_cache, g_feats)
            np.save(query_cache, q_feats)
            gallery_feats = g_feats
            query_feats = q_feats
            print(f"Features extracted in {feat_extract_time:.2f}s ({len(gallery_feats)+len(query_feats)} images)")

        # 3. Query latency benchmark
        q_latency = measure_query_latency(query_feats, gallery_feats, n_trials=50)
        print(f"Query retrieval latency: {q_latency:.2f} ms/query")

        # 4. Evaluate Fine-Grained Retrieval Metrics (articleType relevance)
        metrics_article = compute_retrieval_metrics(
            query_feats, gallery_feats, query_df, gallery_df,
            relevance_col='articleType', top_k_list=[5, 10], mask_identical_hashes=True
        )

        # 5. Evaluate Coarse Category Retrieval Metrics (masterCategory relevance)
        metrics_master = compute_retrieval_metrics(
            query_feats, gallery_feats, query_df, gallery_df,
            relevance_col='masterCategory', top_k_list=[5, 10], mask_identical_hashes=True
        )

        # 6. Category-wise evaluation (Apparel, Footwear, Accessories)
        cat_results = {}
        for cat in ['Apparel', 'Footwear', 'Accessories']:
            sub_q_df = query_df[query_df['masterCategory'] == cat].copy().reset_index(drop=True)
            if len(sub_q_df) > 0:
                sub_q_indices = query_df.index[query_df['masterCategory'] == cat].tolist()
                sub_q_feats = query_feats[sub_q_indices]
                cat_m = compute_retrieval_metrics(
                    sub_q_feats, gallery_feats, sub_q_df, gallery_df,
                    relevance_col='articleType', top_k_list=[5, 10], mask_identical_hashes=True
                )
                cat_results[cat] = cat_m

        exp_result = {
            'backbone': backbone,
            'use_attention': use_att,
            'color_mode': color,
            'parameters': param_count,
            'model_size_mb': model_mb,
            'feature_extraction_time_s': feat_extract_time,
            'query_latency_ms': q_latency,
            'fine_grained_metrics': metrics_article,
            'coarse_metrics': metrics_master,
            'category_wise_metrics': cat_results
        }
        all_results[exp_name] = exp_result

        print(f"Results for {exp_name}:")
        print(f"  Fine-Grained (articleType): P@5={metrics_article['Precision@5']:.4f}, R@5={metrics_article['Recall@5']:.4f}, mAP={metrics_article['mAP']:.4f}")
        print(f"  Coarse (masterCategory):    P@5={metrics_master['Precision@5']:.4f}, MeanSim@5={metrics_master['MeanSim@5']:.4f}")

    # 7. Gallery Scaling Benchmark on ResNet50
    print("\n========================================================")
    print("Running Gallery Scaling Benchmark (ResNet50)")
    print("========================================================")
    gallery_scales = [1000, 5000, 10000, 20000, len(gallery_df)]
    r50_feats = np.load(os.path.join(cache_dir, "gallery_resnet50_Base_rgb.npy"))
    r50_q = np.load(os.path.join(cache_dir, "query_resnet50_Base_rgb.npy"))
    
    scaling_results = []
    for g_size in gallery_scales:
        sub_g = r50_feats[:g_size]
        sub_g_df = gallery_df.iloc[:g_size]
        lat = measure_query_latency(r50_q, sub_g, n_trials=50)
        m = compute_retrieval_metrics(r50_q, sub_g, query_df, sub_g_df, relevance_col='articleType', top_k_list=[5, 10])
        scaling_results.append({
            'gallery_size': g_size,
            'query_latency_ms': lat,
            'Precision@5': m['Precision@5'],
            'Recall@5': m['Recall@5'],
            'mAP': m['mAP']
        })
        print(f"Gallery Size {g_size:>5}: Latency = {lat:.2f} ms, P@5 = {m['Precision@5']:.4f}, mAP = {m['mAP']:.4f}")

    all_results['gallery_scaling'] = scaling_results

    # Save full JSON
    results_json_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(results_json_path, 'w') as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nAll benchmark results successfully saved to {results_json_path}")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visual Retrieval Benchmark")
    parser.add_argument('--dataset_dir', type=str, default='dataset fashion/images')
    parser.add_argument('--styles_csv', type=str, default='dataset fashion/styles.csv')
    parser.add_argument('--output_dir', type=str, default='retrieval_results')
    parser.add_argument('--sample_queries', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    run_benchmark(
        dataset_dir=args.dataset_dir,
        styles_csv=args.styles_csv,
        output_dir=args.output_dir,
        sample_queries=args.sample_queries,
        batch_size=args.batch_size,
        device_name=args.device
    )


