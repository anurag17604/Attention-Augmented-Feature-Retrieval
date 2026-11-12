"""
Interactive Visual Retrieval Demo
Input: Any query fashion image from dataset
Output: Top-K retrieved visually similar items with similarity scores and category tags.
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torchvision.models as models

from models.triplet_attention import TripletAttention


def load_feature_extractor(backbone='resnet50', use_attention=True, device='cpu'):
    if backbone == 'resnet50':
        base = models.resnet50(weights='DEFAULT')
    elif backbone == 'resnet18':
        base = models.resnet18(weights='DEFAULT')
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    layers = list(base.children())[:-2]
    model = torch.nn.Sequential(*layers)
    if use_attention:
        model.add_module('triplet_att', TripletAttention())
    model.add_module('pool', torch.nn.AdaptiveAvgPool2d((1, 1)))
    model.add_module('flatten', torch.nn.Flatten(1))
    model.eval()
    return model.to(device)


def run_demo(query_image_path, gallery_features_path, gallery_csv_path, top_k=5, output_img='retrieval_result.png'):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = load_feature_extractor('resnet50', use_attention=True, device=device)

    # Preprocess query image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    q_img = Image.open(query_image_path).convert('RGB')
    q_tensor = preprocess(q_img).unsqueeze(0).to(device)

    with torch.no_grad():
        q_feat = model(q_tensor)
        q_feat = q_feat / torch.norm(q_feat, p=2, dim=1, keepdim=True)
        q_feat = q_feat.cpu().numpy()[0]

    # Load gallery embeddings and metadata
    gallery_feats = np.load(gallery_features_path)
    gallery_df = pd.read_csv(gallery_csv_path)

    # Compute Euclidean distance and Similarity
    sims = np.dot(gallery_feats, q_feat)
    dists = np.sqrt(np.clip(2.0 - 2.0 * sims, 0.0, 4.0))
    similarity_scores = 1.0 / (1.0 + dists)

    top_indices = np.argsort(-similarity_scores)[:top_k]

    # Visualize results
    fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 4))
    axes[0].imshow(q_img)
    axes[0].set_title("Query Image\n(Input)", fontsize=11, fontweight='bold', color='blue')
    axes[0].axis('off')

    for rank, idx in enumerate(top_indices, 1):
        row = gallery_df.iloc[idx]
        img_id = row['id']
        art_type = row.get('articleType', '')
        score = similarity_scores[idx]

        item_path = os.path.join(os.path.dirname(query_image_path), f"{img_id}.jpg")
        if os.path.exists(item_path):
            item_im = Image.open(item_path)
            axes[rank].imshow(item_im)
        else:
            axes[rank].text(0.5, 0.5, f"ID: {img_id}", ha='center')

        axes[rank].set_title(f"Rank {rank} (Sim: {score:.4f})\n{art_type}", fontsize=10)
        axes[rank].axis('off')

    plt.tight_layout()
    plt.savefig(output_img, dpi=200)
    print(f"Top-{top_k} retrieval visualization saved to: {output_img}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True, help='Path to query image')
    parser.add_argument('--gallery_feats', type=str, default='retrieval_results/feature_cache/gallery_resnet50_Base_rgb.npy')
    parser.add_argument('--gallery_csv', type=str, default='retrieval_results/gallery_split.csv')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--output', type=str, default='retrieval_demo.png')
    args = parser.parse_args()

    run_demo(args.query, args.gallery_feats, args.gallery_csv, args.top_k, args.output)
