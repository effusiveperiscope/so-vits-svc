from sklearn.cluster import MiniBatchKMeans
import glob
import numpy as np
import tqdm
import torch
import argparse
import os
from pathlib import Path

# The fundamental assumption this is making is that whisper features
# have correlation with phonemes in clusters/have a somewhat "smooth"
# latent space. We also don't know how many clusters, if any, are
# appropriate for whisper features.

# It's not clear that the Whisper encoder is intended to be invariant to
# speaker formants in the same way that ContentVec does.

# I don't understand the motivation for using the Whisper encoder this way.

def train_cluster(in_dir, n_clusters, out_cluster_path):
    print(f"Loading features from {in_dir}")
    features = []
    for path in tqdm.tqdm(glob.glob(os.path.join(in_dir,"*.ppg.npy"))):
        features.append(np.load(path))
    features = np.concatenate(features, axis=0)
    print(features.nbytes / 1024**2, "MB , shape:", features.shape,
          features.dtype)
    features = features.astype(np.float32)
    print(f"Clustering features of shape: {features.shape}")
    # Because these tend to be large we default use minibatchkmeans
    # 12:08 
    kmeans = MiniBatchKMeans(
        n_clusters = n_clusters,
        batch_size=4096, max_iter=80, n_init='auto').fit(features)

    cluster_obj = { "n_features_in_": kmeans.n_features_in_, 
        "_n_threads": kmeans._n_threads,
        "cluster_centers_": kmeans.cluster_centers_ }
    torch.save(cluster_obj, out_cluster_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clusters", type=int, default=500)
    parser.add_argument("-p","--ppg_dir", help="whisper directory")
    parser.add_argument("-c","--cluster_dir", help="output cluster directory")
    args = parser.parse_args()

    for speaker in os.listdir(args.ppg_dir):
        train_cluster(
            in_dir = os.path.join(args.ppg_dir,speaker),
            n_clusters = args.num_clusters,
            out_cluster_path = os.path.join(
                args.cluster_dir,speaker+".cluster"))
