from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import torch

def kmean_process(original_img, n_cluster_center):

    depth, height, width = original_img.shape
    pixel_sample = np.reshape(original_img, (height*width, depth))
    compressed_img = np.zeros((height, width, depth), dtype=np.float32)

    kmeans = KMeans(n_clusters=n_cluster_center)
    kmeans.fit(pixel_sample)
    cluster_assignments = kmeans.predict(pixel_sample)
    cluster_centers = kmeans.cluster_centers_

    pixel_count = 0
    for i in range(height):
        for j in range(width):
            cluster_idx = cluster_assignments[pixel_count]
            cluster_value = cluster_centers[cluster_idx]
            compressed_img[i][j] = cluster_value
            pixel_count += 1

    compressed_img = np.reshape(compressed_img, (depth, height, width))
    return compressed_img

def mini_bench_kmean_process(original_img, n_cluster_center, batch_size=100):

    depth, height, width = original_img.shape
    pixel_sample = np.reshape(original_img, (height*width, depth))
    compressed_img = np.zeros((height, width, depth), dtype=np.float32)

    kmeans = MiniBatchKMeans(n_clusters=n_cluster_center, batch_size=batch_size)
    kmeans.fit(pixel_sample)
    cluster_assignments = kmeans.predict(pixel_sample)
    cluster_centers = kmeans.cluster_centers_

    pixel_count = 0
    for i in range(height):
        for j in range(width):
            cluster_idx = cluster_assignments[pixel_count]
            cluster_value = cluster_centers[cluster_idx]
            compressed_img[i][j] = cluster_value
            pixel_count += 1

    compressed_img = np.reshape(compressed_img, (depth, height, width))
    return compressed_img

def Kmeans_cluster(in_tensor, k = 2):
    examples = in_tensor.detach().cpu().numpy()
    assert len(examples.shape) == 4
    inter_res = np.array([kmean_process(example, k) for example in examples])
    assert inter_res.shape == examples.shape
    res = torch.from_numpy(inter_res).cuda()
    return res

def mb_Kmeans_cluster(in_tensor, k = 2):
    examples = in_tensor.detach().cpu().numpy()
    assert len(examples.shape) == 4
    inter_res = np.array([mini_bench_kmean_process(example, k) for example in examples])
    assert inter_res.shape == examples.shape
    res = torch.from_numpy(inter_res).cuda()
    return res

