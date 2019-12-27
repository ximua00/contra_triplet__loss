import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from utils import get_dataset_embeddings


def get_pairwise_distances(model, query_loader, gallery_loader, function='euclidean'):
    query_embeddings, query_targets = get_dataset_embeddings(model, query_loader)
    gallery_embeddings, gallery_targets = get_dataset_embeddings(model, gallery_loader)
    distances = cdist(query_embeddings, gallery_embeddings, metric=function)
    return distances, query_targets, gallery_targets

def count_positives(query, best_result_ids, gallery_targets):
    counter = 0
    positives_counter = np.zeros(best_result_ids.shape)
    for idx in range(best_result_ids.shape[0]):
        best_result_id = int(best_result_ids[idx])
        if gallery_targets[best_result_id] == query:
            counter += 1
            positives_counter[idx] = counter
    return positives_counter

def mean_average_precision(model, query_loader, gallery_loader, k=-1, function='euclidean'):
    if k == -1:
        k = gallery_loader.dataset.data_length

    distances, query_targets, gallery_targets = get_pairwise_distances(model, query_loader, gallery_loader, function)
    sorted_dists = np.argsort(distances, axis=1)[:,:k] # evaluate only precision@k 

    groundtruths_per_class = gallery_loader.dataset.n_groundtruths
    sum_average_precision = 0.0

    for query_id in tqdm(range(query_targets.shape[0])):
        query = int(query_targets[query_id])
        
        normalizer = min(groundtruths_per_class[query], k)
        best_result_ids = sorted_dists[query_id,:normalizer]

        positives_counter = count_positives(query, best_result_ids, gallery_targets)
        results_counter = np.arange(1, normalizer+1)

        scores = positives_counter/results_counter
        average_precision = float(scores.sum()) / normalizer
        sum_average_precision += average_precision

    return sum_average_precision / query_targets.shape[0]

