from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def get_similar_images(full_original_data, encoded_full_set, image_index, num_similar_images):
    calculate_KNN_and_cosine(np.array(encoded_full_set), full_original_data, num_similar_images, image_index, input_image_flag=False)


def calculate_KNN_and_cosine(data, full_original_data, num_similar_images, image_index, input_image_flag):
    start = time.time()
    # KNN
    nbrs = NearestNeighbors(n_neighbors=num_similar_images + 1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    if input_image_flag is False:
        knn_indices = indices[image_index, :]
        cos_image_index = image_index
    else:
        knn_indices = indices[data.shape[0] - 1, :]
        cos_image_index = data.shape[0] - 1
    # cosine similarity
    encoded_full_df = pd.DataFrame(data=data)
    cosine_vals = cosine_similarity(encoded_full_df, dense_output=True)
    cosine_indices = np.argpartition(cosine_vals[cos_image_index, :], -1 * num_similar_images - 1)[-1 * num_similar_images - 1:]
    cosine_indices = list(cosine_indices)
    cosine_indices.pop(cosine_indices.index(cos_image_index))
    cosine_indices.insert(0, cos_image_index)
    # overlap
    overlap_indices = list(set(cosine_indices) & set(knn_indices))
    overlap_indices.pop(overlap_indices.index(cos_image_index))
    print('Total Time To Find Similar Images = ' + str(np.round(time.time() - start, 2)) + 's')
    plot_similar_images(num_similar_images, knn_indices, cosine_indices, overlap_indices, full_original_data)


def plot_similar_images(num_similar_images, knn_indices, cosine_indices, overlap_indices, full_original_data):
    if num_similar_images > 5:
        top_images_for_plotting = 5
    else:
        top_images_for_plotting = num_similar_images
    # initialize the plots
    fig, axes1 = plt.subplots(1, top_images_for_plotting, figsize=(25, 5))
    plt.suptitle('Nearest Neighbors With Minkowski Distance', fontweight='bold', fontsize=20)
    fig2, axes2 = plt.subplots(1, top_images_for_plotting, figsize=(25, 5))
    plt.suptitle('Cosine Metric', fontweight='bold', fontsize=20)
    # figure for overlap between the two sets ---> images with strong similarity
    if overlap_indices:
        fig3, axes3 = plt.subplots(1, top_images_for_plotting, figsize=(25, 5))
        plt.suptitle('KNN and Cosine Overlap', fontweight='bold', fontsize=20)
        iterations = 3
    else:
        iterations = 2

    for x in range(iterations):
        if x == 0:
            index_list = knn_indices[:top_images_for_plotting]
            axes = axes1
        elif x == 1:
            index_list = cosine_indices[:top_images_for_plotting]
            axes = axes2
        elif x == 2:
            index_list = overlap_indices[:top_images_for_plotting]
            axes = axes3
        for index in range(len(index_list)):
            axes[index].imshow(full_original_data[index_list[index]])
            if x == 0 or x == 1:
                if index == 0:
                    axes[index].set_title('Input Image [' + str(index_list[index]) + ']')
                else:
                    axes[index].set_title(str(index_list[index]))
            elif x == 2:
                for _ in range(len(overlap_indices), top_images_for_plotting):
                    axes3[_].set_axis_off()
                axes[index].set_title(str(index_list[index]))
    plt.show()
    print('# Of Similar Image Overlap Between NN and Cosine = ' + str(len(overlap_indices)) + '/' + str(num_similar_images))
