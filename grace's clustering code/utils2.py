import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib
import glob
import tensorflow as tf
import pandas as pd
from skimage import io
import h5py
from sklearn import cluster
from sklearn.decomposition import PCA
from skimage import transform
from scipy.spatial.distance import pdist, squareform
import igraph as ig
import louvain
import sys


def graph(image, patch_size, overlapping, embedding_type, cluster_type, cutoff = .8, nclusters = 3):

    # get patches of image into a dictionary with RiCj as the keys
    print("making patch size " + str(patch_size))
    if not overlapping:
        patch_dict = get_patches(image, patch_size)
    else:
         sys.exit("This capability doesn't exist yet")
    

    # make df
    chip_df = pd.DataFrame()
    chip_df.index = patch_dict.keys()
    chip_df['chip ID'] = patch_dict.keys()
    chip_df['image data'] = patch_dict.values()
    chip_df['image data'] = chip_df['image data'].apply(lambda x: x.flatten())


    # get embedding data
    print("embedding using " + embedding_type)
    if embedding_type == "image_net":
        model_input_shape = patch_size + (3,)
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        base_model = tf.keras.applications.vgg16.VGG16(input_shape=model_input_shape, include_top=False, weights='imagenet')
        base_model.trainable = False

        # add embeddings to dataframe
        chip_emb_dict={}
        for k,v in patch_dict.items():
            chip_emb_dict[k] = img_embedding(v, preprocess_input, base_model, model_input_shape[:2]).numpy().flatten()
    else:
        sys.exit("This capability doesn't exist yet")
        
    chip_df['image encoding data'] = chip_emb_dict.values()
    chip_df['image encoding data'] = chip_df['image encoding data'].apply(lambda x: x.flatten())

    # format chip df
    inv_d, chip_df = format_chips(chip_df)

    # clustering
    print("clustering using " + cluster_type)
    if cluster_type == "louvain":
        results, sim = louvain_cluster(inv_d, chip_df, cutoff)
        data_flat = sim.flatten()

        # merge chip_df and clustering results
        results['row'] = results['chip ID'].apply(lambda x: int(x.split('C')[0].replace('R','')))
        results['col'] = results['chip ID'].apply(lambda x: int(x.split('C')[1]))
        results = results.merge(chip_df[['chip ID','image data']], on='chip ID')

        cluster_graph(results , modality= "image encoding data")

        overlay(results, data_flat,cutoff, modality= "image encoding data")
    elif cluster_type == "agglo":
        results = pd.DataFrame()
        results["chip ID"] = chip_df["chip ID"]
        chip_embeddings = np.vstack(chip_df["image encoding data"].to_numpy())

        print("number of clusters: " + str(nclusters))
        labels = cluster.AgglomerativeClustering(n_clusters=nclusters).fit(chip_embeddings)
        results['Agglomerative_Clusters'] = labels.labels_
        results['row'] = results['chip ID'].apply(lambda x: int(x.split('C')[0].replace('R','')))
        results['col'] = results['chip ID'].apply(lambda x: int(x.split('C')[1]))
        results = results.merge(chip_df[['chip ID','image data']], on='chip ID')

        cluster_graph(results, modality= 'Agglomerative_Clusters')

        overlay(results, modality= 'Agglomerative_Clusters')

    else:
        sys.exit("this is not one of the clustering options")


def cluster_graph(results, modality, min_cluster_members = 3, cmap = "tab10"):
    image_matrix = np.vstack(results["image data"])
    pca = PCA(n_components=2)
    pca_dat = pd.DataFrame(pca.fit_transform(image_matrix),  columns=['PC1', 'PC2',])
    pca_dat["cluster"] = results[modality]


    vc = results[modality].value_counts()
    max_cluster_idx = len(vc.loc[vc>=min_cluster_members])+1

    masked_df = pca_dat.mask(pca_dat["cluster"] > max_cluster_idx)

    plt.figure(figsize=(4, 4))
    plt.scatter(masked_df["PC1"], masked_df["PC2"], c=masked_df["cluster"], cmap = cmap, 
                norm = matplotlib.colors.BoundaryNorm(range(0,max_cluster_idx+1), ncolors=max_cluster_idx))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tick_params(axis='both', which='both', left=False, bottom=False, 
                labelleft=False, labelbottom=False) 
    plt.grid(False) 
    plt.show()





def get_patches(image, patch_size):

    # get image into RGB mode if not in it already
    if image.mode != 'L':
        image_array = np.array(image)
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
        image_array = np.uint8(image_array) 
        image = Image.fromarray(image_array, 'L')

    
    patches = {}
    # Calculate the number of patches along each dimension
    width, height = image.size
    num_patches_x = int(width // patch_size[0])
    num_patches_y = int(height // patch_size[1])

    n = 0
    m = 0
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            left = i * patch_size[0]
            upper = j * patch_size[1]
            right = left + patch_size[0]
            lower = upper + patch_size[1]

            patchname =  f"R{m}C{n}"
            m = m+1
            if m == num_patches_x:
                m= 0
                n = n + 1

            # Extract patch
            patch = image.crop((left, upper, right, lower))
            patches[patchname] = np.array(patch)

    return patches


def img_embedding(image, preprocess_input, base_model, model_input_size=(32,32)):
    #x = tf.image.resize(image, size=model_input_size)
    if len(image.shape) == 2:  # If image is (32, 32), add channels
        image = tf.expand_dims(image, axis=-1)  # Now shape is (32, 32, 1)
        image = tf.image.grayscale_to_rgb(image)  # Convert grayscale to RGB (32, 32, 3)
    x = preprocess_input(image)
    x = base_model(np.expand_dims(x, axis=0))
    return x 


def format_chips(chip_df):
    m = sum(chip_df["chip ID"].str.count("R\d+C0"))
    n = sum(chip_df["chip ID"].str.count("R0C\d+"))
    print("Number of rows: ", m)
    print("Number of columns: ", n)
    d = {}
    _ = 0
    for i in range(m):
        for j in range(n):
            d["R" + str(i) + "C" + str(j)] = _
            _ += 1

    inv_d = {v: k for k, v in d.items()}

    chip_df = chip_df.replace({"chip ID": d}).sort_values(by=['chip ID'])
    chip_df = chip_df.infer_objects(copy=False).replace({"chip ID": inv_d})
    return inv_d, chip_df 


def louvain_cluster(inv_d, chip_df, c ):
    #c = .5
    print(f"Clustering individual modalities using cutoff={c}")
    results = pd.DataFrame()
    results["chip ID"] = chip_df["chip ID"]

    #sims = {}
    sim = squareform(1 - pdist(np.stack(chip_df["image encoding data"].values), metric="cosine"))
    adj = (sim >= c).astype(int)
    g = ig.Graph.Adjacency(adj, mode='undriected')
    part = louvain.find_partition(g, louvain.ModularityVertexPartition, seed=42)
    pred = {}
    for s in range(len(part)):
        for node in part[s]:
            pred[node] = s
    #sims[labels[i]] = sim
    df = pd.DataFrame({"chip ID": pred.keys(), "image encoding data": pred.values()})
    df = df.sort_values(by=['chip ID']).replace({"chip ID": inv_d})
    results = results.merge(df, left_on="chip ID", right_on="chip ID")

    return results, sim



def overlay(results, data_flat = None, cutoff = None, dpi = 150, W = 8, H = 3, cmap = "tab10", min_cluster_members = 3, 
            figure_name = "overlay.png", modality = "AgglomerativeClustering(n_clusters=3)"):
    # enumerate chips in each row and column
    n_rows = len([x for x in results['chip ID'].tolist() if 'R0C' in x])
    n_columns = len([x for x in results['chip ID'].tolist() if x[-2:]=='C0'])

    # create grid to upscale cluster blocks to match size of image
    grid = np.zeros((results['col'].max()+1,results['row'].max()+1), dtype='int')
    for c in results[modality].value_counts().index:
        grid[results.loc[results[modality]==c]['row'].to_numpy(),results.loc[results[modality]==c]['col'].to_numpy()]=c

    # filter clusters with too few members
    vc = results[modality].value_counts()
    max_cluster_idx = len(vc.loc[vc>=min_cluster_members])+1

    if data_flat is not None:
        # plot definition
        fig, ax = plt.subplots(1, 2, figsize=(W, H), dpi=dpi)
        ax[0].hist(data_flat, bins=100)
        ax[0].axvline(x=cutoff, color='red', linestyle='--')

        # resize to fit image
        dim =  int(np.sqrt(np.array(results.iloc[0]['image data']).shape[0]))
        updim = (int(n_rows*dim), int(n_columns*dim))
        g = transform.resize_local_mean(grid, updim, preserve_range=True).astype('int')

        # apply mask
        g=np.ma.masked_where(g>max_cluster_idx, g)

        # plot chips
        chips = format_chips_plot(results, shape=np.array(updim), dim=dim)
        ax[1].imshow(chips, interpolation='none', cmap=cm.gray)

        # plot cluster blocks
        ax[1].imshow(g, alpha=0.5, cmap=cmap, interpolation='none', 
                norm = matplotlib.colors.BoundaryNorm(range(0,max_cluster_idx+1), ncolors=max_cluster_idx)
                )
        ax[1].axis('off')
        plt.show()
    else:
        fig = plt.figure(figsize=(W, H), dpi=dpi)
        ax = fig.gca()
        
        # resize to fit image
        dim =  int(np.sqrt(np.array(results.iloc[0]['image data']).shape[0]))
        updim = (int(n_rows*dim), int(n_columns*dim))
        g = transform.resize_local_mean(grid, updim, preserve_range=True).astype('int')

        # apply mask
        g=np.ma.masked_where(g>max_cluster_idx, g)

        # plot chips
        chips = format_chips_plot(results, shape=np.array(updim), dim=dim)
        ax.imshow(chips, interpolation='none', cmap=cm.gray)

        # plot cluster blocks
        ax.imshow(g, alpha=0.5, cmap=cmap, interpolation='none', 
                norm = matplotlib.colors.BoundaryNorm(range(0,max_cluster_idx+1), ncolors=max_cluster_idx)
                )

        # plot and save
        plt.axis('off')
        plt.margins(0.2)
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.show()




def format_chips_plot(results, shape, dim):
    chips = np.zeros(shape, dtype='int')
    
    for i, drow in results.sort_values(['row','col']).iterrows():
        r = drow['row']
        rstart = r+((dim-1)*r)
        rend = r+((dim-1)*r)+dim
        
        c = drow['col']
        cstart = c+((dim-1)*c)
        cend = c+((dim-1)*c)+dim
    
        img = np.array(drow['image data']).reshape(dim, dim) 
        chips[rstart:rend, cstart:cend] = img
    return chips