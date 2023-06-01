# %% [markdown]
# ## Import Libraries

# %% [code] {"tags":[],"execution":{"iopub.status.busy":"2023-05-31T07:11:56.913021Z","iopub.execute_input":"2023-05-31T07:11:56.913510Z","iopub.status.idle":"2023-05-31T07:12:05.606065Z","shell.execute_reply.started":"2023-05-31T07:11:56.913470Z","shell.execute_reply":"2023-05-31T07:12:05.605027Z"}}
# for loading/processing the images  
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications import vgg16, inception_v3 

# models 
from keras.applications.vgg16 import VGG16 
from keras.applications.inception_v3 import InceptionV3 
from keras.models import Sequential, Model
from keras.layers import Flatten

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Define variables and functions

# %% [code] {"execution":{"iopub.status.busy":"2023-05-31T07:12:05.853781Z","iopub.execute_input":"2023-05-31T07:12:05.854025Z","iopub.status.idle":"2023-05-31T07:12:05.859388Z","shell.execute_reply.started":"2023-05-31T07:12:05.854003Z","shell.execute_reply":"2023-05-31T07:12:05.858486Z"}}
def extract_features(file,model,preprocess_input):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True,verbose=0)
    return features

# %% [code] {"execution":{"iopub.status.busy":"2023-05-31T07:12:05.860613Z","iopub.execute_input":"2023-05-31T07:12:05.861417Z","iopub.status.idle":"2023-05-31T07:12:05.871014Z","shell.execute_reply.started":"2023-05-31T07:12:05.861386Z","shell.execute_reply":"2023-05-31T07:12:05.870104Z"}}
def plot_PCA(pca):
    # Get explained variance ratio
    explained_var = pca.explained_variance_ratio_

    # Plot the explained variance
    plt.plot(np.cumsum(explained_var))
    plt.ylim(0.2,1.0)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)

    # Add a line indicating 0.95 position
    plt.axhline(y=0.95, color='r', linestyle='--', label='0.95 Variance Threshold')
    plt.legend()
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-05-31T07:12:05.872481Z","iopub.execute_input":"2023-05-31T07:12:05.873035Z","iopub.status.idle":"2023-05-31T07:12:05.885801Z","shell.execute_reply.started":"2023-05-31T07:12:05.873006Z","shell.execute_reply":"2023-05-31T07:12:05.884893Z"}}
# holds the cluster id and the images { id: [images] }
def group_id_images(filenames, kmeans):
    groups = {}
    for file, cluster in zip(filenames,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    return groups

# %% [code] {"execution":{"iopub.status.busy":"2023-05-31T07:12:05.886967Z","iopub.execute_input":"2023-05-31T07:12:05.887553Z","iopub.status.idle":"2023-05-31T07:12:05.897809Z","shell.execute_reply.started":"2023-05-31T07:12:05.887505Z","shell.execute_reply":"2023-05-31T07:12:05.896861Z"}}
def cluster_results(k,x,filenames,df_test):
    # cluster feature vectors with k
    kmeans = KMeans(n_clusters=k, random_state=22)
    kmeans.fit(x)
    groups = group_id_images(filenames, kmeans)
    
    weight = []
    for i in range(len(groups)):
        df_cluster = pd.DataFrame(groups[i])
        df_cluster.rename(columns={0:'File Name'},inplace=True)
        results = df_cluster.merge(df_test,how='left',on='File Name')['Image Result'].value_counts(normalize=True)
        if len(results) == 1:
            weight.append(len(groups[i]))
        elif results[0] >= results[1]:
            weight.append(len(groups[i])*results[0])
        else:
            weight.append(len(groups[i])*results[1])

        print(f"Cluster {i} results: total {len(groups[i])} images")
        print(f"{results}")
        
    avg_weight = sum(weight)/len(df_test)
    print(f"Average cluster homogeneity is {avg_weight}")
    return avg_weight

# %% [code] {"execution":{"iopub.status.busy":"2023-05-31T07:12:05.898935Z","iopub.execute_input":"2023-05-31T07:12:05.899439Z","iopub.status.idle":"2023-05-31T07:12:05.908936Z","shell.execute_reply.started":"2023-05-31T07:12:05.899409Z","shell.execute_reply":"2023-05-31T07:12:05.908024Z"}}
def plot_avg_cluster_homo(k_weight):
    plt.plot(k_weight.keys(), k_weight.values())
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Average cluster homogeneity')
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-05-31T07:12:05.910252Z","iopub.execute_input":"2023-05-31T07:12:05.910755Z","iopub.status.idle":"2023-05-31T07:12:05.921133Z","shell.execute_reply.started":"2023-05-31T07:12:05.910725Z","shell.execute_reply":"2023-05-31T07:12:05.920077Z"}}
# function that lets you view a cluster (based on identifier)        
def view_cluster(cluster,groups):
    num_images = min(15, len(groups[cluster]))
    num_cols = 3
    num_rows = math.ceil(num_images / num_cols)

    plt.figure(figsize=(num_cols * 8, num_rows * 8))

    for index, file in enumerate(groups[cluster][:num_images]):
        plt.subplot(num_rows, num_cols, index + 1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
        
        # Add the file name as a title
        plt.title(file.split('/')[-1], fontsize=20)

    plt.tight_layout()