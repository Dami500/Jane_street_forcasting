
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap.umap_ as umap
import pyarrow.parquet as pq
import warnings

warnings.filterwarnings('ignore')

def reduce_dimensionality(df)->list:
    df.fillna(0, inplace = True)
    df = df.iloc[:, 4:83]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    pca = PCA(n_components=0.95, random_state=42)  # Retaining 95% variance
    pca_transformed = pca.fit_transform(scaled)
    print(pca.explained_variance_ratio_)
    feature_importance =np.abs(pca.components_)
    for i, component in enumerate(feature_importance):
   	 print(f"Top features for PC{i+1}:")
   	 top_features = np.argsort(component)[::-1]
   	 print(df.columns[top_features[ :10]])

   #plt.figure(figsize=(8, 5))
   #plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
   #plt.show()
   # umap_model = umap.UMAP(n_components=2)
    #embeddings = umap_model.fit_transform(pca_transformed)
   # print(embeddings)
   # embeddings_df = pd.DataFrame(embeddings, columns=['UMAP1', 'UMAP2'])
   # feature_correlations = data.corrwith(embeddings_df['UMAP1'])
   # print(feature_correlations.sort_values(ascending=False).head())

   # plt.scatter(X_umap[:, 0], X_umap[:, 1], s=1, alpha=0.5)
    #plt.title("UMAP Embedding")
   # plt.show()

print('script is working')
train = ['/mnt/disk2/train.parquet/partition_id=3/part-0.parquet',
'/mnt/disk2/train.parquet/partition_id=8/part-0.parquet', '/mnt/disk2/train.parquet/partition_id=5/part-0.parquet',
'/mnt/disk2/train.parquet/partition_id=4/part-0.parquet', '/mnt/disk2/train.parquet/partition_id=6/part-0.parquet',
'/mnt/disk2/train.parquet/partition_id=1/part-0.parquet', '/mnt/disk2/train.parquet/partition_id=0/part-0.parquet',
'/mnt/disk2/train.parquet/partition_id=7/part-0.parquet', '/mnt/disk2/train.parquet/partition_id=2/part-0.parquet',
'/mnt/disk2/train.parquet/partition_id=9/part-0.parquet']
data = pd.read_parquet(train[1])
reduce_dimensionality(data)

