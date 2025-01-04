
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
def reduce_dimensionality(df) -> list:
    df.fillna(0, inplace=True)
    df = df.iloc[:, 4:83]
    
    # Scale the data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    
    # Apply PCA to retain 95% variance
    pca = PCA(n_components=0.95, random_state=42)
    pca_transformed = pca.fit_transform(scaled)
    
    # Get explained variance ratios
    ratios = pca.explained_variance_ratio_
    allowed = []
    for r in ratios:
        if r >= 0.03:  # Include components explaining at least 3% variance
            allowed.append(r)
    
   # print(allowed)
   # print(np.sum(allowed))
    
    # Determine feature importance for each significant principal component
    feature_importance = np.abs(pca.components_)
    features = []
    for i, component in enumerate(feature_importance[:len(allowed)]):
    #    print(f"Top features for PC{i + 1}:")
        top_features = np.argsort(component)[::-1]  # Sort features by importance
        t = list(df.columns[top_features[:10]])  # Get top 10 features
        for f in t:
            features.append(f)
    return features

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
dictionary = {}
for i in range(len(train)):
	data  = pd.read_parquet(train[i])
#	print('Explained Variance Ratio for parquet %s' % i)
	features = reduce_dimensionality(data)
	dictionary['parquet %s' %i] = features
#df = pd.DataFrame(dictionary)
#print(df)
df = pd.DataFrame.from_dict(dictionary, orient='index').transpose()
df = df.apply(pd.value_counts)
df.fillna(0, inplace = True)
print(df)
conditions = [ df['parquet 0'] >= 1, df['parquet 1'] >= 1, df['parquet 2'] >= 1, 
df['parquet 3'] >= 1, df['parquet 4'] >= 1, df['parquet 5'] >= 1, 
df['parquet 6'] >= 1, df['parquet 7'] >= 1, df['parquet 8'] >= 1, df['parquet 9'] >= 1] 
# Check the index of each condition
#for i, cond in enumerate(conditions):
   # print(f'Condition {i} index: ', cond.index)
df_filtered = df.copy() 
for i, cond in enumerate(conditions): 
    df_filtered = df_filtered[cond] 
print(df_filtered.index)
