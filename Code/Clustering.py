import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def find_most_important_keywords(role_keywords, n_clusters, max_keywords_per_cluster):
    # Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(role_keywords)

    # Clustering Algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    # Apply Clustering
    kmeans.fit(tfidf_matrix)

    # Label Clusters
    cluster_labels = kmeans.labels_

    # Analyze Clusters
    clustered_keywords = {}
    for i, label in enumerate(cluster_labels):
        if label not in clustered_keywords:
            clustered_keywords[label] = []
        clustered_keywords[label].append(role_keywords[i])

    # Control the number of keywords in each cluster
    filtered_keywords = {}
    for cluster, keywords in clustered_keywords.items():
        filtered_keywords[cluster] = keywords[:max_keywords_per_cluster]

    return filtered_keywords

df = pd.read_excel('KeywordsList.xlsx')

# Extract keywords for each role
data_analyst_keywords = df['Data Analyst'].tolist()
data_engineer_keywords = df['Data Engineer'].tolist()
data_scientist_keywords = df['Data Scientist'].tolist()

# Define the number of clusters and max keywords per cluster for each role
da_clusters = 4  
de_clusters = 4  
ds_clusters = 4 

max_keywords_per_cluster = 15  # Set your desired maximum number of keywords per cluster

# Find most important keywords for each role
important_keywords_data_analyst = find_most_important_keywords(data_analyst_keywords, da_clusters, max_keywords_per_cluster)
important_keywords_data_engineer = find_most_important_keywords(data_engineer_keywords, de_clusters, max_keywords_per_cluster)
important_keywords_data_scientist = find_most_important_keywords(data_scientist_keywords, ds_clusters, max_keywords_per_cluster)

# Example: Printing the clustered keywords for Role 1
for cluster, keywords in important_keywords_data_analyst.items():
    print(f"Cluster {cluster}: {', '.join(keywords)}")

for cluster, keywords in important_keywords_data_engineer.items():
    print(f"Cluster {cluster}: {', '.join(keywords)}")

for cluster, keywords in important_keywords_data_scientist.items():
    print(f"Cluster {cluster}: {', '.join(keywords)}")


