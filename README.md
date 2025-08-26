# Patent_Topic_modeling_and_clustering
Topic modeling and clustering patents to discover hidden innovations trends.

This project implements topic modeling and clustering of patent abstracts to uncover key technological themes and trends. Using Latent Dirichlet Allocation (LDA) for topic modeling and KMeans for clustering, it provides insights into how patents group into thematic clusters and evolve over time.

The pipeline includes:

Text Preprocessing with SpaCy (lemmatization, stopword removal, custom domain stopwords).

Topic Modeling with Gensim’s LDA to identify latent themes in patent abstracts.

Optimal Topic Selection using coherence scores.

Clustering with KMeans, evaluated by silhouette score and the elbow method.

Visualization:

Coherence vs. number of topics

Word clouds for topics

Patent cluster distributions

Time-series trends per cluster

Growth Analysis: Computes Compound Annual Growth Rate (CAGR) for each cluster.

Cluster Profiling: Extracts dominant topics and top words per cluster.

This tool is particularly useful for technology trend analysis, R&D strategy, and patent landscaping.
