# --- Imports ---
import re
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import multiprocessing

# --- Stopwords ---
stop_words = set(STOP_WORDS)
custom_stopwords_ice = {
    # --- Patent boilerplate ---
    "method","methods","system","systems","device","devices","apparatus","apparatuses","process","processes",
    "means","invention","inventions","embodiment","embodiments","arrangement","arrangements","configuration","configurations",
    "operation","operations","structure","structures","present","described","according","prior","art","example","examples",
    "thereof","wherein","therein","thereby","therewith","said","fig","figure","figures","claim","claims","mean",
    
    # --- Generic filler / functional terms ---
    "provide","provided","provides","improve","improved","improvement","efficient","efficiency","enable","enabled","comprise","comprising",
    "including","include","includes","having","configured","applied","formed","form","forms","use","used","using","utilize","utilized",
    "least","greater","less","first","second","third","plurality","one","two","three","four","five","six","seven","eight","nine","ten",
    *(str(i) for i in range(1, 101)),

    # --- Domain-generic technical terms ---
    "engine","engines","pistons","cylinders","valves","crankshaft","camshaft","intake","exhaust",
    "chambers","stroke","ignition","injector","fuel","gasoline","diesel","air","water",
    "burning","torque","rpm","power","output","input","internal","drive","driven","shaft","rotational","rotation","speed",
    "mechanism","mechanical","gear","transmission","lubrication","lubricant","oil","pump","flow","system",

    # --- ICE-specific physical/thermal environment terms ---
    "temperature","pressure","cooling","heat","thermal","expansion","exhaust","gas","gases","manifold","outlet","inlet","signal",
    "housing","mount","bracket","cover","case","ring","bearing","materials","component","components","valve"
    "element","elements","member","members","portion","unit","units","assembly","assemblies","module","modules","path","carbon","comprises","value",

    # --- General scientific/research terms ---
    "study","result","results","conclusion","conclusions","data","approach","approaches","effect","effects","analysis","model","models",
    "proposed","paper","investigation","report","observed","behavior","properties","also", "amount","problem"
}

stop_words.update(custom_stopwords_ice)


# --- Preprocessing function ---
def preprocess_batch(texts, nlp_model):
    processed_texts = []
    for doc in nlp_model.pipe(texts, batch_size=500, n_process=multiprocessing.cpu_count()):
        tokens = [token.lemma_ for token in doc
                  if token.is_alpha and token.lemma_ not in stop_words and len(token) > 3]
        processed_texts.append(tokens)
    return processed_texts


# --- Function to compute coherence ---
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import multiprocessing

def compute_coherence_gensim(texts, corpus, dictionary, topic_range, passes=10, iterations=200, threshold=0.005):
    coherence_scores = []

    # --- Step 1: Compute all coherence scores ---
    for k in topic_range:
        lda = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            workers=multiprocessing.cpu_count(),
            passes=passes,
            iterations=iterations,
            random_state=42
        )
        cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence="c_v")
        score = cm.get_coherence()
        coherence_scores.append((k, score))
        print(f"Topics: {k}, Coherence: {score:.4f}")

    # --- Step 2: Find optimal number of topics ---
    best_k = None
    first_local_max = None
    for i in range(1, len(coherence_scores) - 1):
        prev_k, prev_score = coherence_scores[i]
        prev_prev_k, prev_prev_score = coherence_scores[i - 1]
        next_k, next_score = coherence_scores[i + 1]

        # Check for local maximum
        if prev_score > prev_prev_score and prev_score > next_score:
            if (prev_score - prev_prev_score > threshold) and (prev_score - next_score > threshold):
                best_k = prev_k
                print(f"\n✅ Found strong local maximum at k={best_k}, coherence={prev_score:.4f}")
                return best_k, coherence_scores
            if first_local_max is None:
                first_local_max = (prev_k, prev_score)

    # --- Step 3: Fallback cases ---
    if best_k is None:
        if first_local_max:
            best_k = first_local_max[0]
            print(f"\n⚠️ No strong maxima found. Returning first local maximum at k={best_k}, coherence={first_local_max[1]:.4f}")
        else:
            best_k = coherence_scores[-1][0]
            print(f"\n⚠️ No maxima at all. Returning last tried k={best_k}, coherence={coherence_scores[-1][1]:.4f}")

    return best_k, coherence_scores


   
# --- Main execution (Windows safe) ---
if __name__ == "__main__":
    # --- Load your data ---
    df = pd.read_csv("C:\\Users\\PARVAZ\\Desktop\\New folder (2)\\Book1_cleaned_ice.csv", dtype = {31:str})  # replace with your file path
    texts = df['Abstract'].dropna().astype(str).tolist()

    # --- Load SpaCy model ---
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # --- Preprocess texts ---
    print("Preprocessing texts...")
    processed_texts = preprocess_batch(texts, nlp)

    # --- Create dictionary & corpus ---
    print("Creating dictionary and corpus...")
    id2word = corpora.Dictionary(processed_texts)
    id2word.filter_extremes(no_below=5, no_above=0.4)
    processed_texts = [[w for w in text if w in id2word.token2id] for text in processed_texts]
    gensim_corpus = [id2word.doc2bow(text) for text in processed_texts]

    # --- Coherence search for optimal topics ---
    print("Computing coherence for different topic numbers...")
    topic_range = range(3, 13)  # test 5,7,9,...15
    

    # --- Select optimal topic number ---
    optimal_topics , coherence_score= compute_coherence_gensim(processed_texts, gensim_corpus, id2word, topic_range)
    print(f"\n✅ Optimal number of topics based on coherence: {optimal_topics}")
    x = [k for k, _ in coherence_score]
    y = [score for _, score in coherence_score]
    plt.plot(x, y, marker = 'o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Finding Optimal Number of Topics')
    plt.tight_layout()
    plt.savefig('optima_topic.png', dpi = 300)

    # --- Train final LDA model ---
    print("Training final LDA model...")
    final_lda = LdaMulticore(
        corpus=gensim_corpus,
        id2word=id2word,
        num_topics=optimal_topics,
        workers=multiprocessing.cpu_count(),
        passes=20,
        iterations=200,
        random_state=42
    )

    # --- Print top words per topic ---
    for k in range(final_lda.num_topics):
        print(f"Topic {k+1}: {[w for w, _ in final_lda.show_topic(k, topn=5)]}")

    # --- Optional: Word clouds ---     #In this plt I used 5 instead of optimal topic
    fig, axes = plt.subplots(1, optimal_topics, figsize=(5*optimal_topics, 5))
    for k in range(final_lda.num_topics):
        freq = {w: float(p) for w, p in final_lda.show_topic(k, topn=7)}
        wc = WordCloud(background_color='white').generate_from_frequencies(freq)
        axes[k].imshow(wc, interpolation='bilinear')
        axes[k].axis("off")
        axes[k].set_title(f"Topic {k+1}")
    plt.tight_layout()
    plt.savefig('Topic Modeling')
    print('Clusterring Starts...')
    topic_distributions = [final_lda.get_document_topics(bow, minimum_probability=0)
                           for bow in gensim_corpus]
    X = np.array([[prob for _, prob in doc] for doc in topic_distributions])

    # --- KMeans clustering ---
    K_range = range(2, 11)
    wcss = []
    silhouette_scores = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
    
    plt.figure(figsize=(12,5))

# WCSS / Elbow Method
    plt.subplot(1, 2, 1)
    plt.plot(K_range, wcss, marker='o')
    plt.title('Elbow Method (WCSS)')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')

# Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.savefig("cluster_metrics.png", dpi=300)  # Save the figure
    plt.show()


    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"✅ Optimal number of clusters: {best_k}")

    print('Training Best KMeans...')

    kmeans_final = KMeans(n_clusters=best_k, random_state=42)
    final_labels = kmeans_final.fit_predict(X)
    df.dropna(subset=['Abstract'], inplace= True)
    df['Cluster'] = final_labels

    df['Cluster'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title("Number of Patents per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Patents")
    plt.savefig('Number of patents Per Cluster.png', dpi = 300)

    cluster_year_counts = df.groupby(['Cluster', 'Publication Year']).size().reset_index(name='Count')

# رسم روند زمانی با پالت رنگی جدید
    plt.figure(figsize=(12, 6))
    sns.lineplot(
    data=cluster_year_counts,
    x='Publication Year',
    y='Count',
    hue='Cluster',
    marker='o',
    palette='Set2'  # 🌈 پالت رنگی جدید
    )
    plt.title("Patent Trends Over Time per Cluster", fontsize=14)
    plt.ylabel("Number of Patents")
    plt.xlabel("Year")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Patent Trend over time per cluster.png', dpi = 300)

    print('Start CGAR Calculations...')
    def calculate_cagr(start_value, end_value, periods):
        if start_value == 0 or periods == 0:
            return 0
        return ((end_value / start_value) ** (1 / periods)) - 1

    cagr_results = []

    for cluster_id in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster_id]
        yearly_counts = cluster_data['Publication Year'].value_counts().sort_index()
    
        if len(yearly_counts) >= 2:
            start_year = yearly_counts.index[0]
            end_year = yearly_counts.index[-1]
            start_value = yearly_counts.iloc[0]
            end_value = yearly_counts.iloc[-1]
            periods = end_year - start_year
            cagr = calculate_cagr(start_value, end_value, periods)
            cagr_results.append((cluster_id, start_year, end_year, cagr))

# --- Save to CSV ---
    output_file = "cluster_cagr.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Cluster", "Start Year", "End Year", "CAGR"])
        for cid, sy, ey, cagr in cagr_results:
            writer.writerow([cid, sy, ey, f"{cagr:.4f}"])  # store CAGR as decimal
    print(f"✅ CAGR results saved to {output_file}")

    print('Finding dominant Topics of each cluster...')


# --- Get topic distributions ---
   
    import csv

# Convert sparse topic distribution to dense vector
    topic_distributions = []
    num_topics = final_lda.num_topics

    for doc in gensim_corpus:
        dist = final_lda.get_document_topics(doc, minimum_probability=0.0)
        dense_vector = np.zeros(num_topics)
        for topic_id, prob in dist:
            dense_vector[topic_id] = prob
        topic_distributions.append(dense_vector)

    X = np.array(topic_distributions)

# Add topic vector column to DataFrame
    df['topic_vector'] = topic_distributions

# Number of clusters
    n_clusters = df['Cluster'].nunique()

# --- Save dominant topics ---
    output_file = "cluster_dominant_topics.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Cluster", "Dominant Topics", "Top Words per Topic"])

        for cluster_id in range(n_clusters):
        # Select topic vectors belonging to this cluster
            cluster_vectors = np.array(df[df['Cluster'] == cluster_id]['topic_vector'].tolist())

            if cluster_vectors.size == 0:
                continue

        # Compute mean topic distribution
            mean_vector = cluster_vectors.mean(axis=0)

        # Get strongest topics (top 2)
            top_topic_indices = mean_vector.argsort()[::-1][:2]

            dominant_topics = []
            top_words_list = []
            for topic_idx in top_topic_indices:
                topic_words = [word for word, _ in final_lda.show_topic(topic_idx, topn=10)]
                dominant_topics.append(str(topic_idx))
                top_words_list.append(" | ".join(topic_words))

            writer.writerow([cluster_id, ", ".join(dominant_topics), " || ".join(top_words_list)])

    print(f"✅ Dominant topics per cluster saved to {output_file}")
