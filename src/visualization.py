import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from typing import Dict, List

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from utils import clean_text

def generate_pie_chart(data: pd.Series, title: str) -> plt.Figure:
    """Generate a pie chart from value counts"""
    fig, ax = plt.subplots()
    ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(title)
    return fig

def generate_wordcloud(text: str, title: str) -> plt.Figure:
    """Generate a word cloud from text"""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title)
    return fig

def generate_temporal_plot(
        x_values: List,
        y_values: List,
        text_values: List,
        y_labels: Dict,
        title: str,
        x_label: str,
        y_label: str,
        color: str = 'purple'
) -> plt.Figure:
    """Generate a temporal plot with annotations"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_values, y_values, marker='o', linestyle='-', color=color, markersize=8)

    # Add text annotations
    for x, y, text in zip(x_values, y_values, text_values):
        text = clean_text(text, 32)
        ax.text(x, y + 0.1, text,
                ha='center', va='bottom', rotation=45, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    # Style the plot
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(list(y_labels.keys()))
    ax.set_yticklabels(list(y_labels.values()))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig

def plot_cluster_topics(cluster_topics):
    """Plot the key topics for each cluster."""
    if not cluster_topics:
        return None

    plt.figure(figsize=(10, 6))
    clusters = list(cluster_topics.keys())
    topics = list(cluster_topics.values())

    # Create horizontal bar plot
    y_pos = range(len(clusters))
    plt.barh(y_pos, [5] * len(clusters), color='skyblue')  # Dummy values for bars
    plt.yticks(y_pos, clusters)

    # Add topic text
    for i, (cluster, topic) in enumerate(cluster_topics.items()):
        plt.text(0.5, i, topic, ha='left', va='center', fontsize=10)

    plt.title('Key Topics for Each Cluster')
    plt.xlabel('Top Terms')
    plt.xlim(0, 1)  # Just for spacing
    plt.gca().xaxis.set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.tight_layout()
    return plt.gcf()


def plot_grouped_histogram(df):
    """Plot grouped histogram for word and sentence counts."""
    plot_df = df.melt(id_vars=['Text ID'],
                      value_vars=['Sentences', 'Words'],
                      var_name='Metric',
                      value_name='Count')

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df,
                     x='Text ID',
                     y='Count',
                     hue='Metric',
                     palette=['#1f77b4', '#ff7f0e'])

    plt.title('Word and Sentence Counts by Text')
    plt.xlabel('Text ID')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Metric')

    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points')

    plt.tight_layout()
    return plt.gcf()


def plot_dendrogram(linkage_matrix, labels):
    """Plot a dendrogram from linkage matrix."""
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix,
               labels=labels,
               orientation='left',
               leaf_font_size=10,
               color_threshold=0.7)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')
    plt.tight_layout()
    return plt.gcf()


def get_insights(sentiment_df: pd.DataFrame, outcome_df: pd.DataFrame) -> Dict:
    """Generate visual insights from analysis results"""
    insights = {}

    # Sentiment distribution
    senti_counts = sentiment_df['Sentiment'].value_counts()
    insights['sentiment_pie'] = generate_pie_chart(senti_counts, "Sentiment Distribution")

    # Outcome distribution
    out_counts = outcome_df['Outcome'].value_counts()
    insights['outcome_pie'] = generate_pie_chart(out_counts, "Outcome Distribution")

    # Combined sentiment-outcome distribution
    combined_df = outcome_df.copy()
    combined_df['Sentiment'] = sentiment_df['Sentiment']
    combined_df['Senti_Outco'] = combined_df['Sentiment'] + ' & ' + combined_df['Outcome']
    combined_counts = combined_df['Senti_Outco'].value_counts()
    insights['combined_pie'] = generate_pie_chart(combined_counts, "Sentiment & Outcome Distribution")

    # Word clouds
    insights['pos_wordcloud'] = generate_wordcloud(
        ' '.join(sentiment_df.loc[sentiment_df['Sentiment'] == 'Positive']['Explain']),
        "Positive Sentiment Topics"
    )

    insights['neg_wordcloud'] = generate_wordcloud(
        ' '.join(sentiment_df.loc[sentiment_df['Sentiment'] == 'Negative']['Explain']),
        "Negative Sentiment Topics"
    )

    insights['resolved_wordcloud'] = generate_wordcloud(
        ' '.join(str(ele) for ele in outcome_df.loc[outcome_df['Outcome'] == 'Issue Resolved']['Issue']),
        "Resolved Issue Topics"
    )

    insights['action_wordcloud'] = generate_wordcloud(
        ' '.join(str(ele) for ele in outcome_df.loc[outcome_df['Outcome'] == 'Follow-up Action Needed']['Issue']),
        "Follow-up Action Topics"
    )

    return insights
