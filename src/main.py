from analysis_functions import *
from data_loading import *
from evaluation_metrics import *
from visualization import *
from utils import *

def main():
    # File upload section
    with st.sidebar:
        st.header("Upload Transcripts")
        uploaded_files = st.file_uploader(
            "Choose transcript files:",
            type=["txt"],
            accept_multiple_files=True
        )

        if uploaded_files and not st.session_state.analysis_complete:
            st.session_state.documents, st.session_state.agent_docs = load_utterances(uploaded_files)
            st.success(f"Loaded {len(st.session_state.documents)} transcripts")

    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Brief EDA",
        "Sentiment Analysis",
        "Outcome Analysis",
        "Evaluation",
        "Insights"
    ])

    # Brief EDA
    with tab1:
        st.header("Brief EDA")
        if st.button("Run Brief EDA", key="eda_btn"):
            if st.session_state.documents:
                texts = [text for _, text in st.session_state.documents.items()]
                # Analyze texts
                df, linkage_matrix, texts, cluster_topics, cluster_counts = analyze_texts(texts)

                # Display results
                st.header("Text Analysis Results")
                st.dataframe(df)

                # Visualizations
                st.header("Visualizations")

                # Grouped histogram for word and sentence counts
                st.subheader("Word and Sentence Counts")
                fig = plot_grouped_histogram(df)
                st.pyplot(fig)

                # Show topics if available
                if 'Main Topics' in df.columns:
                    st.subheader("Main Topics by TF-IDF")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df['Topics Truncated'] = df['Main Topics'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
                    sns.barplot(data=df, x='Topics Truncated', y='Text ID', ax=ax)
                    plt.xticks(rotation=45, ha='right')
                    plt.xlabel("Top Keywords")
                    plt.ylabel("Text ID")
                    st.pyplot(fig)

                # Hierarchical clustering visualization
                if linkage_matrix is not None:
                    st.subheader("Hierarchical Clustering Dendrogram")
                    st.write("""
                        This dendrogram shows how texts are clustered based on their TF-IDF similarity.
                        Texts that are closer together are more similar in content.
                        """)

                    labels = [f"Text {i + 1}: {text[:30]}..." for i, text in enumerate(texts)]
                    fig = plot_dendrogram(linkage_matrix, labels)
                    st.pyplot(fig)

                    # Cluster topics visualization
                    if cluster_topics:
                        st.subheader("Key Topics for Each Cluster")
                        st.write("""
                            These are the most important terms (by TF-IDF) that characterize each cluster.
                            """)

                        fig = plot_cluster_topics(cluster_topics)
                        st.pyplot(fig)

                        # Display cluster members
                        st.subheader("Cluster Members")
                        for cluster_num in sorted(df['Cluster'].unique()):
                            cluster_df = df[df['Cluster'] == cluster_num]
                            st.markdown(f"**Cluster {cluster_num}** ({len(cluster_df)} texts):")
                            st.dataframe(cluster_df[['Text ID', 'Main Topics']])

                        # Generate and display pie chart
                        cluster_pie_chart = generate_pie_chart(cluster_counts, "Top 6 Clusters by Number of Texts")
                        st.subheader("Cluster Distribution")
                        st.pyplot(cluster_pie_chart)

                # Show raw data
                if st.checkbox("Show raw data"):
                    st.subheader("Raw Data")
                    st.write(df)

            else:
                st.error("No documents uploaded yet. Please upload files first.")

    # Sentiment Analysis Tab
    with tab2:
        st.header("Sentiment Analysis")

        if st.button("Run Sentiment Analysis", key="sentiment_btn"):
            if st.session_state.documents:
                output_data = []
                progress_bar = st.progress(0)
                total_files = len(st.session_state.documents)

                for i, (file_name, text) in enumerate(st.session_state.documents.items()):
                    sentiment_why = get_sentiment(text)
                    output_data.append((
                        file_name,
                        sentiment_why['Sentiment'],
                        sentiment_why['Confidence'],
                        sentiment_why['Why']
                    ))
                    progress_bar.progress((i + 1) / total_files)

                output_df = pd.DataFrame(
                    output_data,
                    columns=['File', 'Sentiment', 'Confidence', 'Explain']
                )
                output_df.to_csv(f'{DATA_DIR}/sentiment_data.csv', index=False)
                st.session_state.sentiment_df = output_df
                st.session_state.analysis_complete = True
                st.success("Sentiment analysis completed!")
                st.dataframe(output_df)
            else:
                st.error("No documents uploaded yet. Please upload files first.")

    # Outcome Analysis Tab
    with tab3:
        st.header("Outcome Analysis")

        if st.button("Run Outcome Analysis", key="outcome_btn"):
            if st.session_state.documents:
                outcomes = []
                progress_bar = st.progress(0)
                total_files = len(st.session_state.documents)

                for i, (file_name, text) in enumerate(st.session_state.documents.items()):
                    issue_outcome = get_outcome(text)
                    outcomes.append((
                        file_name,
                        issue_outcome['Outcome'],
                        issue_outcome['Confidence'],
                        issue_outcome['Issue']
                    ))
                    progress_bar.progress((i + 1) / total_files)

                outcomes_df = pd.DataFrame(
                    outcomes,
                    columns=['File', 'Outcome', 'Confidence', 'Issue']
                )
                outcomes_df.to_csv(f'{DATA_DIR}/outcome_data.csv', index=False)
                st.session_state.outcome_df = outcomes_df
                st.session_state.analysis_complete = True
                st.success("Outcome analysis completed!")
                st.dataframe(outcomes_df)
            else:
                st.error("No documents uploaded yet. Please upload files first.")

    # Evaluation Tab
    with tab4:
        st.header("Evaluation Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment Evaluation")
            std_senti_file = st.file_uploader(
                "Upload standard sentiment results (CSV):",
                type=["csv"],
                key="std_senti"
            )
            prd_senti_file = st.file_uploader(
                "Upload predicted sentiment results (CSV):",
                type=["csv"],
                key="prd_senti"
            )

            if std_senti_file and prd_senti_file:
                std_df = pd.read_csv(std_senti_file)
                prd_df = pd.read_csv(prd_senti_file)
                sentimetrics = sentiment_eval_metrics(std_df, prd_df)

                st.subheader("Confusion Matrix:")
                st.dataframe(sentimetrics["Confusion Matrix"])

                st.subheader("Classification Report:")
                st.dataframe(sentimetrics["Classification Report"].style.format("{:.3f}"))

        with col2:
            st.subheader("Outcome Evaluation")
            std_outco_file = st.file_uploader(
                "Upload standard outcome results (CSV):",
                type=["csv"],
                key="std_outco"
            )
            prd_outco_file = st.file_uploader(
                "Upload predicted outcome results (CSV):",
                type=["csv"],
                key="prd_outco"
            )

            if std_outco_file and prd_outco_file:
                std_df = pd.read_csv(std_outco_file)
                prd_df = pd.read_csv(prd_outco_file)
                outcometrics = outcome_eval_metrics(std_df, prd_df)

                st.subheader("Confusion Matrix:")
                st.dataframe(outcometrics["Confusion Matrix"])

                st.subheader("Classification Report:")
                st.dataframe(outcometrics["Classification Report"].style.format("{:.3f}"))

    # Insights Tab
    with tab5:
        st.header("Useful Insights")
        insight_senti_file = st.file_uploader(
            "Upload predicted sentiment results (CSV):",
            type=["csv"],
            key="insight_senti"
        )
        insight_outco_file = st.file_uploader(
            "Upload predicted outcome results (CSV):",
            type=["csv"],
            key="insight_outco"
        )

        if insight_senti_file and insight_outco_file:
            insight_senti_df = pd.read_csv(insight_senti_file)
            insight_outco_df = pd.read_csv(insight_outco_file)

            insights = get_insights(insight_senti_df, insight_outco_df)

            st.subheader("Distribution Analysis")
            cols = st.columns(3)
            with cols[0]:
                st.pyplot(insights['sentiment_pie'])
            with cols[1]:
                st.pyplot(insights['outcome_pie'])
            with cols[2]:
                st.pyplot(insights['combined_pie'])

            st.subheader("Topic Analysis")
            cols = st.columns(2)
            with cols[0]:
                st.pyplot(insights['pos_wordcloud'])
            with cols[1]:
                st.pyplot(insights['neg_wordcloud'])

            cols = st.columns(2)
            with cols[0]:
                st.pyplot(insights['resolved_wordcloud'])
            with cols[1]:
                st.pyplot(insights['action_wordcloud'])
        else:
            st.error("Please upload proper sentiment and outcome analysis results.")

    # Call Analysis Section
    st.sidebar.header("Call Analysis Tools")

    with st.sidebar:
        call_id = st.text_input("Enter Call ID for detailed analysis:")

        if call_id and st.session_state.documents:
            if call_id in st.session_state.documents:
                if st.button("Analyze Call"):
                    details = deep_call_analyze(call_id)
                    if details:
                        st.subheader("Temporal Sentiment Flow")
                        st.pyplot(details['Sentiment'])

                        st.subheader("Temporal Outcome Flow")
                        st.pyplot(details['Outcome'])
            else:
                st.error("Call ID not found in documents")

    # Agent Verification Section
    with st.sidebar:
        id_senti_outco = st.text_input("Enter 'Call ID|Sentiment|Outcome' to verify with agent transcript:")

        if id_senti_outco and st.session_state.documents:
            verify_id, senti, outco = id_senti_outco.split('|')
            if verify_id in st.session_state.documents:
                if st.button("Verify Analysis"):
                    explanation = check_agent_analysis(verify_id, senti, outco)
                    if explanation:
                        st.write(f"The {verify_id} was analyzed as {senti} and {outco}.")
                        st.write(explanation)
            else:
                st.error("Call ID not found in documents")

    # Reset Button
    if st.sidebar.button("Reset Analysis"):
        st.session_state.documents = {}
        st.session_state.agent_docs = {}
        st.session_state.analysis_complete = False
        if hasattr(st.session_state, 'sentiment_df'):
            del st.session_state.sentiment_df
        if hasattr(st.session_state, 'outcome_df'):
            del st.session_state.outcome_df
        st.rerun()


if __name__ == "__main__":
    main()