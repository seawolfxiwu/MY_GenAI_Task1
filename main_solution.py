import streamlit as st
import pandas as pd
import ollama
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import os
from typing import Dict, Tuple, List, Optional

# Constants
GENERATION_MODEL = "deepseek-r1:7b"  # LLM for reasoning (~20s per text)
DATA_DIR = "data/output"
os.makedirs(DATA_DIR, exist_ok=True)

# System instructions and prompts
SYS_INSTRUCTIONS = [
    "You are a compliance specialist with deep expertise in Health Insurance regulations. Your task is to meticulously analyze call transcripts to identify potential regulatory risks, non-compliant statements, and adherence to disclosure and privacy guidelines.",
    "You are a customer experience analyst skilled in understanding emotions and sentiment in Health Insurance conversations. Your role is to evaluate transcripts to detect frustration, satisfaction, confusion, and other emotional cues, and recommend improvements to enhance service quality.",
    "You are a domain expert in Health Insurance claims and policy interpretation. Analyze call transcripts to assess accuracy of information provided, identify misunderstandings about policy terms, and flag potentially incorrect guidance given to customers.",
    "You are a Health Insurance sales strategist. Your job is to review call transcripts and extract insights on upselling opportunities, customer objections, retention risks, and sales effectiveness in agent-customer interactions.",
    "You are a quality assurance coach specializing in Health Insurance customer service. Your task is to evaluate call transcripts to assess agent performance, communication clarity, adherence to scripts, and areas for coaching or training improvement."
]

SENTIMENT_PROMPTS = [
    "Please classify the sentiment of the following TEXT and briefly explain why with a few words. The sentiment should be Positive, Negative, or Neutral. The OUTPUT format should look like:\nSentiment: ...\nWhy: ... \n",
    "Analyze the sentiment of the following TEXT and classify it as Positive, Negative, or Neutral. Support your classification with a short explanation based on the language or tone used. Use the following format for your OUTPUT:\nSentiment: ...\nWhy: ...\n",
    "Determine whether the sentiment of the following TEXT is Positive, Negative, or Neutral. Briefly explain your reasoning in a few words. Return your answer in this format:\nSentiment: ...\nWhy: ...\n",
    "Read the TEXT below and decide if the sentiment is Positive, Negative, or Neutral. Include a short explanation of your reasoning. Format your response like this:\nSentiment: ...\nWhy: ...\n",
    "Evaluate the sentiment expressed in the following TEXT. Assign one of the three labels: Positive, Negative, or Neutral. Then, provide a brief rationale for your decision. Format your OUTPUT as:\nSentiment: ...\nWhy: ...\n"
]

OUTCOME_PROMPTS = [
    "Please analyse the following TEXT, detect the Issue and determine the outcome. The outcome should be 'Issue Resolved' or 'Follow-up Action Needed'. ",
     "Carefully review the provided TEXT to identify the key issue. Based on your assessment, determine whether the issue has been resolved or if further action is required. The outcome should be 'Issue Resolved' or 'Follow-up Action Needed'. ",
     "Analyze the following TEXT. Identify the issue and assess whether it has been resolved or needs further attention. The outcome should be 'Issue Resolved' or 'Follow-up Action Needed'. ",
     "Let’s look into the TEXT together. Try to detect what the main issue is, and then decide if it’s fully resolved or if more steps are needed. The outcome should be 'Issue Resolved' or 'Follow-up Action Needed'. ",
     "Read the TEXT and identify any described problem or issue. Then, evaluate whether the issue has been addressed satisfactorily. The outcome should be 'Issue Resolved' or 'Follow-up Action Needed'. "
]

# Set up Streamlit UI
st.set_page_config(page_title="Health Transcript Analysis", layout="wide")
st.title("Health Transcript Analysis")
st.sidebar.header("Configuration")


def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    if 'agent_docs' not in st.session_state:
        st.session_state.agent_docs = {}
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False


initialize_session_state()


# Utility Functions
def clean_text(text: str, max_length: int = 128) -> str:
    """Clean and truncate text for display"""
    text = text.strip()
    if len(text) > max_length:
        return text[:max_length] + '...'
    return text


def validate_prompts(sys_instructions: List[str], prompts: List[str]) -> bool:
    """Validate that system instructions and prompts are properly mapped"""
    return len(sys_instructions) == len(prompts)


# Data Loading
def load_utterances(files: List) -> Tuple[Dict, Dict]:
    """Load and separate member and agent utterances from uploaded files"""
    documents = {}  # Member-Only Utterances
    agent_docs = {}  # Agent-Only Utterances

    for file in files:
        try:
            file_name = os.path.splitext(file.name)[0]
            file_ext = os.path.splitext(file.name)[-1].lower()

            if file_ext == ".txt":
                text = file.read().decode("utf-8")
                if text:
                    lines = text.split("\n")

                    # Process member lines
                    member_lines = [
                        line.split("Member:", 1)[1].strip()
                        for line in lines
                        if line.startswith("Member:")
                    ]
                    documents[file_name] = "\n".join(member_lines)

                    # Process agent lines
                    agent_lines = [
                        line.split("PA Agent:", 1)[1].strip()
                        for line in lines
                        if line.startswith("PA Agent:")
                    ]
                    agent_docs[file_name] = "\n".join(agent_lines)
        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")

    return documents, agent_docs


# Analysis Functions
def call_llm(model: str, system_prompt: str, user_prompt: str) -> str:
    """Make a call to the LLM with proper error handling"""
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response["message"]["content"]
    except Exception as e:
        st.error(f"Error calling LLM: {str(e)}")
        return ""


def parse_sentiment_output(output: str) -> Dict:
    """Parse sentiment analysis output from LLM"""
    sentiment = "Neutral"
    why = "Not explained!"

    if 'Sentiment:' in output:
        senti_sent = output[output.find('Sentiment:') + len('Sentiment:'):]
        for senti in ["Positive", "Negative", "Neutral"]:
            if senti in senti_sent:
                sentiment = senti
                break

    if 'Why:' in output:
        why = output[output.find('Why:') + len('Why:'):].strip()
        why = clean_text(why)

    return {'Sentiment': sentiment, 'Why': why}


def parse_outcome_output(output: str) -> Dict:
    """Parse outcome analysis output from LLM"""
    issue, outcome = "No specific issue.", "Not applicable."

    if 'Issue:' in output:
        output = output[output.find('Issue:') + len('Issue:'):]
        if 'Outcome:' in output:
            position = output.find('Outcome:')
            issue = output[:position].strip()

            for outc in ["Issue Resolved", "Follow-up Action Needed"]:
                if outc in output:
                    outcome = outc
                    break

    return {'Issue': clean_text(issue), 'Outcome': outcome}


def get_sentiment(text: str, k: int = 3) -> Dict:
    """Analyze sentiment of text using majority voting from k LLM calls"""
    if not validate_prompts(SYS_INSTRUCTIONS, SENTIMENT_PROMPTS):
        st.error("System instructions and sentiment prompts are not properly mapped!")
        return None

    responses = []
    for i in range(min(k, len(SYS_INSTRUCTIONS))):
        prompt = f"{SENTIMENT_PROMPTS[i]}TEXT:\n{text}\nOUTPUT:"
        response = call_llm(GENERATION_MODEL, SYS_INSTRUCTIONS[i], prompt)
        responses.append(response)

    results = [parse_sentiment_output(response) for response in responses if response]

    if not results:
        return {'Sentiment': 'Neutral', 'Confidence': 0, 'Why': 'Analysis failed'}

    # Majority voting
    sentis = [result['Sentiment'] for result in results]
    sentiment_counter = Counter(sentis)
    sentiment = sentiment_counter.most_common(1)[0][0]
    confidence = round(sentiment_counter[sentiment] / len(results), 3)

    # Get first explanation that matches the chosen sentiment
    why = next((result['Why'] for result in results if result['Sentiment'] == sentiment), "No explanation")

    return {'Sentiment': sentiment, 'Confidence': confidence, 'Why': why}


def get_outcome(text: str, k: int = 3) -> Dict:
    """Analyze outcome of text using majority voting from k LLM calls"""
    if not validate_prompts(SYS_INSTRUCTIONS, OUTCOME_PROMPTS):
        st.error("System instructions and outcome prompts are not properly mapped!")
        return None

    responses = []
    for i in range(min(k, len(SYS_INSTRUCTIONS))):
        prompt = f"{OUTCOME_PROMPTS[i]}Your OUTPUT format should look like:\nIssue: ...\nOutcome: ... \nTEXT:\n{text}\nOUTPUT:"
        response = call_llm(GENERATION_MODEL, SYS_INSTRUCTIONS[i], prompt)
        responses.append(response)

    results = [parse_outcome_output(response) for response in responses if response]

    if not results:
        return {'Outcome': 'Not applicable', 'Confidence': 0, 'Issue': 'Analysis failed'}

    # Majority voting
    outcos = [result['Outcome'] for result in results]
    outcome_counter = Counter(outcos)
    outcome = outcome_counter.most_common(1)[0][0]
    confidence = round(outcome_counter[outcome] / len(results), 3)

    # Get first issue that matches the chosen outcome
    issue = next((result['Issue'] for result in results if result['Outcome'] == outcome), "No issue identified")

    return {'Outcome': outcome, 'Confidence': confidence, 'Issue': issue}


# Evaluation Metrics
def calculate_metrics(y_true: List, y_pred: List, label_map: Dict) -> Dict:
    """Calculate classification metrics and confusion matrix"""
    # Map labels
    y_true_mapped = [label_map.get(ele, ele) for ele in y_true]
    y_pred_mapped = [label_map.get(ele, ele) for ele in y_pred]

    labels = list(label_map.values())

    # Confusion matrix
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f'Actual {l}' for l in labels],
        columns=[f'Predicted {l}' for l in labels]
    )

    # Classification report
    report = classification_report(y_true_mapped, y_pred_mapped, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return {"Classification Report": report_df, "Confusion Matrix": cm_df}


def sentiment_eval_metrics(std_df: pd.DataFrame, prd_df: pd.DataFrame) -> Dict:
    """Calculate sentiment evaluation metrics"""
    y_true = std_df['Sentiment']
    y_pred = prd_df['Sentiment']

    label_map = {"Negative": "NEG", "Neutral": "NEU", "Positive": "POS"}
    return calculate_metrics(y_true, y_pred, label_map)


def outcome_eval_metrics(std_df: pd.DataFrame, prd_df: pd.DataFrame) -> Dict:
    """Calculate outcome evaluation metrics"""
    y_true = std_df['Outcome']
    y_pred = prd_df['Outcome']

    label_map = {"Issue Resolved": "IRD", "Not applicable.": "N/A", "Follow-up Action Needed": "FAN"}
    return calculate_metrics(y_true, y_pred, label_map)


# Visualization Functions
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


def deep_call_analyze(call_id: str) -> Optional[Dict]:
    """Perform turn-by-turn analysis of a specific call"""
    if call_id not in st.session_state.documents:
        return None

    rounds = st.session_state.documents[call_id].split('\n')
    turn_numbers = range(1, len(rounds) + 1)

    # Sentiment analysis
    round_sentiments = []
    sentiment_values = []
    sentiment_map = {"Negative": -1, "Neutral": 0, "Positive": 1}

    for round in rounds:
        sentiment = get_sentiment(round, k=1)['Sentiment']
        round_sentiments.append(sentiment)
        sentiment_values.append(sentiment_map[sentiment])

    sentiment_fig = generate_temporal_plot(
        turn_numbers, sentiment_values, rounds,
        {-1: "Negative", 0: "Neutral", 1: "Positive"},
        "Sentiment Flow During Call",
        "Dialogue Turn Number",
        "Sentiment"
    )

    # Outcome analysis
    round_outcomes = []
    outcome_values = []
    outcome_map = {"Follow-up Action Needed": -1, "Not applicable.": 0, "Issue Resolved": 1}

    for round in rounds:
        outcome = get_outcome(round, k=1)['Outcome']
        round_outcomes.append(outcome)
        outcome_values.append(outcome_map[outcome])

    outcome_fig = generate_temporal_plot(
        turn_numbers, outcome_values, rounds,
        {-1: "Follow-up Action", 0: "No Issue", 1: "Issue Resolved"},
        "Outcome Flow During Call",
        "Dialogue Turn Number",
        "Outcome"
    )

    return {'Sentiment': sentiment_fig, 'Outcome': outcome_fig}


def check_agent_analysis(call_id: str, senti: str, outco: str) -> Optional[str]:
    """Verify analysis against agent transcript"""
    if call_id not in st.session_state.agent_docs:
        return None

    user_text = st.session_state.documents[call_id]
    agent_text = st.session_state.agent_docs[call_id]

    prompt = (
        f"The following Customer-side Call Transcript was analyzed for Sentiment and Outcome. "
        f"The predicted Sentiment is {senti} and the predicted Outcome is {outco}. "
        f"Do you agree? Please briefly prove your opinion by checking the Agent-side Transcript at the end.\n"
        f"Customer-side Call Transcript:\n{user_text}\n\n"
        f"Agent-side Transcript:\n{agent_text}\n"
        f"OUTPUT:"
    )

    response = call_llm(
        GENERATION_MODEL,
        "You are an expert good at analyzing Health Insurance textual data.",
        prompt
    )

    if not response:
        return "Analysis failed"

    # Clean up the response if it contains XML tags
    if '</think>' in response:
        output = response[response.find('</think>') + len('</think>'):].strip()
        if 'Answer:' in output:
            output = output[:output.find('Answer:')].strip()
        return output
    return response


# Main Application
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "Sentiment Analysis",
        "Outcome Analysis",
        "Evaluation",
        "Insights"
    ])

    # Sentiment Analysis Tab
    with tab1:
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
    with tab2:
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
    with tab3:
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
    with tab4:
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
