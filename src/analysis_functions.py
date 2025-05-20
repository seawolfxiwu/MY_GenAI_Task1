import streamlit as st
import ollama
from collections import Counter
from typing import Dict, Optional
import os
from .utils import clean_text, validate_prompts
from .visualization import generate_temporal_plot

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
