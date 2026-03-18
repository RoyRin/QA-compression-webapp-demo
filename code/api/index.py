import os
import json
import requests
from flask import Flask, send_file, jsonify, request

app = Flask(__name__)

# OpenRouter model IDs
SMALL_MODEL = "anthropic/claude-haiku-4-5-20251001"
LARGE_MODEL = "anthropic/claude-sonnet-4-5-20250929"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_model(api_key, model, messages, temperature=0.0, max_tokens=4000):
    """Call a model via OpenRouter."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def format_qa_transcript(questions, answers):
    """Format Q&A pairs for inclusion in prompts."""
    lines = []
    for q, a in zip(questions, answers):
        lines.append(f"Question: {q} ; Answer: {'Yes' if a else 'No'}")
    return "\n".join(lines)


# --- Routes ---

@app.route("/")
def home():
    return send_file(os.path.join(os.path.dirname(__file__), "index.html"))


@app.route("/qa-diagram.png")
def diagram():
    return send_file(os.path.join(os.path.dirname(__file__), "qa-diagram.png"), mimetype="image/png")


@app.route("/api/step1_opus_answer", methods=["POST"])
def step1_opus_answer():
    """Step 1: Opus answers the question (reference answer)."""
    data = request.json
    api_key = data["api_key"]
    question = data["question"]

    messages = [{"role": "user", "content": question}]
    answer = call_model(api_key, LARGE_MODEL, messages)
    return jsonify({"answer": answer})


@app.route("/api/step2_haiku_initial", methods=["POST"])
def step2_haiku_initial():
    """Step 2: Haiku answers the question (initial attempt)."""
    data = request.json
    api_key = data["api_key"]
    question = data["question"]

    messages = [{"role": "user", "content": question}]
    answer = call_model(api_key, SMALL_MODEL, messages)
    return jsonify({"answer": answer})


@app.route("/api/step3_generate_questions", methods=["POST"])
def step3_generate_questions():
    """Step 3: Haiku generates 10 binary questions about its answer."""
    data = request.json
    api_key = data["api_key"]
    question = data["question"]
    haiku_answer = data["haiku_answer"]

    prompt = f"""You are a small language model trying to answer a prompt. The original prompt is: {question}
Your current answer is: {haiku_answer}

Generate exactly 10 yes/no questions that would help you answer this prompt better.
Each question should be:
- A yes/no question
- Specific and focused
- Something you can use to improve your answer
- Each question should explore a different aspect

Format your response as a numbered list:
1. <question>
2. <question>
...
10. <question>

Return ONLY the numbered list, nothing else."""

    messages = [{"role": "user", "content": prompt}]
    resp = call_model(api_key, SMALL_MODEL, messages, max_tokens=2000)

    # Parse numbered list
    import re
    questions = []
    for line in resp.strip().split("\n"):
        line = line.strip()
        match = re.match(r'^\d+[\.)\s]+(.+)', line)
        if match:
            q = match.group(1).strip()
            if q:
                questions.append(q)

    # Ensure exactly 10
    questions = questions[:10]

    return jsonify({"questions": questions, "raw": resp})


@app.route("/api/step4_answer_questions", methods=["POST"])
def step4_answer_questions():
    """Step 4: Opus answers the 10 binary questions."""
    data = request.json
    api_key = data["api_key"]
    question = data["question"]
    opus_answer = data["opus_answer"]
    haiku_answer = data["haiku_answer"]
    questions = data["questions"]

    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    prompt = f"""You are a large language model helping guide a small model. The original prompt is: {question}

Your answer is: {opus_answer}

The small model's answer is: {haiku_answer}

Answer each yes/no question about the small model's answer:
{questions_text}

For each question, respond with ONLY "Yes" or "No" on a separate line.
Format your response as:
1. Yes/No
2. Yes/No
...
{len(questions)}. Yes/No

Return ONLY the numbered list of Yes/No answers, nothing else."""

    messages = [{"role": "user", "content": prompt}]
    resp = call_model(api_key, LARGE_MODEL, messages, max_tokens=500)

    # Parse answers
    import re
    answers = []
    for line in resp.strip().split("\n"):
        line = line.strip()
        match = re.match(r'^\d+[\.)\s]+\s*(yes|no)', line, re.IGNORECASE)
        if match:
            answers.append(match.group(1).lower() == "yes")

    # Pad if needed
    while len(answers) < len(questions):
        answers.append(False)
    answers = answers[:len(questions)]

    return jsonify({"answers": answers, "raw": resp})


@app.route("/api/step5_haiku_revised", methods=["POST"])
def step5_haiku_revised():
    """Step 5: Haiku revises its answer using the Q&A transcript."""
    data = request.json
    api_key = data["api_key"]
    question = data["question"]
    haiku_answer = data["haiku_answer"]
    questions = data["questions"]
    answers = data["answers"]

    qa_transcript = format_qa_transcript(questions, answers)

    prompt = f"""Original question: {question}

Current answer: {haiku_answer}

Additional information to incorporate:
{qa_transcript}

Provide an improved answer that incorporates all the above information, without any preamble or acknowledgment. Answer the original question directly."""

    messages = [{"role": "user", "content": prompt}]
    answer = call_model(api_key, SMALL_MODEL, messages)
    return jsonify({"answer": answer})
