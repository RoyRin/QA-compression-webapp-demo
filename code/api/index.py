import os
import json
import requests
from flask import Flask, send_file, jsonify, request

app = Flask(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Allowed model IDs (whitelist to prevent abuse)
ALLOWED_MODELS = {
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.6",
}

DEFAULT_SMALL = "anthropic/claude-haiku-4.5"
DEFAULT_LARGE = "anthropic/claude-opus-4.6"
DEFAULT_API_KEY = os.environ.get("OPENROUTER_DEFAULT_KEY", "")


class APIError(Exception):
    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code


@app.errorhandler(APIError)
def handle_api_error(e):
    return jsonify({"error": e.message}), e.status_code


@app.errorhandler(Exception)
def handle_generic_error(e):
    return jsonify({"error": f"Internal error: {type(e).__name__}: {e}"}), 500


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
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=120)
    except requests.exceptions.Timeout:
        raise APIError("Request timed out. Try again.", 504)
    except requests.exceptions.ConnectionError:
        raise APIError("Could not connect to OpenRouter.", 502)

    if r.status_code == 401:
        raise APIError("Invalid API key. Check your OpenRouter key.", 401)
    if r.status_code == 402:
        raise APIError("Insufficient credits on your OpenRouter account.", 402)
    if r.status_code == 429:
        raise APIError("Rate limited. Wait a moment and try again.", 429)
    if not r.ok:
        raise APIError(f"OpenRouter error ({r.status_code}): {r.text[:200]}", r.status_code)

    data = r.json()
    if "choices" not in data or not data["choices"]:
        raise APIError(f"Unexpected response from OpenRouter: {json.dumps(data)[:200]}")
    return data["choices"][0]["message"]["content"]


def get_api_key(data):
    """Get API key from request, falling back to the default key."""
    key = (data.get("api_key") or "").strip()
    if key:
        return key
    default = os.environ.get("OPENROUTER_DEFAULT_KEY", "").strip()
    if default:
        return default
    raise APIError("Please enter your OpenRouter API key.", 400)


def get_model(data, key, default):
    """Get a model ID from request data, falling back to default if missing/invalid."""
    model = data.get(key, default)
    if model not in ALLOWED_MODELS:
        raise APIError(f"Model '{model}' is not allowed.", 400)
    return model


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


@app.route("/api/has_default_key", methods=["GET"])
def has_default_key():
    """Check if a default API key is configured (without exposing it)."""
    return jsonify({"available": bool(os.environ.get("OPENROUTER_DEFAULT_KEY", ""))})


@app.route("/api/step1_opus_answer", methods=["POST"])
def step1_opus_answer():
    """Step 1: Large model answers the question (reference answer)."""
    data = request.json
    api_key = get_api_key(data)
    question = data["question"]
    large_model = get_model(data, "large_model", DEFAULT_LARGE)

    messages = [{"role": "user", "content": question}]
    answer = call_model(api_key, large_model, messages)
    return jsonify({"answer": answer})


@app.route("/api/step2_haiku_initial", methods=["POST"])
def step2_haiku_initial():
    """Step 2: Small model answers the question (initial attempt)."""
    data = request.json
    api_key = get_api_key(data)
    question = data["question"]
    small_model = get_model(data, "small_model", DEFAULT_SMALL)

    messages = [{"role": "user", "content": question}]
    answer = call_model(api_key, small_model, messages)
    return jsonify({"answer": answer})


@app.route("/api/judge_answer", methods=["POST"])
def judge_answer():
    """Judge whether an answer is correct by comparing to the reference."""
    data = request.json
    api_key = get_api_key(data)
    question = data["question"]
    answer = data["answer"]
    reference = data["reference"]

    prompt = f"""Compare these two answers to the question: "{question}"

Reference answer (correct):
{reference}

Answer to evaluate:
{answer}

Is the answer to evaluate essentially correct? Consider whether it reaches the same conclusion/result, even if the wording or approach differs.

Respond with ONLY "correct" or "incorrect" on the first line, then a brief one-sentence explanation."""

    large_model = get_model(data, "large_model", DEFAULT_LARGE)
    messages = [{"role": "user", "content": prompt}]
    resp = call_model(api_key, large_model, messages, max_tokens=200)

    is_correct = "correct" in resp.strip().split("\n")[0].lower() and "incorrect" not in resp.strip().split("\n")[0].lower()
    explanation = resp.strip().split("\n")[-1] if "\n" in resp.strip() else resp.strip()

    return jsonify({"correct": is_correct, "explanation": explanation, "raw": resp})


@app.route("/api/step3_generate_questions", methods=["POST"])
def step3_generate_questions():
    """Step 3: Haiku generates 10 binary questions about its answer."""
    data = request.json
    api_key = get_api_key(data)
    question = data["question"]
    haiku_answer = data["haiku_answer"]
    questioner_model = get_model(data, "questioner_model", DEFAULT_SMALL)

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
    resp = call_model(api_key, questioner_model, messages, max_tokens=2000)

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
    api_key = get_api_key(data)
    question = data["question"]
    opus_answer = data["opus_answer"]
    haiku_answer = data["haiku_answer"]
    questions = data["questions"]
    large_model = get_model(data, "large_model", DEFAULT_LARGE)

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
    resp = call_model(api_key, large_model, messages, max_tokens=500)

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
    api_key = get_api_key(data)
    question = data["question"]
    haiku_answer = data["haiku_answer"]
    questions = data["questions"]
    answers = data["answers"]
    small_model = get_model(data, "small_model", DEFAULT_SMALL)

    qa_transcript = format_qa_transcript(questions, answers)

    prompt = f"""Original question: {question}

Current answer: {haiku_answer}

Additional information to incorporate:
{qa_transcript}

Provide an improved answer that incorporates all the above information, without any preamble or acknowledgment. Answer the original question directly."""

    messages = [{"role": "user", "content": prompt}]
    answer = call_model(api_key, small_model, messages)
    return jsonify({"answer": answer})
