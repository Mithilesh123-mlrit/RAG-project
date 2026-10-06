import os
import re

from flask import Flask, render_template_string, request
import urllib.request
import urllib.error
import json

app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in Render.")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{MODEL}:generateContent"
)

# ============================================================
# KNOWLEDGE BASE
# ============================================================

KNOWLEDGE_BASE = """
Welcome to InnovateCorp! This Knowledge Transfer (KT) guide is designed to
help new employees navigate their initial weeks and understand key aspects
of our operations. Our core values are Innovation, Collaboration, and
Customer Focus.

Team Structure:
You will be joining the 'Project Alpha' team, reporting to Sarah Chen,
the Senior Project Manager.

Your direct teammates include:
David Lee (Lead Developer),
Maria Rodriguez (UI/UX Designer),
and Tom Jackson (QA Engineer).

Team meetings are held every Monday at 10 AM in Conference Room 3,
and daily stand-ups are at 9:30 AM via Google Meet.

Key Tools & Software:
For project management, we use Jira for task tracking and Confluence
for documentation.

Our primary communication tool is Slack for instant messaging and
Google Workspace for email and calendars.

Development work is primarily done using Python and JavaScript,
with code hosted on GitHub.

Access to these tools will be granted within your first three days.

Onboarding Process:
Your first week will focus on setup and introductions.

You'll receive your laptop and login credentials on day one.

HR will conduct an orientation session on Tuesday covering company
policies, benefits, and payroll.

You'll have one-on-one meetings with your team members throughout the week.

By the end of your second week, you should have access to all necessary
systems and have completed mandatory compliance training modules.

Important Resources:
The company's internal knowledge base can be found at
internal.innovatecorp.com/kb.

This includes FAQs, best practices, and troubleshooting guides.

For IT support, submit a ticket via support.innovatecorp.com
or call extension 5555.

Health and wellness benefits information is available on the HR portal.

Culture & Expectations:
InnovateCorp encourages a proactive and collaborative environment.

We value open communication and continuous learning.

Don't hesitate to ask questions; your team is here to support your growth.

Performance reviews are conducted quarterly, and professional development
courses are available through the 'InnovateLearn' platform.
""".strip()


# ============================================================
# SIMPLE LOCAL RAG RETRIEVAL
# ============================================================

def make_chunks(text):
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    return paragraphs


CHUNKS = make_chunks(KNOWLEDGE_BASE)


def tokenize(text):
    return set(
        re.findall(
            r"[a-zA-Z0-9]+",
            text.lower()
        )
    )


CHUNK_TOKENS = [
    tokenize(chunk)
    for chunk in CHUNKS
]


def retrieve(question, top_k=3):
    question_tokens = tokenize(question)

    scored = []

    for chunk, chunk_tokens in zip(CHUNKS, CHUNK_TOKENS):
        overlap = question_tokens.intersection(chunk_tokens)

        score = len(overlap)

        # Give extra weight to important exact phrases.
        question_lower = question.lower()
        chunk_lower = chunk.lower()

        if "project alpha" in question_lower and "project alpha" in chunk_lower:
            score += 20

        if "report" in question_lower and "reporting to" in chunk_lower:
            score += 10

        if "who" in question_lower and "sarah chen" in chunk_lower:
            score += 5

        scored.append((score, chunk))

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        chunk
        for score, chunk in scored[:top_k]
        if score > 0
    ]


# ============================================================
# GEMINI
# ============================================================

def ask_gemini(question, context):
    prompt = f"""
You are the InnovateCorp Knowledge Transfer assistant.

Answer the user's question using ONLY the provided knowledge-base context.

Do not invent information.

If the answer is present in the context, answer directly and clearly.

If the information is not present, say:

"The information is not available in the InnovateCorp KT guide."

Knowledge-base context:

{context}

User question:

{question}

Answer:
""".strip()

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=60
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as error:
        details = error.read().decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            f"Gemini API error {error.code}: {details[:1000]}"
        )

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Unable to connect to Gemini: {error.reason}"
        )

    candidates = result.get("candidates", [])

    if not candidates:
        raise RuntimeError(
            "Gemini returned no answer."
        )

    parts = candidates[0].get(
        "content",
        {}
    ).get(
        "parts",
        []
    )

    answer = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict)
    ).strip()

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty answer."
        )

    return answer


# ============================================================
# HTML
# ============================================================

PAGE = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1">

<title>RAG Assistant</title>

<style>

body {
    font-family: Arial, sans-serif;
    max-width: 1000px;
    margin: 40px auto;
    padding: 0 20px;
}

h1 {
    font-size: 40px;
}

textarea {
    width: 100%;
    min-height: 120px;
    padding: 12px;
    font-size: 18px;
    box-sizing: border-box;
}

button {
    margin-top: 15px;
    padding: 12px 25px;
    font-size: 17px;
    cursor: pointer;
}

.result {
    margin-top: 25px;
    padding: 20px;
    border-radius: 10px;
    background: #f4f4f4;
}

.error {
    margin-top: 25px;
    padding: 20px;
    background: #ffe5e5;
    color: #b00000;
}

.context {
    margin-top: 25px;
    padding: 20px;
    background: #f8f8f8;
}

pre {
    white-space: pre-wrap;
    font-family: Arial, sans-serif;
    line-height: 1.5;
}

</style>

</head>

<body>

<h1>RAG Assistant</h1>

<form method="POST">

<textarea
    name="question"
    placeholder="Ask a question about the InnovateCorp KT guide..."
>{{ question }}</textarea>

<br>

<button type="submit">
Ask
</button>

</form>

{% if error %}

<div class="error">

<strong>Error:</strong>

<pre>{{ error }}</pre>

</div>

{% endif %}


{% if answer %}

<div class="result">

<h2>Answer</h2>

<pre>{{ answer }}</pre>

</div>


<div class="context">

<h2>Retrieved Context</h2>

{% for chunk in chunks %}

<pre>{{ chunk }}</pre>

<hr>

{% endfor %}

</div>

{% endif %}

</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    question = ""
    answer = None
    chunks = []
    error = None

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        if not question:

            error = "Please enter a question."

        else:

            try:

                chunks = retrieve(question)

                if not chunks:

                    answer = (
                        "The information is not available "
                        "in the InnovateCorp KT guide."
                    )

                else:

                    context = "\n\n".join(
                        chunks
                    )

                    answer = ask_gemini(
                        question,
                        context
                    )

            except Exception as exc:

                error = str(exc)

    return render_template_string(
        PAGE,
        question=question,
        answer=answer,
        chunks=chunks,
        error=error
    )


@app.route("/health")
def health():

    return {
        "status": "ok",
        "model": MODEL,
        "chunks": len(CHUNKS)
    }


if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "10000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
