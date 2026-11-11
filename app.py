"""StudyBuddy AI - an AI study tutor for SDG 4 (Quality Education).

NLP: TF-IDF + cosine similarity retrieval answers student questions from a
knowledge base. Personalisation: a per-topic mastery score drives adaptive
quiz difficulty and resource recommendations. Runs fully offline.
"""
import json
import os
import random
from collections import defaultdict

from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "data", "knowledge.json"), encoding="utf-8") as f:
    DB = json.load(f)

app = Flask(__name__)

# ---------- NLP index (chatbot tutor) ----------
KB = DB["knowledge"]
kb_docs = [f'{k["title"]}. {k["text"]}' for k in KB]
kb_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
kb_matrix = kb_vec.fit_transform(kb_docs)

# ---------- Resource index (recommender) ----------
RES = DB["resources"]
res_docs = [f'{r["title"]} {r["tags"]} {r["topic"]}' for r in RES]
res_vec = TfidfVectorizer(stop_words="english")
res_matrix = res_vec.fit_transform(res_docs)

# ---------- Student progress (in memory, single demo user) ----------
progress = defaultdict(lambda: {"correct": 0, "total": 0, "level": 1})
asked_ids = set()

GREETINGS = {"hi", "hello", "hey", "hii", "namaste"}


def mastery(topic):
    p = progress[topic]
    return round(100 * p["correct"] / p["total"]) if p["total"] else None


def recommend(query=None, topic=None, level=None, k=3):
    if query:
        sims = cosine_similarity(res_vec.transform([query]), res_matrix).ravel()
    else:
        sims = [0.0] * len(RES)
    scored = []
    for i, r in enumerate(RES):
        s = float(sims[i])
        if topic and r["topic"] == topic:
            s += 0.5
        if level:
            s += 0.3 - 0.1 * abs(r["level"] - level)
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for s, r in scored[:k] if s > 0.05]


@app.route("/")
def index():
    return render_template("index.html", topics=DB["topics"])


@app.route("/api/chat", methods=["POST"])
def chat():
    msg = (request.json or {}).get("message", "").strip()
    if not msg:
        return jsonify({"answer": "Please type a question and I will help!", "resources": []})
    if msg.lower().strip("!. ") in GREETINGS:
        return jsonify({
            "answer": "Hi! I'm StudyBuddy, your AI tutor. Ask me about maths, science, programming or English - for example: 'What is photosynthesis?'",
            "resources": [],
        })
    sims = cosine_similarity(kb_vec.transform([msg]), kb_matrix).ravel()
    best = int(sims.argmax())
    if sims[best] < 0.08:
        return jsonify({
            "answer": "I'm not sure about that yet. Try rephrasing, or ask about topics like quadratic equations, Newton's laws, loops, functions or active voice.",
            "resources": recommend(query=msg),
        })
    hit = KB[best]
    runner = [KB[i]["title"] for i in sims.argsort()[::-1][1:3] if sims[i] > 0.08]
    answer = f'**{hit["title"]}** ({hit["topic"]})\n{hit["text"]}'
    if runner:
        answer += "\n\nRelated topics: " + ", ".join(runner)
    return jsonify({
        "answer": answer,
        "resources": recommend(query=msg, topic=hit["topic"], level=progress[hit["topic"]]["level"]),
    })


@app.route("/api/quiz")
def quiz():
    topic = request.args.get("topic") or random.choice(DB["topics"])
    level = progress[topic]["level"]
    pool = [q for q in DB["quiz"] if q["topic"] == topic and q["level"] == level]
    if not pool:
        pool = [q for q in DB["quiz"] if q["topic"] == topic]
    q = random.choice(pool)
    return jsonify({"id": q["id"], "topic": q["topic"], "level": q["level"], "q": q["q"], "options": q["options"]})


@app.route("/api/answer", methods=["POST"])
def answer():
    data = request.json or {}
    q = next((x for x in DB["quiz"] if x["id"] == data.get("id")), None)
    if q is None:
        return jsonify({"error": "unknown question"}), 400
    correct = data.get("choice") == q["answer"]
    p = progress[q["topic"]]
    p["total"] += 1
    p["correct"] += int(correct)
    if correct:
        p["level"] = min(3, p["level"] + 1)
    else:
        p["level"] = max(1, p["level"] - 1)
    return jsonify({
        "correct": correct,
        "right_answer": q["options"][q["answer"]],
        "new_level": p["level"],
        "mastery": mastery(q["topic"]),
        "resources": [] if correct else recommend(topic=q["topic"], level=q["level"], query=q["q"]),
    })


@app.route("/api/dashboard")
def dashboard():
    rows = []
    for t in DB["topics"]:
        p = progress[t]
        rows.append({"topic": t, "attempted": p["total"], "correct": p["correct"], "mastery": mastery(t), "level": p["level"]})
    attempted = [r for r in rows if r["attempted"]]
    weakest = min(attempted, key=lambda r: r["mastery"])["topic"] if attempted else None
    recs = recommend(topic=weakest, level=progress[weakest]["level"]) if weakest else []
    return jsonify({"rows": rows, "weakest": weakest, "resources": recs})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
