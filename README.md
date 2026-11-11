# StudyBuddy AI

An AI-powered study tutor for **UN SDG 4: Quality Education**. It gives students a chatbot tutor, an adaptive quiz and personalized resource recommendations.

## Features
- **AI chatbot tutor**: answers questions using TF-IDF + cosine similarity (NLP) over a knowledge base.
- **Adaptive quiz**: difficulty rises or falls with each answer.
- **Personalized recommendations**: resources ranked by relevance, topic and your level.
- **Progress dashboard**: mastery per topic and a suggested focus area.

## Tech stack
Python 3, Flask, scikit-learn (TF-IDF, cosine similarity), HTML/CSS/JavaScript. Runs fully offline, no API key needed.

## Run locally
```bash
pip install flask scikit-learn
python app.py
# open http://localhost:5000
```

## Project structure
```
app.py              Flask backend and AI logic
data/knowledge.json Knowledge base, quiz questions, resources
templates/index.html Web interface
screenshots/        Proof-of-work screenshots
```

## Future scope
Add an LLM (OpenAI / Hugging Face) for open-ended answers, multilingual support for regional languages, a mobile app, user accounts with saved progress, and teacher dashboards.
