# 🧠 Sentiment Analysis API (Multilingual - English, Hindi, Marathi)

This project is a **FastAPI-powered Sentiment Analysis API** capable of analyzing comments written in **English, Hindi, and Marathi**. It uses a **transformer-based multilingual model** (`nlptown/bert-base-multilingual-uncased-sentiment`) to classify each comment as either **Positive** or **Negative**, and optionally generates a **real-time sentiment distribution chart**.

---

## 🚀 Features

- ✅ Analyze sentiment of up to 1000 multilingual comments in one request.
- ✅ Supports **English**, **Hindi**, and **Marathi** text (via Unicode range matching).
- ✅ Classifies sentiments into **Positive** and **Negative** based on star ratings.
- ✅ Optional filtering by sentiment type.
- ✅ Visual representation using a **pie chart** (PNG image) of sentiment distribution.
- ✅ CORS-enabled for integration with frontend apps.
- ✅ Logging for monitoring and debugging.
- ✅ Health check endpoint.

---

## 🧰 Tech Stack

- **Backend:** Python 3.9+, [FastAPI](https://fastapi.tiangolo.com/)
- **Model:** `nlptown/bert-base-multilingual-uncased-sentiment` from Hugging Face Transformers
- **Visualization:** Matplotlib
- **Middleware:** CORS (enabled for all origins)

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/sentiment-analysis-api.git
cd sentiment-analysis-api
pip install -r requirements.txt
