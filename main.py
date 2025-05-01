from fastapi import FastAPI, HTTPException, Response, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import io
import base64
from transformers import pipeline
import logging
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing sentiment in English, Hindi, and Marathi comments",
    version="1.0.1",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sentiment analysis pipeline
try:
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
    logger.info(f"Loaded sentiment analysis model: {model_name}")
except Exception as e:
    logger.error(f"Failed to load sentiment model: {str(e)}")
    raise

# Define the input models
class Comment(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        pattern=r"^[\u0900-\u097F\uA8E0-\uA8FFa-zA-Z0-9\s'’.,!?\-]+$",  # Supports Hindi, Marathi, English
    )
    id: Optional[str] = None


class CommentsRequest(BaseModel):
    comments: List[Comment] = Field(..., min_items=1, max_items=1000)


# Normalize text (remove special characters)
def normalize_text(text: str) -> str:
    return text.replace("’", "'")


# Convert model's output (1-5 ratings) into "POSITIVE" / "NEGATIVE"
def map_sentiment_label(rating: str) -> str:
    score = int(rating.split()[0])  # Extract numerical score from label (e.g., "5 stars")
    if score in [4, 5]:
        return "POSITIVE"
    elif score in [1, 2]:
        return "NEGATIVE"
#else:
       # return "NEUTRAL"  # Keep neutral but count it in `total_count`


# Store the last analyzed sentiment data for chart generation
latest_sentiment_data = {
    "positive_count": 0,
    "negative_count": 0,
    "total_count": 0
}


# Analyze sentiment and return results
def analyze_sentiment(comments: List[Comment], sentiment_filter: Optional[str] = None) -> Dict[str, Any]:
    try:
        global latest_sentiment_data  # Ensure access to update the latest results

        results = {
            "positive_count": 0,
            "negative_count": 0,
            "total_count": len(comments),  # Always count all comments
            "positive_comments": [],
            "negative_comments": [],
        }

        comment_texts = [normalize_text(comment.text) for comment in comments]
        logger.info(f"Analyzing {len(comment_texts)} comments")

        raw_results = sentiment_analyzer(comment_texts)

        for i, analysis in enumerate(raw_results):
            sentiment = map_sentiment_label(analysis["label"])

            if sentiment_filter and sentiment_filter.upper() != sentiment:
                continue  # Skip non-matching sentiment

            if sentiment == "POSITIVE":
                results["positive_count"] += 1
                results["positive_comments"].append(comment_texts[i])
            elif sentiment == "NEGATIVE":
                results["negative_count"] += 1
                results["negative_comments"].append(comment_texts[i])

        # Update latest sentiment data for the chart
        latest_sentiment_data = {
            "positive_count": results["positive_count"],
            "negative_count": results["negative_count"],
            "total_count": results["total_count"]
        }

        return results

    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Generate sentiment chart
def generate_precise_chart(positive_count: int, negative_count: int) -> bytes:
    total = positive_count + negative_count

    if total == 0:
        logger.warning("No sentiment data available for chart generation.")
        return b""

    labels = ['Positive', 'Negative']
    sizes = [
        (positive_count / total) * 100,
        (negative_count / total) * 100
    ]
    colors = ['#4CAF50', '#F44336']

    logger.info(f"Generating chart: Positive {sizes[0]}%, Negative {sizes[1]}%")

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Precise Sentiment Distribution (Positive vs Negative)')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf.getvalue()


# POST route to analyze comments and provide response
@app.post("/analyze-comments/")
async def analyze_comments(
    comments_request: CommentsRequest,
    filter: Optional[str] = Query(None, enum=["positive", "negative"], description="Filter comments by sentiment"),
):
    try:
        results = analyze_sentiment(comments_request.comments, filter)
        positive_percentage = round(
            (results["positive_count"] / results["total_count"]) * 100, 1
        ) if results["total_count"] > 0 else 0.0
        negative_percentage = round(
            (results["negative_count"] / results["total_count"]) * 100, 1
        ) if results["total_count"] > 0 else 0.0

        response_data = {
            "positive_count": results["positive_count"],
            "negative_count": results["negative_count"],
            "total_count": results["total_count"],
            "positive_percentage": positive_percentage,
            "negative_percentage": negative_percentage,
            "positive_comments": results["positive_comments"],
            "negative_comments": results["negative_comments"],
        }

        logger.info(f"Sentiment Analysis Result: {response_data}")  # Log for debugging

        return response_data

    except Exception as e:
        logger.error(f"Error in API endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# GET route to fetch the sentiment chart
@app.get("/get-chart/")
async def get_chart():
    try:
        chart_data = generate_precise_chart(
            positive_count=latest_sentiment_data["positive_count"],
            negative_count=latest_sentiment_data["negative_count"]
        )
        return Response(content=chart_data, media_type="image/png", headers={"Cache-Control": "no-store"})
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "loaded"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
