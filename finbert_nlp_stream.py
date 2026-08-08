import re
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from edgar import set_identity, Company

# Set SEC EDGAR User-Agent Identity (Required by SEC policy)
set_identity("QuantResearch user@quantfund.com")

class FinBERTStreamAnalyzer:
    """
    Real-Time Financial Sentiment Engine using FinBERT.
    Handles long-document windowing for SEC filings and real-time news scoring.
    """
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[+] Loading FinBERT Model ({model_name}) on device: {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # FinBERT Class Mapping: 0: Positive, 1: Negative, 2: Neutral (or mapped dynamically)
        self.labels = ["positive", "negative", "neutral"]

    def _chunk_text(self, text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
        """
        Splits large documents (like SEC filings) into overlapping chunks
        to stay safely within FinBERT's 512 token constraint.
        """
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + max_tokens])
            chunks.append(chunk)
            i += max_tokens - overlap
        return chunks if chunks else [text]

    @torch.no_grad()
    def score_text_batch(self, texts: List[str]) -> np.ndarray:
        """
        Runs batched GPU/CPU inference across text inputs.
        Returns array of shape (N, 3) containing softmax probabilities: [Pos, Neg, Neu].
        """
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        return probs

    def analyze_document(self, text: str, title: str = "Document") -> Dict[str, Any]:
        """
        Scores long-form SEC documents by chunking, running inference, 
        and aggregating document-level sentiment metrics.
        """
        # Clean text
        clean_text = re.sub(r'\s+', ' ', text).strip()
        chunks = self._chunk_text(clean_text)
        
        # Batch inference over chunks
        probs = self.score_text_batch(chunks)
        
        # Average probability distribution across all chunks
        avg_probs = probs.mean(axis=0)
        
        # Map indices (ProsusAI/finbert outputs: 0 -> positive, 1 -> negative, 2 -> neutral)
        pos_prob, neg_prob, neu_prob = avg_probs[0], avg_probs[1], avg_probs[2]
        
        # Normalized Composite Sentiment Index Score in range [-1.0, +1.0]
        sentiment_score = float(pos_prob - neg_prob)
        
        return {
            "title": title,
            "num_chunks": len(chunks),
            "sentiment_score": round(sentiment_score, 4),
            "prob_positive": round(float(pos_prob), 4),
            "prob_negative": round(float(neg_prob), 4),
            "prob_neutral": round(float(neu_prob), 4),
        }

    def fetch_and_score_sec_filing(self, ticker: str, form_type: str = "8-K") -> Dict[str, Any]:
        """
        Ingests the latest SEC filing (8-K/10-K) for a given ticker and runs sentiment scoring.
        """
        print(f"[+] Ingesting latest SEC {form_type} filing for {ticker}...")
        company = Company(ticker)
        filings = company.get_filings(form=form_type)
        
        if not filings:
            return {"error": f"No {form_type} filing found for {ticker}"}
            
        latest_filing = filings[0]
        text_content = latest_filing.text() # Clean plain text parsing via edgartools
        
        result = self.analyze_document(
            text=text_content, 
            title=f"SEC {form_type} - {ticker} ({latest_filing.filing_date})"
        )
        result["ticker"] = ticker
        result["form_type"] = form_type
        return result

# Execution Stream Test
if __name__ == "__main__":
    analyzer = FinBERTStreamAnalyzer()
    
    # Example 1: Real-Time News Headlines Stream
    news_headlines = [
        "NVIDIA reports record Q2 revenue of $30B, beating analyst estimates by 15%.",
        "Company warns of supply chain bottlenecks and margin compression in coming quarters.",
        "Board of Directors approves $50B share buyback program and increases quarterly dividend."
    ]
    
    print("\n" + "="*60)
    print(" REAL-TIME NEWS SENTIMENT SCORE STREAM")
    print("="*60)
    
    scores = analyzer.score_text_batch(news_headlines)
    for headline, prob in zip(news_headlines, scores):
        # Index 0: Positive, Index 1: Negative, Index 2: Neutral
        score = prob[0] - prob[1]
        print(f"Headline: {headline[:60]}...")
        print(f" -> Score: {score:+.4f} | Pos: {prob[0]:.2f} | Neg: {prob[1]:.2f} | Neu: {prob[2]:.2f}\n")
        
    # Example 2: SEC 8-K Filing Ingestion & Scoring
    sec_analysis = analyzer.fetch_and_score_sec_filing(ticker="NVDA", form_type="8-K")
    print("="*60)
    print(" SEC FILING AGGREGATED SCORE")
    print("="*60)
    for k, v in sec_analysis.items():
        print(f"{k:<18}: {v}")
