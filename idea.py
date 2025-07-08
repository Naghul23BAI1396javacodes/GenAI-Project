import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
import re

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Suppress warnings (optional)
warnings.filterwarnings('ignore')

class LegalSentimentAnalyzer:
    def __init__(self):
        # Initialize VADER sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Load legal-specific sentiment model (if available)
        try:
            self.legal_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-small-uncased")
            self.legal_model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-small-uncased")
            self.legal_sentiment = pipeline("text-classification", 
                                          model=self.legal_model,
                                          tokenizer=self.legal_tokenizer)
            self.legal_model_loaded = True
        except:
            self.legal_model_loaded = False
        
        # Initialize general sentiment analyzer as fallback
        self.general_sentiment = pipeline("sentiment-analysis")
        
    def analyze_sentiment(self, text):
        """Analyze sentiment using multiple approaches"""
        results = {}
        
        # Basic VADER analysis
        vader_result = self.sia.polarity_scores(text)
        results['vader'] = {
            'compound': vader_result['compound'],
            'positive': vader_result['pos'],
            'negative': vader_result['neg'],
            'neutral': vader_result['neu']
        }
        
        # Legal-specific analysis if model is available
        if self.legal_model_loaded:
            try:
                legal_result = self.legal_sentiment(text[:512])[0]  # Truncate to model max length
                results['legal'] = {
                    'label': legal_result['label'],
                    'score': legal_result['score']
                }
            except:
                pass
        
        # General sentiment analysis as fallback
        general_result = self.general_sentiment(text[:512])[0]
        results['general'] = {
            'label': general_result['label'],
            'score': general_result['score']
        }
        
        return results
    
    def generate_response(self, text):
        """Generate a human-readable response based on sentiment analysis"""
        analysis = self.analyze_sentiment(text)
        
        # Extract key metrics
        compound_score = analysis['vader']['compound']
        general_label = analysis['general']['label']
        general_score = analysis['general']['score']
        
        # Determine overall sentiment
        if compound_score >= 0.05:
            overall_sentiment = "positive"
        elif compound_score <= -0.05:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        # Generate response
        response = f"Sentiment Analysis Results:\n"
        response += f"- Overall sentiment: {overall_sentiment} (VADER compound score: {compound_score:.2f})\n"
        response += f"- General sentiment: {general_label} (confidence: {general_score:.2f})\n"
        
        if 'legal' in analysis:
            legal_label = analysis['legal']['label']
            legal_score = analysis['legal']['score']
            response += f"- Legal-specific sentiment: {legal_label} (confidence: {legal_score:.2f})\n"
        
        # Add interpretation
        response += "\nInterpretation:\n"
        if overall_sentiment == "positive":
            response += "The text appears favorable or supportive in tone. "
            response += "This might indicate pro-plaintiff language, favorable terms, or positive outcomes."
        elif overall_sentiment == "negative":
            response += "The text appears unfavorable or critical in tone. "
            response += "This might indicate pro-defendant language, unfavorable terms, or negative outcomes."
        else:
            response += "The text appears neutral or balanced in tone. "
            response += "Legal documents often maintain neutrality, but examine specific clauses carefully."
        
        return response

def chatbot():
    analyzer = LegalSentimentAnalyzer()
    print("Legal Sentiment Analyzer Chatbot")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("Please enter legal text to analyze (or 'quit'): ")
        
        if user_input.lower() == 'quit':
            break
            
        if not user_input.strip():
            print("Please enter some text to analyze.")
            continue
            
        print("\nAnalyzing text...\n")
        response = analyzer.generate_response(user_input)
        print(response)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    chatbot()
