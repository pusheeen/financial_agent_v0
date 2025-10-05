#!/usr/bin/env python3
"""
Reddit Data Scraper for Financial Agent
Scrapes posts and comments from specified subreddits mentioning our target tickers.
"""

import praw
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import List, Dict, Any
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
REDDIT_DATA_DIR = BASE_DIR / "data" / "unstructured" / "reddit"
REDDIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Target tickers from your subset
TARGET_TICKERS = {
    'NVDA', 'MU', 'AVGO', 'TSM', 'VRT', 'SMCI', 'INOD', 'RR', 'IREN', 
    'CIFR', 'RIOT', 'OKLO', 'SMR', 'CCJ', 'VST', 'NXE', 'EOSE', 'QS'
}

# Subreddits to scrape
SUBREDDITS = [
    'stocks', 'investing', 'wallstreetbets', 'SecurityAnalysis', 'ValueInvesting',
    'UraniumSqueeze', 'uraniumstocks', 'renewableenergy', 'cryptomining', 
    'BitcoinMining', 'NVDA'
]

# Time range: past 1 month
DAYS_BACK = 30
SINCE_DATE = datetime.now() - timedelta(days=DAYS_BACK)

class RedditScraper:
    def __init__(self):
        """Initialize Reddit API client and sentiment analyzer."""
        # Reddit API credentials (you'll need to set these)
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'FinancialAgent/1.0')
        )
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def contains_ticker(self, text: str) -> List[str]:
        """Check if text contains any of our target tickers."""
        text_upper = text.upper()
        found_tickers = []
        
        for ticker in TARGET_TICKERS:
            # Look for ticker as whole word (not part of another word)
            pattern = r'\b' + re.escape(ticker) + r'\b'
            if re.search(pattern, text_upper):
                found_tickers.append(ticker)
        
        return found_tickers
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text using VADER."""
        scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'bullish'
        elif scores['compound'] <= -0.05:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'compound_score': scores['compound'],
            'positive_score': scores['pos'],
            'negative_score': scores['neg'],
            'neutral_score': scores['neu']
        }
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text using simple keyword matching."""
        text_lower = text.lower()
        topics = []
        
        # Define topic keywords
        topic_keywords = {
            'earnings': ['earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'revenue', 'profit'],
            'technical_analysis': ['chart', 'ta', 'technical', 'support', 'resistance', 'breakout'],
            'fundamentals': ['pe ratio', 'p/e', 'valuation', 'book value', 'debt', 'cash flow'],
            'news': ['news', 'announcement', 'press release', 'ceo', 'management'],
            'partnerships': ['partnership', 'deal', 'acquisition', 'merger', 'collaboration'],
            'regulatory': ['sec', 'fda', 'approval', 'regulation', 'compliance'],
            'market_sentiment': ['bullish', 'bearish', 'optimistic', 'pessimistic', 'hype'],
            'competition': ['competitor', 'competition', 'market share', 'vs', 'versus'],
            'future_outlook': ['outlook', 'forecast', 'prediction', 'guidance', 'target']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def scrape_subreddit(self, subreddit_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Scrape posts from a specific subreddit."""
        print(f"Scraping r/{subreddit_name}...")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        posts_data = []
        
        try:
            # Get hot posts from the past month
            for post in subreddit.hot(limit=limit):
                # Check if post is from the past month
                post_date = datetime.fromtimestamp(post.created_utc)
                if post_date < SINCE_DATE:
                    continue
                
                # Combine title and selftext
                full_text = f"{post.title}\n{post.selftext or ''}"
                
                # Check if post mentions any of our tickers
                mentioned_tickers = self.contains_ticker(full_text)
                if not mentioned_tickers:
                    continue
                
                # Analyze sentiment
                sentiment_data = self.analyze_sentiment(full_text)
                
                # Extract topics
                topics = self.extract_topics(full_text)
                
                # Get top comments (up to 5)
                comments_data = []
                post.comments.replace_more(limit=0)  # Don't load more comments
                for comment in post.comments[:5]:
                    if hasattr(comment, 'body') and comment.body != '[deleted]':
                        comment_tickers = self.contains_ticker(comment.body)
                        if comment_tickers:
                            comment_sentiment = self.analyze_sentiment(comment.body)
                            comments_data.append({
                                'id': comment.id,
                                'body': comment.body[:500],  # Limit length
                                'score': comment.score,
                                'created_utc': comment.created_utc,
                                'mentioned_tickers': comment_tickers,
                                'sentiment': comment_sentiment['sentiment'],
                                'compound_score': comment_sentiment['compound_score']
                            })
                
                # Store post data
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'selftext': post.selftext or '',
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'subreddit': subreddit_name,
                    'url': f"https://reddit.com{post.permalink}",
                    'mentioned_tickers': mentioned_tickers,
                    'sentiment': sentiment_data['sentiment'],
                    'compound_score': sentiment_data['compound_score'],
                    'positive_score': sentiment_data['positive_score'],
                    'negative_score': sentiment_data['negative_score'],
                    'topics': topics,
                    'comments': comments_data
                }
                
                posts_data.append(post_data)
                print(f"  Found post about {mentioned_tickers}: {post.title[:50]}...")
                
                # Rate limiting
                time.sleep(1)
        
        except Exception as e:
            print(f"Error scraping r/{subreddit_name}: {e}")
        
        return posts_data
    
    def scrape_all_subreddits(self) -> List[Dict[str, Any]]:
        """Scrape all specified subreddits."""
        all_posts = []
        
        for subreddit in SUBREDDITS:
            try:
                posts = self.scrape_subreddit(subreddit)
                all_posts.extend(posts)
                print(f"  Scraped {len(posts)} posts from r/{subreddit}")
            except Exception as e:
                print(f"Failed to scrape r/{subreddit}: {e}")
                continue
        
        return all_posts
    
    def save_data(self, posts_data: List[Dict[str, Any]]):
        """Save scraped data to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all posts
        all_posts_file = REDDIT_DATA_DIR / f"reddit_posts_{timestamp}.json"
        with open(all_posts_file, 'w') as f:
            json.dump(posts_data, f, indent=2)
        
        # Save summary by ticker
        ticker_summary = {}
        for post in posts_data:
            for ticker in post['mentioned_tickers']:
                if ticker not in ticker_summary:
                    ticker_summary[ticker] = {
                        'total_posts': 0,
                        'bullish_posts': 0,
                        'bearish_posts': 0,
                        'neutral_posts': 0,
                        'avg_sentiment': 0,
                        'subreddits': set()
                    }
                
                ticker_summary[ticker]['total_posts'] += 1
                ticker_summary[ticker]['subreddits'].add(post['subreddit'])
                
                if post['sentiment'] == 'bullish':
                    ticker_summary[ticker]['bullish_posts'] += 1
                elif post['sentiment'] == 'bearish':
                    ticker_summary[ticker]['bearish_posts'] += 1
                else:
                    ticker_summary[ticker]['neutral_posts'] += 1
        
        # Convert sets to lists for JSON serialization
        for ticker_data in ticker_summary.values():
            ticker_data['subreddits'] = list(ticker_data['subreddits'])
        
        summary_file = REDDIT_DATA_DIR / f"reddit_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(ticker_summary, f, indent=2)
        
        print(f"\nSaved {len(posts_data)} posts to {all_posts_file}")
        print(f"Saved summary to {summary_file}")
        
        return all_posts_file, summary_file

def main():
    """Main function to run the Reddit scraper."""
    print("Starting Reddit scraper...")
    print(f"Target tickers: {', '.join(sorted(TARGET_TICKERS))}")
    print(f"Subreddits: {', '.join(SUBREDDITS)}")
    print(f"Time range: Past {DAYS_BACK} days")
    
    # Check for Reddit API credentials
    if not os.getenv('REDDIT_CLIENT_ID'):
        print("Error: REDDIT_CLIENT_ID environment variable not set")
        print("Please set up Reddit API credentials:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Create a new app (script type)")
        print("3. Set environment variables:")
        print("   export REDDIT_CLIENT_ID='your_client_id'")
        print("   export REDDIT_CLIENT_SECRET='your_client_secret'")
        print("   export REDDIT_USER_AGENT='FinancialAgent/1.0'")
        return
    
    scraper = RedditScraper()
    posts_data = scraper.scrape_all_subreddits()
    
    if posts_data:
        all_posts_file, summary_file = scraper.save_data(posts_data)
        print(f"\nScraping complete! Found {len(posts_data)} posts mentioning target tickers.")
    else:
        print("No posts found mentioning target tickers.")

if __name__ == "__main__":
    main()


