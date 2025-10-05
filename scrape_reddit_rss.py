#!/usr/bin/env python3
"""
Reddit RSS Data Scraper for Financial Agent
Scrapes Reddit posts via RSS feeds - no authentication required.
"""

import feedparser
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
REDDIT_DATA_DIR = BASE_DIR / "data" / "unstructured" / "reddit"
REDDIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Target tickers
TARGET_TICKERS = {
    'NVDA', 'MU', 'AVGO', 'TSM', 'VRT', 'SMCI', 'INOD', 'RR', 'IREN', 
    'CIFR', 'RIOT', 'OKLO', 'SMR', 'CCJ', 'VST', 'NXE', 'EOSE', 'QS'
}

# Subreddits with RSS feeds
SUBREDDIT_RSS = {
    'stocks': 'https://www.reddit.com/r/stocks.rss',
    'investing': 'https://www.reddit.com/r/investing.rss',
    'wallstreetbets': 'https://www.reddit.com/r/wallstreetbets.rss',
    'SecurityAnalysis': 'https://www.reddit.com/r/SecurityAnalysis.rss',
    'ValueInvesting': 'https://www.reddit.com/r/ValueInvesting.rss',
    'UraniumSqueeze': 'https://www.reddit.com/r/UraniumSqueeze.rss',
    'uraniumstocks': 'https://www.reddit.com/r/uraniumstocks.rss',
    'renewableenergy': 'https://www.reddit.com/r/renewableenergy.rss',
    'cryptomining': 'https://www.reddit.com/r/cryptomining.rss',
    'BitcoinMining': 'https://www.reddit.com/r/BitcoinMining.rss',
    'NVDA': 'https://www.reddit.com/r/NVDA.rss'
}

class RedditRSSScraper:
    def __init__(self):
        """Initialize RSS scraper and sentiment analyzer."""
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
        
        topic_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'guidance', 'beat', 'miss'],
            'technical_analysis': ['chart', 'resistance', 'support', 'breakout', 'rsi', 'macd'],
            'fundamentals': ['pe', 'p/e', 'valuation', 'book value', 'debt', 'cash'],
            'news': ['news', 'announcement', 'press release', 'update'],
            'AI': ['ai', 'artificial intelligence', 'machine learning', 'gpu', 'data center'],
            'crypto': ['crypto', 'bitcoin', 'ethereum', 'mining', 'blockchain'],
            'uranium': ['uranium', 'nuclear', 'reactor', 'fuel'],
            'renewable': ['renewable', 'solar', 'wind', 'clean energy', 'green']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def scrape_subreddit_rss(self, subreddit_name: str, rss_url: str) -> List[Dict[str, Any]]:
        """Scrape posts from a subreddit RSS feed."""
        print(f"Scraping r/{subreddit_name} via RSS...")
        
        posts_data = []
        
        try:
            # Parse RSS feed
            feed = feedparser.parse(rss_url)
            
            if feed.bozo:
                print(f"  Warning: RSS feed may be malformed for r/{subreddit_name}")
            
            for entry in feed.entries:
                # Combine title and description
                title = entry.get('title', '')
                description = entry.get('description', '')
                full_text = f"{title}\n{description}"
                
                # Check if post mentions any of our tickers
                mentioned_tickers = self.contains_ticker(full_text)
                if not mentioned_tickers:
                    continue
                
                # Analyze sentiment
                sentiment_data = self.analyze_sentiment(full_text)
                
                # Extract topics
                topics = self.extract_topics(full_text)
                
                # Extract post ID from link
                post_id = entry.get('id', '').split('/')[-1] if entry.get('id') else f"rss_{subreddit_name}_{len(posts_data)}"
                
                # Parse published date
                published = entry.get('published_parsed')
                if published:
                    created_utc = datetime(*published[:6]).timestamp()
                else:
                    created_utc = datetime.now().timestamp()
                
                post_data = {
                    'id': post_id,
                    'title': title,
                    'selftext': description,
                    'score': 0,  # RSS doesn't provide upvotes
                    'upvote_ratio': 0.5,  # Default neutral ratio
                    'num_comments': 0,  # RSS doesn't provide comment count
                    'created_utc': created_utc,
                    'subreddit': subreddit_name,
                    'url': entry.get('link', ''),
                    'sentiment': sentiment_data['sentiment'],
                    'compound_score': sentiment_data['compound_score'],
                    'positive_score': sentiment_data['positive_score'],
                    'negative_score': sentiment_data['negative_score'],
                    'topics': topics,
                    'mentioned_tickers': mentioned_tickers
                }
                
                posts_data.append(post_data)
            
            print(f"  Scraped {len(posts_data)} posts from r/{subreddit_name}")
            
        except Exception as e:
            print(f"  Error scraping r/{subreddit_name}: {e}")
        
        return posts_data
    
    def scrape_all_subreddits(self) -> List[Dict[str, Any]]:
        """Scrape all subreddits via RSS feeds."""
        print("Starting Reddit RSS scraper...")
        print(f"Target tickers: {', '.join(sorted(TARGET_TICKERS))}")
        print(f"Subreddits: {', '.join(SUBREDDIT_RSS.keys())}")
        print("Data source: RSS feeds (no authentication required)")
        print()
        
        all_posts = []
        
        for subreddit_name, rss_url in SUBREDDIT_RSS.items():
            posts = self.scrape_subreddit_rss(subreddit_name, rss_url)
            all_posts.extend(posts)
        
        return all_posts
    
    def save_data(self, posts_data: List[Dict[str, Any]]):
        """Save scraped data to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all posts
        all_posts_file = REDDIT_DATA_DIR / f"reddit_posts_rss_{timestamp}.json"
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
        
        summary_file = REDDIT_DATA_DIR / f"reddit_summary_rss_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(ticker_summary, f, indent=2)
        
        print(f"\nSaved {len(posts_data)} posts to {all_posts_file}")
        print(f"Saved summary to {summary_file}")
        
        return all_posts_file, summary_file

def main():
    """Main function to run the Reddit RSS scraper."""
    scraper = RedditRSSScraper()
    
    # Scrape all subreddits
    posts_data = scraper.scrape_all_subreddits()
    
    if not posts_data:
        print("No posts found mentioning target tickers.")
        return
    
    # Save data
    scraper.save_data(posts_data)
    
    print(f"\nReddit RSS scraping complete!")
    print(f"Total posts found: {len(posts_data)}")

if __name__ == "__main__":
    main()
