#!/usr/bin/env python3
"""
Reddit Data Ingestion for Neo4j Graph Database
Processes scraped Reddit data and adds it to the financial graph.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv
import re

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
REDDIT_DATA_DIR = BASE_DIR / "data" / "unstructured" / "reddit"

# Neo4j connection
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

def ingest_reddit_posts(driver, posts_data):
    """Ingest Reddit posts into Neo4j graph."""
    print("Ingesting Reddit posts into Neo4j...")
    
    with driver.session() as session:
        # Create Reddit post nodes
        for post in posts_data:
            # Create or update Reddit post
            create_post_query = """
            MERGE (p:RedditPost {id: $post_id})
            SET p.title = $title,
                p.selftext = $selftext,
                p.score = $score,
                p.upvote_ratio = $upvote_ratio,
                p.num_comments = $num_comments,
                p.created_utc = $created_utc,
                p.subreddit = $subreddit,
                p.url = $url,
                p.sentiment = $sentiment,
                p.compound_score = $compound_score,
                p.positive_score = $positive_score,
                p.negative_score = $negative_score,
                p.topics = $topics
            """
            
            session.run(create_post_query, {
                'post_id': post['id'],
                'title': post['title'],
                'selftext': post['selftext'],
                'score': post['score'],
                'upvote_ratio': post['upvote_ratio'],
                'num_comments': post['num_comments'],
                'created_utc': post['created_utc'],
                'subreddit': post['subreddit'],
                'url': post['url'],
                'sentiment': post['sentiment'],
                'compound_score': post['compound_score'],
                'positive_score': post['positive_score'],
                'negative_score': post['negative_score'],
                'topics': post['topics']
            })
            
            # Link post to mentioned companies
            for ticker in post['mentioned_tickers']:
                link_company_query = """
                MATCH (c:Company {ticker: $ticker})
                MATCH (p:RedditPost {id: $post_id})
                MERGE (p)-[:MENTIONS]->(c)
                """
                session.run(link_company_query, {
                    'ticker': ticker,
                    'post_id': post['id']
                })
            
            # Create sentiment nodes and link to posts
            sentiment_query = """
            MATCH (p:RedditPost {id: $post_id})
            MERGE (s:Sentiment {type: $sentiment})
            MERGE (p)-[:HAS_SENTIMENT]->(s)
            """
            session.run(sentiment_query, {
                'post_id': post['id'],
                'sentiment': post['sentiment']
            })
            
            # Create topic nodes and link to posts
            for topic in post['topics']:
                topic_query = """
                MATCH (p:RedditPost {id: $post_id})
                MERGE (t:Topic {name: $topic})
                MERGE (p)-[:DISCUSSES_TOPIC]->(t)
                """
                session.run(topic_query, {
                    'post_id': post['id'],
                    'topic': topic
                })
            
            # Process comments
            for comment in post.get('comments', []):
                create_comment_query = """
                MERGE (c:RedditComment {id: $comment_id})
                SET c.body = $body,
                    c.score = $score,
                    c.created_utc = $created_utc,
                    c.sentiment = $sentiment,
                    c.compound_score = $compound_score
                """
                
                session.run(create_comment_query, {
                    'comment_id': comment['id'],
                    'body': comment['body'],
                    'score': comment['score'],
                    'created_utc': comment['created_utc'],
                    'sentiment': comment['sentiment'],
                    'compound_score': comment['compound_score']
                })
                
                # Link comment to post
                link_comment_query = """
                MATCH (p:RedditPost {id: $post_id})
                MATCH (c:RedditComment {id: $comment_id})
                MERGE (c)-[:REPLIES_TO]->(p)
                """
                session.run(link_comment_query, {
                    'post_id': post['id'],
                    'comment_id': comment['id']
                })
                
                # Link comment to mentioned companies
                for ticker in comment['mentioned_tickers']:
                    link_comment_company_query = """
                    MATCH (co:Company {ticker: $ticker})
                    MATCH (c:RedditComment {id: $comment_id})
                    MERGE (c)-[:MENTIONS]->(co)
                    """
                    session.run(link_comment_company_query, {
                        'ticker': ticker,
                        'comment_id': comment['id']
                    })
    
    print("Reddit posts ingestion complete!")

def create_reddit_indexes(driver):
    """Create indexes for Reddit data."""
    print("Creating Reddit indexes...")
    
    with driver.session() as session:
        # Create indexes for better query performance
        indexes = [
            "CREATE INDEX reddit_post_id IF NOT EXISTS FOR (p:RedditPost) ON (p.id)",
            "CREATE INDEX reddit_post_subreddit IF NOT EXISTS FOR (p:RedditPost) ON (p.subreddit)",
            "CREATE INDEX reddit_post_sentiment IF NOT EXISTS FOR (p:RedditPost) ON (p.sentiment)",
            "CREATE INDEX reddit_post_created IF NOT EXISTS FOR (p:RedditPost) ON (p.created_utc)",
            "CREATE INDEX reddit_comment_id IF NOT EXISTS FOR (c:RedditComment) ON (c.id)",
            "CREATE INDEX sentiment_type IF NOT EXISTS FOR (s:Sentiment) ON (s.type)",
            "CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)"
        ]
        
        for index_query in indexes:
            try:
                session.run(index_query)
            except Exception as e:
                print(f"Index creation warning: {e}")

def get_latest_reddit_file():
    """Get the most recent Reddit data file."""
    if not REDDIT_DATA_DIR.exists():
        return None
    
    reddit_files = list(REDDIT_DATA_DIR.glob("reddit_posts_*.json"))
    if not reddit_files:
        return None
    
    # Sort by modification time and get the latest
    latest_file = max(reddit_files, key=lambda f: f.stat().st_mtime)
    return latest_file

def main():
    """Main function to ingest Reddit data."""
    print("Starting Reddit data ingestion...")
    
    # Get latest Reddit data file
    reddit_file = get_latest_reddit_file()
    if not reddit_file:
        print("No Reddit data files found. Please run scrape_reddit.py first.")
        return
    
    print(f"Using Reddit data file: {reddit_file}")
    
    # Load Reddit data
    with open(reddit_file, 'r') as f:
        posts_data = json.load(f)
    
    print(f"Loaded {len(posts_data)} Reddit posts")
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    try:
        # Create indexes
        create_reddit_indexes(driver)
        
        # Ingest posts
        ingest_reddit_posts(driver, posts_data)
        
        print("Reddit data ingestion completed successfully!")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()


