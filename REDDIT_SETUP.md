# Reddit Integration Setup

This guide explains how to set up Reddit data scraping and analysis for the Financial Agent.

## Prerequisites

1. **Reddit API Credentials**: You need a Reddit app to access the API
2. **Neo4j Database**: Must be running and accessible
3. **Python Dependencies**: Already installed via requirements.txt

## Reddit API Setup

### Step 1: Create Reddit App
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Note down your:
   - Client ID (under the app name)
   - Client Secret (the secret field)

### Step 2: Set Environment Variables
Add these to your `.env` file or export them:

```bash
# Reddit API Credentials
REDDIT_CLIENT_ID="your_client_id_here"
REDDIT_CLIENT_SECRET="your_client_secret_here"
REDDIT_USER_AGENT="FinancialAgent/1.0"
```

## Usage

### 1. Scrape Reddit Data
```bash
cd /Users/maggiechen/Documents/financial_agent/Financial_ADK_Agent_Graph_Database
python scrape_reddit.py
```

This will:
- Scrape posts from 9 subreddits over the past month
- Filter for posts mentioning your 18 target tickers
- Analyze sentiment using VADER
- Extract topics and engagement metrics
- Save data to `data/unstructured/reddit/`

### 2. Ingest Reddit Data into Neo4j
```bash
python ingest_reddit.py
```

This will:
- Load the latest Reddit data file
- Create RedditPost, RedditComment, Sentiment, and Topic nodes
- Link posts to companies and create relationships
- Create indexes for better query performance

### 3. Query Reddit Data via Agent
Once ingested, you can ask questions like:
- "What's the Reddit sentiment on NVDA this week?"
- "Show me the most discussed topics about uranium stocks"
- "What are people saying about MU's earnings on r/stocks?"
- "Find the most upvoted posts about AVGO in the past month"

## Data Structure

### Neo4j Schema
```
RedditPost -> MENTIONS -> Company
RedditComment -> REPLIES_TO -> RedditPost
RedditPost -> HAS_SENTIMENT -> Sentiment
RedditPost -> DISCUSSES_TOPIC -> Topic
```

### Reddit Data Fields
- **Posts**: title, selftext, score, upvote_ratio, sentiment, topics, subreddit
- **Comments**: body, score, sentiment, mentioned_tickers
- **Sentiment**: bullish/bearish/neutral with compound scores
- **Topics**: earnings, technical_analysis, fundamentals, news, etc.

## Subreddits Monitored
- r/stocks, r/investing, r/wallstreetbets, r/SecurityAnalysis, r/ValueInvesting
- r/UraniumSqueeze, r/uraniumstocks, r/renewableenergy
- r/cryptomining, r/BitcoinMining, r/NVDA

## Target Tickers
NVDA, MU, AVGO, TSM, VRT, SMCI, INOD, RR, IREN, CIFR, RIOT, OKLO, SMR, CCJ, VST, NXE, EOSE, QS

## Troubleshooting

### Common Issues
1. **"No Reddit data files found"**: Run `scrape_reddit.py` first
2. **"Reddit API credentials not set"**: Check your environment variables
3. **Rate limiting**: The scraper includes delays, but Reddit may still rate limit
4. **Empty results**: Check if posts actually mention your tickers

### Rate Limits
- Reddit API: 60 requests per minute
- The scraper includes 1-second delays between requests
- For large datasets, consider running during off-peak hours

## Next Steps
- Set up automated daily scraping
- Add more sophisticated sentiment analysis
- Include additional social media platforms
- Create sentiment trend visualizations



