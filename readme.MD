# Financial AI Assistant

A multi-agent conversational financial analytics platform that combines company fundamentals analysis, SEC filing intelligence, and machine learning-based stock price prediction through an intuitive chat interface.

This project applies architectural principles from knowledge graphs and AI agents, adapting them from healthcare to finance and market analysis. The core idea demonstrates how a combination of graph databases, RAG (Retrieval-Augmented Generation), and predictive models can create powerful, domain-specific AI assistants.

## Background Articles

This project builds on concepts from several technical articles of mine:

- [Building Knowledge Graphs from Scratch Using Neo4j and Vertex AI](https://medium.com/@rubenszimbres/building-knowledge-graphs-from-scratch-using-neo4j-and-vertex-ai-8311eb69a472) - March 2024
- [Use LLMs to Turn CSVs into Knowledge Graphs: A Case in Healthcare](https://medium.com/@rubenszimbres/use-llms-to-turn-csvs-into-knowledge-graphs-a-case-in-healthcare-158d3ee0afde) - June 2024
- [Understanding Alzheimer's: Building Knowledge Graphs from Unstructured Data with Gemini](https://medium.com/@rubenszimbres/understanding-alzheimers-building-knowledge-graphs-from-unstructured-data-with-gemini-ba11167da31d) - February 2025

## What This System Does

The Financial AI Assistant empowers investors, analysts, and researchers to analyze market data through natural language queries. Instead of writing complex database queries or manually sifting through lengthy SEC filings, users can simply ask questions like:

- "What was the revenue for Apple in 2023?"
- "Summarize the key risks mentioned in NVIDIA's latest 10-K filing?"

This natural language interface activates tools in the backend agent system to provide comprehensive answers.

### Core Capabilities

The system provides four main capabilities via its agent tools:

1. **Corporate Financial Analysis**: Query company fundamentals, financials, and key metrics from structured data
2. **SEC Filing Intelligence**: Perform semantic search and summarization on the full text of annual 10-K reports to extract qualitative insights
3. **Social Media Sentiment Analysis**: Collect and analyze Reddit discussions from finance-related subreddits to gauge market sentiment
4. **Predictive Analytics**: Generate next-day price predictions for individual stocks using pre-trained autoregressive models

## Technical Architecture

### System Overview

The Financial AI Assistant uses a sophisticated multi-layered architecture:

1. **Frontend Layer**: Clean web interface with HTML, CSS, and JavaScript chat interface
2. **API Layer**: FastAPI server handling request validation and response formatting
3. **Agent Orchestration**: Root/Sub-Agent architecture using Google's ADK (Agent Development Kit)
4. **Data Storage**: Multi-modal Neo4j database serving as both graph and vector database
5. **External AI Services**: Google Vertex AI for language understanding and embeddings

### Agent Architecture

The system employs a Root/Sub-Agent pattern:

- **Root Agent**: Acts as the "lead analyst," analyzing questions and delegating to appropriate specialists
- **Sub-Agents**: Function as "specialist analysts" with specific tools:
  - **Graph QA Agent**: Generates Cypher queries for statistical questions about company financials
  - **Document RAG Agent**: Performs semantic search through SEC filings
  - **Social Sentiment Agent**: Analyzes Reddit discussions for market sentiment and community insights
  - **Stock Predictor Agent**: Runs ML models for next-day price predictions

### Data Flow

1. **Query Analysis**: Root Agent analyzes user intent
2. **Tool Selection**: Appropriate sub-agent selected based on query type
3. **Data Retrieval**: Sub-agent executes its tool (Cypher query, embedding search, or ML prediction)
4. **Response Synthesis**: LLM processes results into natural language
5. **Delivery**: Final answer sent through API to web interface

## Technical Challenges & Solutions

### Data Ingestion
- Historical data pulled using `yfinance` library
- SEC 10-K reports fetched from EDGAR database in complex iXBRL format
- Reddit discussions collected via two complementary approaches:
  - **RSS Scraping**: Public feeds from finance subreddits (no authentication required)
  - **API Scraping**: Full Reddit API access for richer data including upvotes, comments, and engagement metrics

### Social Media Sentiment Processing
- **Target Communities**: 11 finance-focused subreddits (stocks, investing, wallstreetbets, UraniumSqueeze, etc.)
- **Ticker Detection**: Regex pattern matching for 18 target stock symbols
- **Sentiment Analysis**: VADER sentiment analyzer classifies posts as bullish/bearish/neutral
- **Topic Extraction**: Automated categorization (earnings, AI, crypto, uranium, renewables, etc.)
- **Data Output**: Structured JSON with sentiment scores, engagement metrics, and topic tags

### Handling "Noise"
- SEC filings contain extensive boilerplate legal jargon
- RAG pipeline uses `UnstructuredHTMLLoader` to extract core textual content
- Reddit data filtered to target tickers only, reducing noise from general market discussions

### Prediction Modeling
- Stock prediction demonstrates ML integration (not financial advice)
- Simplified autoregressive model trained on historical price/volume
- Does not account for market sentiment, news, or macroeconomic factors

### Cost Optimization
- Uses efficient `gemini-1.5-flash-001` model to balance performance and cost
- Cloud deployment scales to zero to minimize idle costs

## Project Structure

```
financial-assistant/
├── app/
│   ├── agents/
│   │   └── agents.py
│   ├── graph_db/
│   │   ├── connection.py
│   │   └── 2_populate_graph.py
│   ├── models/
│   │   ├── predict.py
│   │   └── train_predictor.py
│   ├── saved_models/
│   ├── templates/
│   │   └── index.html
│   └── main.py
├── data/
│   ├── structured/
│   │   ├── financials/
│   │   └── prices/
│   └── unstructured/
│       ├── 10k/
│       └── reddit/
├── fetch_data.py
├── populate_graph.py
├── scrape_reddit.py
├── scrape_reddit_rss.py
├── companies.csv
├── .env
├── .env-example
├── Dockerfile
├── README.md
└── requirements.txt
```

## Quick Start

### Prerequisites

- Google Cloud Platform account with Vertex AI enabled
- Neo4j database instance (Neo4j AuraDB Free Tier recommended)
- Python 3.11 or higher
- Docker (for deployment)

### Environment Setup

1. Clone the repository and create a virtual environment:

```bash
git clone https://github.com/your-username/financial-assistant.git
cd financial-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment variables by creating a `.env` file from `.env-example`:

```env
GOOGLE_PROJECT_ID="your-gcp-project-id"
GOOGLE_LOCATION="us-central1"
NEO4J_URI="neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your-password"
SEC_USER_AGENT="Your Name your.email@provider.com"

# Optional: Reddit API credentials for enhanced social sentiment analysis
REDDIT_CLIENT_ID="your-reddit-client-id"
REDDIT_CLIENT_SECRET="your-reddit-client-secret"
REDDIT_USER_AGENT="YourApp/1.0 by u/yourusername"
```

### Reddit API Setup (Optional)

For enhanced social media analysis, you can set up Reddit API credentials:

1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Choose "script" as the application type
4. Note down your `client_id` and `client_secret`
5. Set the environment variables above

**Note**: The RSS scraper (`scrape_reddit_rss.py`) works without credentials, while the API scraper (`scrape_reddit.py`) provides richer data but requires authentication.

3. Run the complete data pipeline:

```bash
# Step 1: Create the list of target companies
python3 create_companies.py

# Step 2: Download all raw data from APIs (this will take a while)
python3 fetch_data.py

# Step 3: Populate the Neo4j database (Graph + Embeddings)
python3 app/graph_db/2_populate_graph.py

# Step 4: Collect social media sentiment data
python3 scrape_reddit_rss.py    # No credentials required (RSS feeds)
python3 scrape_reddit.py        # Requires Reddit API credentials

# Step 5: Train the predictive models (one for each stock)
python3 app/models/train_predictor.py
```

4. Start the development server:

```bash
uvicorn app.main:app --reload --port 8080
```

5. Open your browser to `http://127.0.0.1:8080` and start asking questions.

## Example Queries

### Company Fundamentals (Graph QA)
- "How many companies are in the technology sector?"
- "What was the net income for MSFT in 2023?"
- "List the major events for TSLA in their 2024 filing."

### SEC Filing Insights (Document RAG)
- "What did Google say about their AI strategy in the last 10-K?"
- "Summarize Amazon's management outlook from their 2024 filing."
- "What are the common concerns about supply chain disruptions?"

### Social Media Sentiment Analysis
- "What is the Reddit sentiment for NVDA this week?"
- "Show me the most discussed uranium stocks on social media."
- "What are retail investors saying about IREN?"

### Predictive Analytics (Prediction Tool)
- "Predict the next closing price for AAPL."
- "What is the stock price prediction for NVDA tomorrow?"

## Screenshots of the Neo4j Graph Database

<img src="img/Screenshot from 2025-09-08 15-04-02.png" alt="A screenshot showing text in Neo4j" width="800">

<img src="img/Screenshot from 2025-09-08 15-04-18.png" alt="A general screenshot of Neo4j" width="800">


## Deployment

### Google Cloud Run

The system is containerized for easy deployment to Google Cloud Run.


```bash
gcloud auth login
gcloud config set project YOUR-PROJECT

gcloud artifacts repositories create financial-assistant-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for financial assistant service"

gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR-PROJECT/financial-assistant-repo/assistant-service:latest

gcloud run deploy financial-assistant-service \
    --image=us-central1-docker.pkg.dev/financial-agent-474022/financial-assistant-repo/assistant-service:latest \
    --platform=managed \
    --region=us-central1 \
    --allow-unauthenticated \
    --env-vars-file=.env \ 
    --min-instances 0 \
    --max-instances 3 \
    --cpu 4 \
    --memory 8192Mi \
    --concurrency 10
```

After deployment, update the API URL in `templates/index.html` to point to your new Cloud Run service URL.

### AWS ECS

```bash
# 1. Create ECR repository
aws ecr create-repository \
    --repository-name financial-assistant-repo \
    --region us-east-1

# 2. Login, Tag, and Push Docker image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker tag assistant-service:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/financial-assistant-repo:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/financial-assistant-repo:latest

# 3. Create ECS Cluster and Service
aws ecs create-cluster --cluster-name financial-assistant-cluster
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service \
    --cluster financial-assistant-cluster \
    --service-name financial-assistant-service \
    --task-definition financial-assistant-task \
    --desired-count 1 \
    --launch-type "FARGATE" \
    --network-configuration "awsvpcConfiguration={subnets=SUBNET_ID,securityGroups=SECURITY_GROUP_ID,assignPublicIp=ENABLED}"
```

Create a `task-definition.json` file:

```json
{
    "family": "financial-assistant-task",
    "networkMode": "awsvpc",
    "containerDefinitions": [
        {
            "name": "financial-assistant-container",
            "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/financial-assistant-repo:latest",
            "portMappings": [
                {
                    "containerPort": 8080,
                    "hostPort": 8080
                }
            ],
            "essential": true,
            "environment": [
                { "name": "GOOGLE_PROJECT_ID", "value": "YOUR_PROJECT" },
                { "name": "GOOGLE_LOCATION", "value": "us-central1" },
                { "name": "NEO4J_URI", "value": "neo4j+s://XXXXXXXX" },
                { "name": "NEO4J_USERNAME", "value": "neo4j" },
                { "name": "NEO4J_PASSWORD", "value": "XXXXXXXX" },
                { "name": "SEC_USER_AGENT", "value": "Your Name your.email@provider.com" }
            ]
        }
    ],
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "4096",
    "memory": "8192"
}
```

## Key Design Decisions

### Why Neo4j for Financial Data?
Financial information is inherently interconnected. Companies operate in sectors, have competitors, file reports, and are affected by market events. A graph database makes exploring these complex relationships natural and efficient.

### Root/Sub-Agent Architecture
Using Google's ADK, the separation between a root agent (orchestrator) and specialist sub-agents creates a clean division of responsibilities, making the system modular and easier to extend with new tools.

### Hybrid Search Strategy
Different financial questions require different approaches. The system automatically selects the right tool: structured graph queries for quantitative facts (e.g., revenue) and semantic vector search for qualitative insights (e.g., management sentiment).

### LLM-Powered Entity Extraction
SEC filings are dense and unstructured. Using an LLM to intelligently extract key entities like risks, events, and strategies is far more flexible and robust than building rigid parsers.

## Limitations and Future Work

The current assistant does not build charts, has no conversational memory, and the predictive model is simplified. Future work could focus on:

- **Richer Predictions**: Integrating news sentiment, macroeconomic indicators, and alternative data into the prediction model
- **Backtesting Engine**: Building a framework to rigorously evaluate the historical performance of prediction models
- **MLOps Pipeline**: Creating an automated model retraining pipeline to keep the price predictors up-to-date
- **Advanced Visualizations**: Implementing dynamic chart generation to visually represent financial data and trends
- **Conversational Memory**: Using a tool like LangChain's ConversationBufferMemory to enable multi-turn, context-aware conversations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This system is for educational and research purposes only. The stock price predictions should not be considered financial advice and are not suitable for actual trading decisions. Always consult with qualified financial professionals before making investment decisions.