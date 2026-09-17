# RevenueOS 🚀
### Autonomous Revenue Leak Intelligence & GenAI Recovery Engine

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![AWS SAM](https://img.shields.io/badge/AWS-SAM%20Serverless-orange.svg)](https://aws.amazon.com/serverless/sam/)
[![Amazon Bedrock](https://img.shields.io/badge/Amazon%20Bedrock-Claude%203.5%20Sonnet-purple.svg)](https://aws.amazon.com/bedrock/)
[![Pydantic](https://img.shields.io/badge/Validation-Pydantic%20v2-green.svg)](https://docs.pydantic.dev/)
[![Tests](https://img.shields.io/badge/Tests-64%20Passed-brightgreen.svg)](tests/)

RevenueOS is an enterprise-grade, serverless data intelligence platform engineered to detect, statistically model, diagnose, and recover hidden revenue leaks across retail and e-commerce operations.

Built natively on AWS serverless primitives and Amazon Bedrock foundation models, RevenueOS unifies **rigorous statistical anomaly detection**, **censored time-series demand estimation**, and **state-of-the-art GenAI structured reasoning** into an autonomous financial recovery engine.

---

## 🧠 AI / ML & Statistical Intelligence Architecture

RevenueOS combines statistical learning algorithms for quantitative signal detection with generative AI for operational root-cause diagnosis.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           AI & STATISTICAL INTELLIGENCE STACK                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  [ QUANTITATIVE SIGNAL LAYER ]                                                          │
│  ├── Censored Demand Estimation (Stockouts)    ──► Non-censored sales velocity modeling │
│  ├── Binomial Proportion Z-Score Testing       ──► Statistical return anomaly detection │
│  ├── Funnel Drop-off Friction Analysis         ──► Multi-stage conversion abandonment   │
│  └── Dynamic Margin & Pricing Dispersion       ──► Margin erosion & discount anomalies  │
│                                      │                                                  │
│                                      ▼                                                  │
│  [ COMPOSITE OPPORTUNITY SCORING ENGINE ]                                               │
│  └── Multi-factor ranking: Log-impact (40%) + Confidence (30%) + Velocity (20%) + Sample (10%) │
│                                      │                                                  │
│                                      ▼                                                  │
│  [ GENAI STRUCTURED REASONING LAYER (Amazon Bedrock) ]                                  │
│  ├── Foundation Model: Anthropic Claude 3.5 Sonnet via Bedrock Converse API            │
│  ├── Zero-Hallucination Grounding Protocol (Immutable Financial Context Injection)     │
│  ├── Tool Calling / Structured Outputs (`publish_leak_analysis` tool spec)             │
│  ├── Pydantic v2 Schema Enforcement (`PlaybookSchema`)                                  │
│  └── Adversarial Prompt Injection Defense & Security Guardrails                         │
│                                      │                                                  │
│                                      ▼                                                  │
│  [ INTERACTIVE COPILOT: 'Ask RevenueOS' ]                                               │
│  └── Natural language financial copilot querying DynamoDB GSI verified records          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. GenAI Root-Cause Reasoner (Amazon Bedrock & Claude 3.5 Sonnet)

RevenueOS leverages Anthropic Claude 3.5 Sonnet via Amazon Bedrock (`us.anthropic.claude-3-5-sonnet-20240620-v1:0`) to synthesize operational root causes and generate prescriptive step-by-step remediation playbooks.

#### 🛡️ Zero-Hallucination Grounding Protocol
LLMs often hallucinate financial figures. RevenueOS eliminates financial hallucinations with an immutable grounding policy:
- Deterministic leak metrics and financial telemetry are precomputed in the quantitative engine.
- Bedrock system prompts enforce an explicit mandate: **all figures, percentages, currency amounts (₹), SKU identifiers, and timestamps are immutable facts**.
- The model is restricted from altering, recalculating, rounding, or inventing numbers.

#### 🛠️ Structured Function Calling & Tool Use
Instead of unstructured free-text generation, RevenueOS forces Claude 3.5 Sonnet to execute strict tool calls using Bedrock's `toolConfig` and `toolChoice`:
- **Tool Name**: `publish_leak_analysis`
- **Enforced JSON Schema**:
  ```json
  {
    "executive_summary": "string (1-800 chars)",
    "primary_root_cause": "string (1-1200 chars)",
    "contributing_factors": ["array of strings (max 12)"],
    "recommended_action": "string (1-1200 chars)",
    "expected_recovery_pct": "number (bounded 0.00 to 1.00)",
    "operational_effort": "LOW | MEDIUM | HIGH"
  }
  ```
- **Pydantic Validation**: All outputs are parsed directly into `PlaybookSchema` (Pydantic v2) with frozen runtime immutability and strict type constraints before persisting to DynamoDB.

#### 🤖 Interactive Financial Copilot (`Ask RevenueOS`)
- **Endpoint**: `POST /ai/investigate`
- Retail operators can ask natural language questions regarding their revenue leaks (e.g., *"Why did return rates spike on SKU-789?"*).
- Dynamically retrieves active verified leaks from DynamoDB GSI1 (`TENANT#{id}#LEAKS`) sorted by Opportunity Score.
- Injects verified database records directly into the LLM context for real-time, grounded executive insights.
- **Adversarial Guardrails**: Automatically detects and deflects prompt injection attacks, developer mode bypasses, and out-of-scope queries.

---

### 2. Statistical Anomaly & Demand Estimation Engines

The quantitative layer detects leak signals before passing them to the AI reasoning engine:

#### 📉 Censored Demand Estimation (Stockout Leaks)
- When a product is out of stock, observed sales drop to zero. Standard moving averages severely underestimate true customer demand due to **right-censorship**.
- The algorithm filters out zero-inventory intervals (`_as_active_demand_samples`), isolating active selling days.
- Calculates sample mean $\mu$ and sample variance $s^2$:
  $$s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \mu)^2$$
- Computes **Two-Tailed 95% Confidence Intervals** ($Z = 1.96$) to estimate true uncaptured demand and revenue loss with standard error margins.
- Derives statistical confidence dynamically from the **Coefficient of Variation** ($CV = \frac{\sigma}{\mu}$) and active sample depth.

#### 🔄 Binomial Proportion Hypothesis Testing (Return Rate Spikes)
- Detects sudden surges in return velocity using one-tailed hypothesis testing ($Z_{\alpha=0.05} = 1.645$).
- Evaluates statistical significance between the rolling baseline return rate and the current evaluation window.
- Isolates SKU-level defect correlation and reason prevalence vectors (e.g., sizing mismatches, damaged packaging).

#### 🏷️ Discount Abuse & Negative Margin Detection
- Identifies unauthorized coupon stacking, anomalous markdown velocity, and transactions where unit gross margin drops below cost-of-goods-sold thresholds.

#### 🛒 Multi-Stage Checkout Funnel Modeling
- Models step-by-step conversion drop-offs (Cart $\to$ Shipping $\to$ Payment $\to$ Completed Order).
- Identifies anomalous stage drop-offs deviating significantly from cohort baselines.

---

### 3. Composite Opportunity Scoring Engine

RevenueOS ranks all detected leaks using a deterministic, multi-factor scoring function bounded strictly between **0 and 100**:

$$\text{Opportunity Score} = \min\left(100, \text{Financial Scale} + \text{Confidence} + \text{Urgency} + \text{Evidence}\right)$$

| Component | Weight | Mathematical Formulation | Description |
|---|---|---|---|
| **Financial Impact** | 0 – 40 pts | $\frac{\log_{10}(\max(1000, \text{Impact})) - 3}{3} \times 40$ | Logarithmic scaling from ₹1,000 to ₹10,00,000 to prevent whale-order distortion |
| **Algorithmic Confidence** | 0 – 30 pts | $\text{Confidence} \times 30$ | Derived from sample depth, $Z$-score significance, and variance stability |
| **Urgency / Velocity** | 0 – 20 pts | $\text{Urgency Weight} \times 20$ | Capital bleed velocity and anomaly persistence |
| **Evidence Sample Strength** | 0 – 10 pts | $\text{Evidence Strength} \times 10$ | Transaction density and correlation factor support |

---

## 🏗️ End-to-End System Architecture

```
[ Shopify / CSV Feeds ]
           │
           ▼
[ S3 Ingestion Bucket ]
           │ (ObjectCreated Event)
           ▼
[ Stream Normalizer (Lambda) ]
           │ (Pydantic Canonical Entities)
           ▼
[ DynamoDB Core (Single-Table) ] ◄── GSI1 / GSI2 / GSI3
           │
           ▼
[ Step Functions: Leak Orchestrator ]
   ├── Stockout Detector (Censored Demand Modeling)
   ├── Return Spike Detector (Binomial Z-Score)
   ├── Discount Abuse Detector (Margin Erosion)
   ├── Checkout Funnel Detector (Drop-off Modeling)
   └── Underpriced SKU Detector (Elasticity Dispersion)
           │
           ▼
[ Unified Opportunity Scoring Engine ]
           │
           ▼
[ Amazon Bedrock GenAI Synthesizer (Claude 3.5 Sonnet) ]
           │ (Function Calling & Pydantic Validation)
           ▼
[ Verified Recovery Playbooks & EventBridge Alerts ]
```

---

## 📁 Repository Structure

```plaintext
RevenueOS/
├── backend/
│   ├── functions/
│   │   ├── ai/
│   │   │   ├── bedrock_reasoner.py    # Claude 3.5 Sonnet Bedrock Converse & Playbook synthesis
│   │   │   └── investigate_handler.py # 'Ask RevenueOS' interactive copilot with guardrails
│   │   ├── connections/
│   │   │   └── shopify_connector.py   # Shopify GraphQL/REST automated ingestion
│   │   ├── dashboard/
│   │   │   └── api_handlers.py        # Leak queries, summary KPIs & aggregations
│   │   ├── ingestion/
│   │   │   └── presigned_url.py       # Secure S3 presigned PUT URL generator
│   │   ├── leak_engine/               # Specialized statistical detection engines
│   │   │   ├── stockout_detector.py   # Censored demand estimation engine
│   │   │   ├── return_detector.py     # Return rate spike & defect anomaly detector
│   │   │   ├── pricing_detector.py    # Margin erosion & underpriced SKU detector
│   │   │   ├── funnel_detector.py     # Conversion funnel drop-off engine
│   │   │   └── retention_detector.py  # Customer cohort churn & LTV leak engine
│   │   └── normalization/
│   │       ├── csv_normalizer.py      # High-throughput CSV stream parser & batch writer
│   │       └── error_handler.py       # DLQ router & structured error telemetry
│   └── shared/
│       ├── calculations/
│       │   ├── engines.py             # Core mathematical algorithms & statistical tests
│       │   ├── scoring.py             # Composite Opportunity Scoring formula
│       │   └── aggregations.py        # Financial aggregations & KPI rollup logic
│       ├── resilience.py              # Exponential backoff with jitter & circuit breakers
│       └── schemas/
│           └── canonical.py           # Canonical Pydantic schemas (Order, Leak, Playbook)
├── infrastructure/
│   ├── statemachine/
│   │   └── leak_orchestrator.asl.json # AWS Step Functions state machine definition
│   └── template.yaml                  # AWS SAM CloudFormation template
├── tests/                             # 64 verified unit & integration tests
├── data/
│   ├── external/                      # Data pipeline scripts & reference datasets
│   └── sample/                        # Verified e-commerce transaction test fixtures
├── manual_test_inputs/                # CSV test upload payloads
├── requirements.txt                   # Core runtime dependencies
├── requirements-dev.txt               # Testing & linting dependencies
├── .gitignore                         # Python, SAM & OS ignore patterns
└── README.md                          # Technical architecture & documentation
```

---

## ⚙️ Prerequisites & Tech Stack

- **Runtime**: Python 3.12+ (tested with Python 3.12 and 3.14)
- **Foundation Model**: Anthropic Claude 3.5 Sonnet (`us.anthropic.claude-3-5-sonnet-20240620-v1:0`) via Amazon Bedrock
- **Cloud Primitives**: AWS Lambda, DynamoDB (Single-Table), Amazon S3, AWS Step Functions, Amazon EventBridge, Amazon Cognito, Amazon SQS
- **Infrastructure as Code**: AWS Serverless Application Model (SAM)
- **Data Validation**: Pydantic v2

---

## 🚀 Quickstart & Testing

### 1. Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/unnatikdm/RevenueOS.git
cd RevenueOS

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
```

### 2. Run Test Suite

Run the full automated test suite covering all statistical algorithms, schema validations, resilience circuit breakers, and Lambda handlers:

```bash
python -m unittest discover -s tests
```

**Status**: 64 tests passing with 100% success rate.

---

## ☁️ Deployment (AWS SAM)

```bash
# 1. Build serverless artifacts
sam build -t infrastructure/template.yaml

# 2. Deploy to AWS environment
sam deploy --guided
```

Configuration parameters:
- `Environment`: `dev` | `staging` | `prod`
- `BedrockModelId`: `us.anthropic.claude-3-5-sonnet-20240620-v1:0`
- `CognitoDomain`: Domain prefix for Cognito User Pool Hosted UI

---

## 🔌 API Reference

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/uploads/presigned-url` | Cognito JWT | Generate S3 presigned PUT URL for raw CSV ingest |
| `GET` | `/dashboard/summary` | Cognito JWT | Get high-level revenue leak and recovery KPIs |
| `GET` | `/leaks` | Cognito JWT | Query active detected leaks sorted by Opportunity Score |
| `POST` | `/leaks/{id}/playbook` | Cognito JWT | Trigger Bedrock Claude 3.5 AI root-cause synthesis |
| `POST` | `/ai/investigate` | Cognito JWT | Natural language copilot query ('Ask RevenueOS') |
| `POST` | `/connectors/shopify/sync` | Cognito JWT | Trigger automated ingestion from Shopify |

---
