# RevenueOS 🚀
### Autonomous Revenue Leak Intelligence & Recovery Engine

RevenueOS is an enterprise-grade, serverless data intelligence platform designed to detect, diagnose, and recover hidden revenue leaks across e-commerce operations. Built natively on AWS cloud primitives, it combines event-driven serverless computing with Amazon Bedrock foundation models to deliver real-time leak detection and automated recovery playbooks.

---

## 🌟 Key Capabilities

- **Streaming Data Ingestion**: Secure S3 Presigned URL generation for high-throughput raw CSV and XLSX transaction uploads.
- **Resilient Stream Normalization**: Memory-efficient stream parsing, Pydantic validation, canonical normalization, and high-throughput DynamoDB batch writes with automatic backoff.
- **Specialized Leak Detection Engines**:
  - 📉 **Stockout Leak Engine**: Identifies uncaptured sales demand and stockout-driven revenue loss using historical sales velocity.
  - 🔄 **Return Spike Engine**: Flags statistical return rate anomalies and product defect trends.
  - 🏷️ **Discount Abuse & Margin Erosion Engine**: Detects unauthorized coupon stacking and negative-margin transactions.
  - 🛒 **Checkout Funnel Drop-off Engine**: Analyzes step-by-step funnel drop-offs and friction points.
  - 💰 **Underpriced SKU Engine**: Evaluates catalog pricing against velocity and baseline margins.
- **GenAI Root Cause Synthesis**: Integrates Amazon Bedrock (Anthropic Claude 3.5 Sonnet) to synthesize root causes and generate actionable step-by-step recovery playbooks.
- **Enterprise Security & Auth**: AWS Cognito User Pools with OAuth2/JWT HTTP API authorization and fine-grained IAM least-privilege policies.
- **Fault-Tolerant Architecture**: Circuit breakers, exponential backoff with jitter, and Dead-Letter Queues (DLQ) for bulletproof resilience.

---

## 🏗️ Architecture Overview

```
[ Data Sources / Shopify / CSV ]
                │
                ▼
       [ S3 Raw Ingestion Bucket ]
                │ (ObjectCreated)
                ▼
   [ CSV Normalizer (Lambda) ]
                │
                ▼
   [ DynamoDB Single-Table Core ] ◄─── GSI1 / GSI2 / GSI3
                │
                ▼
   [ Step Functions: Leak Orchestrator ]
        ├── Stockout Detector
        ├── Return Rate Spike Detector
        ├── Discount Abuse Detector
        ├── Funnel Abandonment Detector
        └── Underpriced SKU Detector
                │
                ▼
   [ Bedrock AI Synthesizer (Claude 3.5) ]
                │
                ▼
   [ EventBridge Event Bus & Alerts ]
```

---

## 📁 Repository Structure

```plaintext
RevenueOS/
├── backend/
│   ├── functions/
│   │   ├── ai/                    # Bedrock Claude 3.5 Sonnet AI synthesis
│   │   ├── connections/           # Shopify and external platform connectors
│   │   ├── dashboard/             # Aggregations, metrics, and KPI calculations
│   │   ├── ingestion/             # S3 presigned URL generation
│   │   ├── leak_engine/           # 5 specialized detection engines
│   │   └── normalization/         # CSV streaming, validation & error handler
│   └── shared/
│       ├── calculations/          # Scoring algorithms & analytical routines
│       ├── schemas/               # Canonical Pydantic data schemas
│       └── resilience.py          # Circuit breakers & retry decorators
├── infrastructure/
│   ├── statemachine/
│   │   └── leak_orchestrator.asl.json # AWS Step Functions state machine
│   └── template.yaml              # AWS SAM Infrastructure as Code (CloudFormation)
├── tests/                         # Full automated test suite (64 unit tests)
├── data/
│   ├── external/                  # Dataset scripts and reference data
│   └── sample/                    # Verified test CSV datasets
├── manual_test_inputs/            # Sample inputs for local verification
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Testing & dev dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## ⚙️ Prerequisites

- **Python**: Python 3.12+ (tested with Python 3.12 & 3.14)
- **AWS CLI**: Configured with appropriate AWS credentials (`aws configure`)
- **AWS SAM CLI**: For local builds and cloud deployments
- **Git**: For version control

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/unnatikdm/RevenueOS.git
cd RevenueOS

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
```

### 2. Run Test Suite

RevenueOS includes 64 comprehensive unit and integration tests validating schemas, calculation algorithms, leak engines, resilience primitives, and Lambda handlers:

```bash
# Run all tests using unittest
python -m unittest discover -s tests

# Or run using pytest with coverage
pytest --cov=backend tests/
```

---

## ☁️ Deployment Guide (AWS SAM)

RevenueOS uses the AWS Serverless Application Model (SAM) for infrastructure provisioning:

### 1. Build the Application

```bash
sam build -t infrastructure/template.yaml
```

### 2. Deploy to AWS

```bash
sam deploy --guided
```

During deployment, SAM will prompt for configuration parameters:
- `Environment`: `dev`, `staging`, or `prod`
- `BedrockModelId`: Default is `us.anthropic.claude-3-5-sonnet-20240620-v1:0`
- `CognitoDomain`: Domain prefix for authentication

---

## 🔌 API Endpoints

All endpoints are protected by Amazon Cognito JWT Authorization:

| Method | Path | Description |
|---|---|---|
| `POST` | `/uploads/presigned-url` | Generate S3 presigned PUT URL for raw CSV ingest |
| `GET` | `/dashboard/summary` | Retrieve high-level revenue leak and recovery KPIs |
| `GET` | `/leaks` | Query active detected leaks filtered by type and severity |
| `POST` | `/leaks/{id}/playbook` | Trigger Bedrock AI synthesis to generate a recovery playbook |
| `POST` | `/connectors/shopify/sync` | Trigger on-demand sync from connected Shopify store |

---

## 🛡️ License

Proprietary and Confidential. All Rights Reserved.