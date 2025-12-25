# AWS Lambda Deployment Guide for Trading Simulation

## Overview
Deploy the simulation to AWS Lambda to run automatically without your computer.

## Cost Estimate
- **Free Tier**: 1M requests/month + 400,000 GB-seconds compute
- **Your Usage**: ~2 runs/day × 30 days = 60 runs/month
- **Estimated Cost**: $0 (within free tier) or <$5/month

## Prerequisites
1. AWS Account
2. AWS CLI installed: `pip install awscli`
3. Docker installed (for building Lambda package)

## Deployment Steps

### 1. Create Lambda Function Package

```bash
# Create deployment directory
mkdir lambda_deployment
cd lambda_deployment

# Copy simulation files
cp ../american\ stocks/daily_trading_simulation.py .
cp ../american\ stocks/stock_picker_advanced_quantitative.py .
cp ../american\ stocks/stock_picker_nlg_explanations.py .

# Create requirements.txt
cat > requirements.txt << EOF
pandas
numpy
yfinance
feedparser
transformers
scikit-learn
statsmodels
arch
scipy
matplotlib
EOF

# Build Lambda package (using Docker for compatibility)
docker run -v "$PWD":/var/task "public.ecr.aws/sam/build-python3.10" /bin/sh -c "pip install -r requirements.txt -t python/lib/python3.10/site-packages/; exit"

# Package everything
zip -r simulation_lambda.zip python daily_trading_simulation.py stock_picker_*.py
```

### 2. Create Lambda Handler

Create `lambda_handler.py`:

```python
import json
import subprocess
import os
from datetime import datetime, timedelta

def lambda_handler(event, context):
    """AWS Lambda handler for trading simulation."""
    
    # Calculate dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Run simulation
    try:
        result = subprocess.run([
            'python3',
            'daily_trading_simulation.py',
            '--start-date', start_date,
            '--end-date', end_date,
            '--twice-daily'
        ], capture_output=True, text=True, timeout=600)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Simulation completed successfully',
                'output': result.stdout
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Simulation failed',
                'error': str(e)
            })
        }
```

### 3. Deploy to AWS

```bash
# Configure AWS CLI
aws configure

# Create Lambda function
aws lambda create-function \
    --function-name TradingSimulation \
    --runtime python3.10 \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
    --handler lambda_handler.lambda_handler \
    --zip-file fileb://simulation_lambda.zip \
    --timeout 900 \
    --memory-size 3008

# Create EventBridge rules for twice-daily execution
# Morning: 9:00 AM EST
aws events put-rule \
    --name TradingSimulation-Morning \
    --schedule-expression "cron(0 14 * * ? *)"

# Afternoon: 4:30 PM EST  
aws events put-rule \
    --name TradingSimulation-Afternoon \
    --schedule-expression "cron(30 21 * * ? *)"

# Add Lambda permissions
aws lambda add-permission \
    --function-name TradingSimulation \
    --statement-id EventBridgeMorning \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn arn:aws:events:us-east-1:YOUR_ACCOUNT_ID:rule/TradingSimulation-Morning

# Connect rules to Lambda
aws events put-targets \
    --rule TradingSimulation-Morning \
    --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:TradingSimulation"
```

### 4. Store Results in S3

Add to lambda_handler.py:

```python
import boto3

s3 = boto3.client('s3')

# Upload results
s3.upload_file(
    'simulation_results/trade_log.csv',
    'your-bucket-name',
    f'results/{datetime.now().strftime("%Y%m%d_%H%M%S")}_trade_log.csv'
)
```

## Monitoring

View logs:
```bash
aws logs tail /aws/lambda/TradingSimulation --follow
```

## Cost Management

Monitor usage:
```bash
aws cloudwatch get-metric-statistics \
    --namespace AWS/Lambda \
    --metric-name Invocations \
    --dimensions Name=FunctionName,Value=TradingSimulation \
    --start-time 2025-12-01T00:00:00Z \
    --end-time 2025-12-31T23:59:59Z \
    --period 86400 \
    --statistics Sum
```

## Cleanup

Delete resources:
```bash
aws lambda delete-function --function-name TradingSimulation
aws events delete-rule --name TradingSimulation-Morning
aws events delete-rule --name TradingSimulation-Afternoon
```
