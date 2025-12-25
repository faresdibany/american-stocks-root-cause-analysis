# RCA Pipeline - AWS Infrastructure Deployment Guide

## Overview

This guide walks you through deploying the complete RCA Pipeline infrastructure to AWS using Terraform.

## Infrastructure Components

✅ **Core Services** (50+ resources):
- VPC with public/private subnets across 3 AZs
- 8 Lambda functions for pipeline stages
- Step Functions state machine for orchestration
- ECS Fargate cluster for heavy compute
- RDS PostgreSQL for structured data
- DynamoDB tables for job tracking
- ElastiCache Redis for caching
- S3 bucket for artifacts
- API Gateway with Cognito authentication
- CloudWatch monitoring and alarms
- SNS for notifications
- SQS for job queuing
- EventBridge for scheduling
- WAF for API protection
- Secrets Manager for credentials

## Prerequisites

### 1. Tools Installation

```powershell
# Install Terraform
choco install terraform

# Install AWS CLI
choco install awscli

# Verify
terraform version  # Should be >= 1.5.0
aws --version
```

### 2. AWS Account Setup

- AWS Account with admin access
- AWS CLI configured
- Sufficient service limits (check AWS Service Quotas)

### 3. Cost Awareness

**Estimated Monthly Cost**: $250-300
- Development: ~$150/month
- Production: ~$300/month + usage

## Step-by-Step Deployment

### Phase 1: Preparation (30 minutes)

#### 1.1 Clone and Navigate

```powershell
cd "c:\Users\fares\OneDrive\Desktop\Stock Picker With Root Cause Analysis\american-stocks-root-cause-analysis"
cd aws\terraform
```

#### 1.2 Configure AWS Credentials

```powershell
# Option A: AWS CLI
aws configure
# Enter your Access Key, Secret Key, Region (us-east-1), Output (json)

# Option B: Environment Variables
$env:AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY"
$env:AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_KEY"
$env:AWS_DEFAULT_REGION = "us-east-1"

# Verify
aws sts get-caller-identity
```

#### 1.3 Create Configuration File

```powershell
# Copy example config
cp terraform.tfvars.example terraform.tfvars

# Edit configuration
notepad terraform.tfvars
```

**Minimum required changes**:
```hcl
# terraform.tfvars
aws_region  = "us-east-1"
environment = "prod"

# Add your email for notifications
notification_emails = [
  "your-email@example.com"
]
```

### Phase 2: Lambda Package Preparation (15 minutes)

Lambda functions need deployment packages before Terraform can create them.

#### Option A: Quick Start (Use Placeholders)

```powershell
# Create placeholder packages
cd ..\lambda
mkdir dist -Force

# Create empty zip files for each function
$functions = @(
  "pipeline-trigger", "driver-analysis", "quant-news-analysis",
  "social-sentiment", "merge-consolidate", "generate-reports",
  "authorizer", "job-status"
)

foreach ($func in $functions) {
    "# Placeholder" | Out-File -FilePath "placeholder.py"
    Compress-Archive -Path "placeholder.py" -DestinationPath "dist\$func.zip" -Force
}

Remove-Item "placeholder.py"
cd ..\terraform
```

#### Option B: Build Actual Packages (Recommended)

See `../lambda/README.md` for detailed instructions to build actual Lambda packages with dependencies.

### Phase 3: Terraform Initialization (5 minutes)

```powershell
# Initialize Terraform
terraform init

# You should see:
# - Provider downloads (AWS provider ~500MB)
# - Module initialization
# - Backend configuration
```

**Expected output**:
```
Initializing modules...
Initializing the backend...
Initializing provider plugins...
- Finding hashicorp/aws versions matching "~> 5.0"...
- Installing hashicorp/aws v5.x.x...

Terraform has been successfully initialized!
```

### Phase 4: Plan Review (10 minutes)

```powershell
# Generate execution plan
terraform plan -out=tfplan

# Review the plan carefully
# Expected: 50+ resources to create
```

**Review checklist**:
- ✅ VPC and subnets being created
- ✅ RDS instance configuration correct
- ✅ Lambda functions with correct memory/timeout
- ✅ S3 bucket with unique name
- ✅ No unexpected deletions (should be all creates on first run)

### Phase 5: Deploy Infrastructure (20-30 minutes)

```powershell
# Apply the plan
terraform apply tfplan

# This will take 20-30 minutes
# Longest operations:
# - RDS instance creation (~15 min)
# - NAT Gateways (~5 min)
# - VPC and networking (~5 min)
```

**Monitor progress**:
```powershell
# In another terminal, watch CloudFormation events
aws cloudformation describe-stack-events `
  --stack-name rca-pipeline-prod `
  --query 'StackEvents[*].[Timestamp,ResourceStatus,LogicalResourceId]' `
  --output table
```

### Phase 6: Post-Deployment Configuration (15 minutes)

#### 6.1 Save Outputs

```powershell
# Save all outputs to file
terraform output -json > infrastructure-outputs.json

# View important outputs
terraform output api_gateway_url
terraform output cognito_user_pool_id
terraform output artifacts_bucket_name
```

#### 6.2 Initialize Database

```powershell
# Get RDS endpoint
$RDS_ENDPOINT = terraform output -raw rds_endpoint

# Get database credentials from Secrets Manager
$SECRET_ARN = terraform output -raw rds_secret_arn
$CREDS = aws secretsmanager get-secret-value --secret-id $SECRET_ARN --query SecretString --output text | ConvertFrom-Json

# Connect and run schema
psql -h $RDS_ENDPOINT -U $CREDS.username -d rcapipeline -f ../database/schema.sql
```

#### 6.3 Create First Cognito User

```powershell
# Get Cognito User Pool ID
$USER_POOL_ID = terraform output -raw cognito_user_pool_id

# Create admin user
aws cognito-idp admin-create-user `
  --user-pool-id $USER_POOL_ID `
  --username admin@yourdomain.com `
  --user-attributes Name=email,Value=admin@yourdomain.com Name=email_verified,Value=true `
  --temporary-password "TempPassword123!" `
  --message-action SUPPRESS

# User will need to change password on first login
```

#### 6.4 Deploy Lambda Code

```powershell
# Build actual Lambda packages
cd ..\lambda
.\build-packages.ps1

# Deploy to Lambda
foreach ($func in Get-ChildItem -Path dist -Filter *.zip) {
    $functionName = "rca-pipeline-prod-" + $func.BaseName
    aws lambda update-function-code `
      --function-name $functionName `
      --zip-file "fileb://dist/$($func.Name)"
}

cd ..\terraform
```

#### 6.5 Verify Deployment

```powershell
# Test API health endpoint
$API_URL = terraform output -raw api_gateway_url
curl "$API_URL/health"

# Check Lambda functions
aws lambda list-functions --query 'Functions[?starts_with(FunctionName, `rca-pipeline-prod`)].FunctionName'

# Check Step Functions
aws stepfunctions list-state-machines --query 'stateMachines[?starts_with(name, `rca-pipeline-prod`)].name'
```

## Testing the Pipeline

### 1. Authenticate User

```powershell
# Using AWS CLI
aws cognito-idp initiate-auth `
  --auth-flow USER_PASSWORD_AUTH `
  --client-id $(terraform output -raw cognito_client_id) `
  --auth-parameters USERNAME=admin@yourdomain.com,PASSWORD=YourNewPassword123!

# Save the IdToken from response
$ID_TOKEN = "eyJraWQiOiI..."
```

### 2. Start Pipeline Execution

```powershell
# Call API
$API_URL = terraform output -raw api_gateway_url

$body = @{
    tickers = @("AAPL", "MSFT", "NVDA")
    period = "6mo"
    interval = "1d"
} | ConvertTo-Json

curl -X POST "$API_URL/analyze" `
  -H "Authorization: Bearer $ID_TOKEN" `
  -H "Content-Type: application/json" `
  -d $body
```

### 3. Check Execution Status

```powershell
# Get execution ARN from response
$EXECUTION_ARN = "arn:aws:states:us-east-1:123456789012:execution:rca-pipeline-prod:..."

# Check status
aws stepfunctions describe-execution --execution-arn $EXECUTION_ARN

# Watch logs
aws logs tail /aws/states/rca-pipeline-prod --follow
```

### 4. Download Results

```powershell
# Get job ID from execution
$JOB_ID = "20241225_143022"

# Download from S3
$BUCKET = terraform output -raw artifacts_bucket_name
aws s3 cp s3://$BUCKET/ranked_signals/$JOB_ID/ . --recursive
```

## Monitoring

### CloudWatch Dashboards

```powershell
# Open dashboard
start "https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=rca-pipeline-prod"
```

### View Logs

```powershell
# Lambda logs
aws logs tail /aws/lambda/rca-pipeline-prod-driver-analysis --follow

# Step Functions logs
aws logs tail /aws/states/rca-pipeline-prod --follow

# API Gateway logs
aws logs tail /aws/apigateway/rca-pipeline-prod --follow
```

### Check Alarms

```powershell
# List all alarms
aws cloudwatch describe-alarms `
  --alarm-name-prefix rca-pipeline-prod `
  --query 'MetricAlarms[*].[AlarmName,StateValue]' `
  --output table
```

## Cost Management

### Monitor Spending

```powershell
# Get current month cost
aws ce get-cost-and-usage `
  --time-period Start=$(Get-Date -Format yyyy-MM-01),End=$(Get-Date -Format yyyy-MM-dd) `
  --granularity MONTHLY `
  --metrics UnblendedCost `
  --filter file://cost-filter.json
```

### Set Budget Alerts

```powershell
# Create budget
aws budgets create-budget `
  --account-id $(aws sts get-caller-identity --query Account --output text) `
  --budget file://budget.json
```

## Troubleshooting

### Common Issues

#### 1. Terraform Init Fails

**Error**: "Failed to download provider"

**Solution**:
```powershell
# Clear cache
Remove-Item -Recurse -Force .terraform
terraform init -upgrade
```

#### 2. RDS Creation Timeout

**Error**: "Error waiting for RDS instance"

**Solution**:
```powershell
# Increase timeout in terraform
# Add to modules/rds/main.tf:
timeouts {
  create = "60m"
  update = "60m"
  delete = "60m"
}
```

#### 3. Lambda VPC Timeout

**Error**: "Task timed out after 300.00 seconds"

**Solution**:
```powershell
# Check NAT Gateway
aws ec2 describe-nat-gateways `
  --filter "Name=state,Values=available" `
  --query 'NatGateways[*].[NatGatewayId,State,VpcId]'

# Verify route tables
aws ec2 describe-route-tables `
  --query 'RouteTables[*].[RouteTableId,VpcId,Routes]'
```

#### 4. API Gateway 403 Forbidden

**Error**: "User is not authorized"

**Solution**:
```powershell
# Verify Cognito token
aws cognito-idp get-user --access-token $ACCESS_TOKEN

# Check API Gateway authorizer
aws apigateway get-authorizers --rest-api-id $(terraform output -raw api_gateway_id)
```

## Updating Infrastructure

### Update Lambda Code Only

```powershell
# Build and deploy
cd ..\lambda
.\build-packages.ps1
.\deploy-functions.ps1
```

### Update Infrastructure Configuration

```powershell
# Modify terraform.tfvars
notepad terraform.tfvars

# Plan changes
terraform plan

# Apply specific module
terraform apply -target=module.lambda
```

### Update All

```powershell
terraform apply
```

## Destroying Infrastructure

**⚠️ WARNING**: This will delete all data and resources.

```powershell
# Plan destruction
terraform plan -destroy

# Destroy
terraform destroy

# Type 'yes' to confirm

# Manual cleanup (if needed)
# - Empty S3 buckets manually
# - Delete RDS snapshots
# - Remove CloudWatch log groups
```

## Backup and Disaster Recovery

### Backup State File

```powershell
# Backup terraform state
cp terraform.tfstate "terraform.tfstate.backup.$(Get-Date -Format yyyyMMdd-HHmmss)"

# Upload to S3
aws s3 cp terraform.tfstate s3://your-backup-bucket/terraform-state/
```

### Backup RDS

```powershell
# Create manual snapshot
aws rds create-db-snapshot `
  --db-instance-identifier rca-pipeline-prod-db `
  --db-snapshot-identifier rca-pipeline-manual-$(Get-Date -Format yyyyMMdd)
```

### Export DynamoDB

```powershell
# Export to S3
$BUCKET = terraform output -raw artifacts_bucket_name

aws dynamodb export-table-to-point-in-time `
  --table-arn $(terraform output -raw jobs_table_arn) `
  --s3-bucket $BUCKET `
  --s3-prefix dynamodb-backup/jobs/
```

## Next Steps

1. ✅ Set up CI/CD pipeline (GitHub Actions)
2. ✅ Configure custom domain for API
3. ✅ Implement monitoring dashboards
4. ✅ Set up log aggregation
5. ✅ Configure backup automation
6. ✅ Implement blue-green deployment
7. ✅ Add integration tests
8. ✅ Configure WAF rules for production
9. ✅ Set up cost alerts
10. ✅ Document API endpoints

## Support

- **Documentation**: See `/documentation` folder
- **Issues**: Create GitHub issue
- **AWS Support**: Enable AWS Support plan for production

---

**Deployment Time**: ~90 minutes for first-time setup
**Maintenance**: ~2 hours/month
**Skill Level**: Intermediate Terraform & AWS knowledge required
