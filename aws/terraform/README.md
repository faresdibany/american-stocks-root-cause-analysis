# RCA Pipeline - Terraform Infrastructure

This directory contains the Terraform infrastructure-as-code for deploying the RCA (Root Cause Analysis) Pipeline SaaS platform on AWS.

## Architecture Overview

The infrastructure follows a **serverless-first hybrid architecture**:

- **API Gateway** - RESTful API endpoints
- **Lambda Functions** - Lightweight compute for I/O-bound tasks
- **Step Functions** - Pipeline orchestration
- **ECS Fargate** - Heavy compute for advanced analytics
- **RDS PostgreSQL** - Structured data (users, subscriptions, billing)
- **DynamoDB** - Job tracking and caching
- **ElastiCache Redis** - Rate limiting and session management
- **S3** - Artifact storage
- **Cognito** - Authentication and authorization
- **CloudWatch** - Monitoring and logging
- **SNS** - Notifications
- **EventBridge** - Scheduled jobs

## Prerequisites

### 1. Install Tools

```powershell
# Terraform (Windows)
choco install terraform

# AWS CLI
choco install awscli

# Verify installations
terraform --version
aws --version
```

### 2. Configure AWS Credentials

```powershell
# Configure AWS CLI
aws configure

# Or set environment variables
$env:AWS_ACCESS_KEY_ID = "your-access-key"
$env:AWS_SECRET_ACCESS_KEY = "your-secret-key"
$env:AWS_DEFAULT_REGION = "us-east-1"
```

### 3. Prepare Lambda Deployment Packages

Before deploying, you need to create Lambda deployment packages:

```powershell
# Navigate to Lambda directory
cd ../lambda

# Create deployment packages (see ../lambda/README.md for details)
./build-packages.ps1
```

## Quick Start

### 1. Initialize Terraform

```powershell
cd aws/terraform
terraform init
```

### 2. Create Configuration

```powershell
# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your values
notepad terraform.tfvars
```

### 3. Plan Deployment

```powershell
# Review what will be created
terraform plan -out=tfplan
```

### 4. Deploy Infrastructure

```powershell
# Apply the plan
terraform apply tfplan
```

This will create approximately **50+ AWS resources**.

### 5. Get Outputs

```powershell
# View important outputs
terraform output

# Get specific output
terraform output api_gateway_url
terraform output cognito_user_pool_id
```

## Module Structure

```
terraform/
├── main.tf                      # Main configuration
├── variables.tf                 # Variable definitions
├── outputs.tf                   # Output definitions
├── terraform.tfvars.example     # Example configuration
├── modules/
│   ├── vpc/                     # Network infrastructure
│   ├── s3/                      # Storage
│   ├── dynamodb/                # NoSQL database
│   ├── rds/                     # PostgreSQL database
│   ├── elasticache/             # Redis cache
│   ├── lambda/                  # Lambda functions
│   ├── step_functions/          # State machine orchestration
│   ├── ecs/                     # Fargate tasks
│   ├── api_gateway/             # API Gateway
│   ├── cognito/                 # Authentication
│   ├── iam/                     # IAM roles and policies
│   ├── secrets/                 # Secrets Manager
│   ├── sns/                     # Notifications
│   ├── sqs/                     # Queues
│   ├── cloudwatch/              # Monitoring
│   ├── eventbridge/             # Scheduled events
│   ├── waf/                     # Web Application Firewall
│   └── timestream/              # Time-series database (optional)
```

## Configuration

### Key Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `aws_region` | AWS region | `us-east-1` |
| `environment` | Environment name | `prod` |
| `vpc_cidr` | VPC CIDR block | `10.0.0.0/16` |
| `rds_instance_class` | RDS instance type | `db.t3.medium` |
| `redis_node_type` | ElastiCache node type | `cache.t3.micro` |

See `variables.tf` for complete list.

### Cost Optimization

#### Development Environment

For development, use smaller instances:

```hcl
# terraform.tfvars
environment = "dev"

rds_instance_class = "db.t3.micro"
rds_multi_az = false

redis_node_type = "cache.t3.micro"
redis_num_nodes = 1

lambda_functions = {
  "driver-analysis" = {
    memory_size = 1024  # Reduced from 2048
    timeout = 180       # Reduced from 300
    ...
  }
}
```

**Estimated cost: ~$150/month**

#### Production Environment

For production, use default values with high availability:

```hcl
# terraform.tfvars
environment = "prod"

rds_instance_class = "db.t3.medium"
rds_multi_az = true

redis_node_type = "cache.t3.small"
redis_num_nodes = 2
```

**Estimated cost: ~$272/month + usage**

## Deployment Process

### Initial Deployment

```powershell
# 1. Initialize
terraform init

# 2. Validate
terraform validate

# 3. Plan
terraform plan -out=tfplan

# 4. Apply
terraform apply tfplan

# 5. Save outputs
terraform output -json > outputs.json
```

### Update Existing Infrastructure

```powershell
# 1. Review changes
terraform plan

# 2. Apply changes
terraform apply

# 3. Target specific module (if needed)
terraform apply -target=module.lambda
```

### Destroy Infrastructure

```powershell
# WARNING: This will delete ALL resources

# 1. Plan destruction
terraform plan -destroy

# 2. Destroy
terraform destroy

# 3. Confirm by typing 'yes'
```

## State Management

### Local State (Default)

Terraform state is stored locally in `terraform.tfstate`.

**⚠️ Warning**: Local state is not suitable for team environments.

### Remote State (Recommended)

For production, use S3 backend:

```hcl
# Uncomment in main.tf
backend "s3" {
  bucket         = "rca-pipeline-terraform-state"
  key            = "prod/terraform.tfstate"
  region         = "us-east-1"
  encrypt        = true
  dynamodb_table = "rca-pipeline-terraform-locks"
}
```

Create backend resources:

```powershell
# Run once to create state bucket
terraform init -backend=false
terraform apply -target=module.state_backend
terraform init -migrate-state
```

## Post-Deployment Tasks

### 1. Configure DNS (Optional)

Point your domain to the API Gateway:

```powershell
# Get API Gateway URL
$API_URL = terraform output -raw api_gateway_url

# Create CNAME record
# api.yourdomain.com -> $API_URL
```

### 2. Set Up Cognito Users

```powershell
# Get Cognito User Pool ID
$USER_POOL_ID = terraform output -raw cognito_user_pool_id

# Create admin user
aws cognito-idp admin-create-user `
  --user-pool-id $USER_POOL_ID `
  --username admin@yourdomain.com `
  --user-attributes Name=email,Value=admin@yourdomain.com `
  --temporary-password "TempPass123!" `
  --message-action SUPPRESS
```

### 3. Upload Lambda Code

```powershell
# Navigate to lambda directory
cd ../lambda

# Deploy all functions
./deploy-functions.ps1
```

### 4. Test the Pipeline

```powershell
# Get API URL
$API_URL = terraform output -raw api_gateway_url

# Test endpoint
curl "${API_URL}/health"
```

### 5. Set Up Monitoring

```powershell
# Get CloudWatch dashboard URL
aws cloudwatch get-dashboard `
  --dashboard-name "rca-pipeline-prod" `
  --query DashboardBody `
  --output text
```

## Monitoring

### CloudWatch Dashboards

Access dashboards:

```powershell
# Open in browser
start "https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=rca-pipeline-prod"
```

### View Logs

```powershell
# Lambda logs
aws logs tail /aws/lambda/rca-pipeline-prod-driver-analysis --follow

# Step Functions logs
aws logs tail /aws/states/rca-pipeline-prod --follow
```

### Check Alarms

```powershell
# List active alarms
aws cloudwatch describe-alarms --alarm-names rca-pipeline-prod-lambda-errors
```

## Troubleshooting

### Common Issues

#### 1. Lambda VPC Timeout

**Symptom**: Lambda functions timeout when accessing external APIs

**Solution**: Ensure NAT Gateway is properly configured

```powershell
# Check NAT Gateway
aws ec2 describe-nat-gateways --filter "Name=state,Values=available"
```

#### 2. RDS Connection Failed

**Symptom**: Cannot connect to RDS from Lambda

**Solution**: Check security group rules

```powershell
# Verify Lambda SG can access RDS SG on port 5432
terraform output rds_security_group_id
```

#### 3. Step Functions Execution Failed

**Symptom**: State machine fails at specific stage

**Solution**: Check CloudWatch logs

```powershell
# Get execution ARN from DynamoDB jobs table
aws logs get-log-events `
  --log-group-name /aws/states/rca-pipeline-prod `
  --log-stream-name <execution-id>
```

#### 4. High Costs

**Symptom**: AWS bill higher than expected

**Solution**: Check CloudWatch metrics

```powershell
# Check Lambda invocations
aws cloudwatch get-metric-statistics `
  --namespace AWS/Lambda `
  --metric-name Invocations `
  --start-time $(Get-Date).AddDays(-7) `
  --end-time $(Get-Date) `
  --period 86400 `
  --statistics Sum
```

## Security Best Practices

### 1. Use Secrets Manager

Never hardcode credentials:

```hcl
# Store in Secrets Manager
resource "aws_secretsmanager_secret" "reddit_api" {
  name = "${var.name_prefix}/reddit-api"
}

# Reference in Lambda
environment {
  REDDIT_SECRET_ARN = aws_secretsmanager_secret.reddit_api.arn
}
```

### 2. Enable Encryption

All data should be encrypted:

- ✅ S3 buckets (SSE-S3 or SSE-KMS)
- ✅ RDS (at rest encryption)
- ✅ DynamoDB (at rest encryption)
- ✅ Secrets Manager (KMS)

### 3. Restrict IAM Permissions

Follow principle of least privilege:

```hcl
# Bad: Wildcard permissions
Resource = "*"

# Good: Specific resources
Resource = "arn:aws:s3:::${var.bucket_name}/*"
```

### 4. Enable VPC Flow Logs

```powershell
# Enable VPC flow logs
terraform apply -target=module.vpc.aws_flow_log.main
```

## Maintenance

### Regular Tasks

#### Weekly

- Review CloudWatch alarms
- Check cost explorer
- Review failed executions

#### Monthly

- Update Lambda runtimes if needed
- Review and rotate secrets
- Optimize Lambda memory allocations
- Archive old S3 artifacts

#### Quarterly

- Review IAM policies
- Update dependencies
- Performance optimization
- Cost optimization review

### Backup Strategy

```powershell
# RDS automated backups (7-day retention by default)
# DynamoDB point-in-time recovery enabled
# S3 versioning enabled

# Manual backup
aws rds create-db-snapshot `
  --db-instance-identifier rca-pipeline-prod-db `
  --db-snapshot-identifier rca-pipeline-manual-$(Get-Date -Format 'yyyyMMdd')
```

## CI/CD Integration

### GitHub Actions

See `.github/workflows/terraform-deploy.yml` for automated deployments.

### Manual Promotion

```powershell
# Deploy to staging
terraform workspace select staging
terraform apply

# Test
./run-integration-tests.ps1

# Deploy to production
terraform workspace select prod
terraform apply
```

## Cost Estimation

### Monthly Costs (Production)

| Service | Cost |
|---------|------|
| Lambda | $83 |
| Step Functions | $15 |
| ECS Fargate | $40 |
| RDS (t3.medium) | $73 |
| ElastiCache | $12 |
| S3 | $16 |
| DynamoDB | $25 |
| API Gateway | $4 |
| Data Transfer | $20 |
| **Total** | **~$288/month** |

### Cost Per Analysis

- **Free tier**: $0.10/analysis
- **Pro tier**: $0.50/analysis
- **Enterprise**: $2.00/analysis

## Support

For issues or questions:

1. Check [AWS Service Health Dashboard](https://status.aws.amazon.com/)
2. Review CloudWatch logs
3. Open an issue in the repository
4. Contact: faresdibany

## License

See main repository for licensing information.

---

**Last Updated**: December 25, 2024
**Terraform Version**: >= 1.5.0
**AWS Provider Version**: ~> 5.0
