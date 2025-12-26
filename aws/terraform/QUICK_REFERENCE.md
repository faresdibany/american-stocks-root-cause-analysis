# Terraform Quick Reference - RCA Pipeline

## Essential Commands

### Initialization
```powershell
terraform init                      # Initialize working directory
terraform init -upgrade            # Upgrade providers
terraform init -reconfigure        # Reconfigure backend
```

### Planning
```powershell
terraform plan                     # Show execution plan
terraform plan -out=tfplan        # Save plan to file
terraform plan -destroy           # Plan destruction
terraform plan -target=module.vpc # Plan specific module
```

### Applying
```powershell
terraform apply                    # Apply changes (interactive)
terraform apply tfplan            # Apply saved plan
terraform apply -auto-approve     # Apply without confirmation
terraform apply -target=module.lambda  # Apply specific module
```

### Destroying
```powershell
terraform destroy                  # Destroy all infrastructure
terraform destroy -target=module.ecs  # Destroy specific module
```

### State Management
```powershell
terraform state list              # List all resources
terraform state show aws_instance.example  # Show resource details
terraform state rm aws_instance.example    # Remove from state
terraform state pull > state.json # Export state
terraform state push state.json   # Import state
```

### Outputs
```powershell
terraform output                   # Show all outputs
terraform output api_gateway_url  # Show specific output
terraform output -json > out.json # Export as JSON
```

### Validation
```powershell
terraform validate                # Validate configuration
terraform fmt                     # Format files
terraform fmt -check              # Check formatting
```

### Workspaces
```powershell
terraform workspace list          # List workspaces
terraform workspace new dev       # Create workspace
terraform workspace select prod   # Switch workspace
terraform workspace delete dev    # Delete workspace
```

## File Structure

```
terraform/
├── main.tf              # Main configuration
├── variables.tf         # Input variables
├── outputs.tf           # Output values
├── terraform.tfvars     # Variable values (DO NOT COMMIT)
├── modules/             # Reusable modules
│   ├── vpc/
│   ├── lambda/
│   ├── rds/
│   └── ...
```

## Common Tasks

### Get API URL
```powershell
terraform output -raw api_gateway_url
```

### Get RDS Endpoint
```powershell
terraform output -raw rds_endpoint
```

### Get All Outputs as JSON
```powershell
terraform output -json | ConvertFrom-Json
```

### Update Lambda Code
```powershell
# 1. Build package
cd ../lambda
./build-packages.ps1

# 2. Update function
aws lambda update-function-code `
  --function-name rca-pipeline-prod-driver-analysis `
  --zip-file fileb://dist/driver-analysis.zip
```

### View Logs
```powershell
# Lambda
aws logs tail /aws/lambda/rca-pipeline-prod-driver-analysis --follow

# Step Functions
aws logs tail /aws/states/rca-pipeline-prod --follow
```

### Check Costs
```powershell
# Current month
aws ce get-cost-and-usage `
  --time-period Start=$(Get-Date -Format yyyy-MM-01),End=$(Get-Date -Format yyyy-MM-dd) `
  --granularity MONTHLY `
  --metrics UnblendedCost
```

## Resource Naming Convention

```
{name_prefix}-{resource_type}-{suffix}

Examples:
- rca-pipeline-prod-vpc
- rca-pipeline-prod-lambda-driver-analysis
- rca-pipeline-prod-db
- rca-pipeline-prod-artifacts-a1b2c3d4
```

## Important ARNs

### Get Lambda ARNs
```powershell
terraform output lambda_function_arns
```

### Get State Machine ARN
```powershell
terraform output state_machine_arn
```

### Get S3 Bucket ARN
```powershell
terraform output artifacts_bucket_arn
```

## Troubleshooting

### Fix State Lock
```powershell
# If state is locked
terraform force-unlock LOCK_ID
```

### Refresh State
```powershell
terraform refresh
```

### Import Existing Resource
```powershell
terraform import aws_instance.example i-1234567890abcdef0
```

### Taint Resource (Force Recreate)
```powershell
terraform taint aws_instance.example
terraform apply
```

### Debug
```powershell
$env:TF_LOG = "DEBUG"
terraform apply
```

## Environment Variables

```powershell
# AWS Credentials
$env:AWS_ACCESS_KEY_ID = "YOUR_KEY"
$env:AWS_SECRET_ACCESS_KEY = "YOUR_SECRET"
$env:AWS_DEFAULT_REGION = "us-east-1"

# Terraform
$env:TF_LOG = "DEBUG"           # Enable debug logging
$env:TF_LOG_PATH = "terraform.log"  # Log to file
$env:TF_VAR_environment = "prod"    # Set variable
```

## Cost Estimates

### Monthly Baseline
- **Development**: ~$150/month
- **Production**: ~$300/month

### Per Analysis
- **Free tier**: $0.10
- **Pro tier**: $0.50
- **Enterprise**: $2.00

## Security Checklist

- [ ] Secrets in Secrets Manager, not code
- [ ] S3 buckets encrypted
- [ ] RDS encryption enabled
- [ ] VPC endpoints for AWS services
- [ ] Security groups follow least privilege
- [ ] IAM roles have minimum permissions
- [ ] CloudTrail enabled
- [ ] VPC Flow Logs enabled
- [ ] MFA on root account
- [ ] Terraform state encrypted

## Pre-Flight Checklist

Before `terraform apply`:

- [ ] Review plan output
- [ ] Verify costs with `terraform plan`
- [ ] Backup current state
- [ ] Check AWS service limits
- [ ] Verify IAM permissions
- [ ] Test in dev environment first
- [ ] Have rollback plan ready
- [ ] Notify team of changes

## Emergency Contacts

- **AWS Support**: https://console.aws.amazon.com/support
- **Service Health**: https://status.aws.amazon.com
- **Terraform Docs**: https://registry.terraform.io/providers/hashicorp/aws/latest/docs

## Quick Fixes

### Lambda Timeout in VPC
```hcl
# Increase timeout
timeout = 900  # 15 minutes
```

### RDS Connection Issues
```powershell
# Check security group
aws ec2 describe-security-groups --group-ids sg-xxxxx
```

### High NAT Gateway Costs
```hcl
# Use VPC endpoints instead
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.us-east-1.s3"
}
```

### S3 Lifecycle Not Working
```powershell
# Verify lifecycle rules
aws s3api get-bucket-lifecycle-configuration --bucket BUCKET_NAME
```

## Useful AWS CLI Commands

```powershell
# List all Lambda functions
aws lambda list-functions --query 'Functions[*].[FunctionName,Runtime,MemorySize]' --output table

# List all RDS instances
aws rds describe-db-instances --query 'DBInstances[*].[DBInstanceIdentifier,DBInstanceStatus]' --output table

# Check Step Functions executions
aws stepfunctions list-executions --state-machine-arn ARN

# S3 bucket size
aws s3 ls s3://BUCKET_NAME --recursive --summarize | grep "Total Size"

# DynamoDB table info
aws dynamodb describe-table --table-name TABLE_NAME
```

## Version Info

- **Terraform**: >= 1.5.0
- **AWS Provider**: ~> 5.0
- **Python Runtime**: 3.11
- **PostgreSQL**: 15.4
- **Redis**: 7.0

---

**Quick Start**: `terraform init && terraform plan && terraform apply`
**Emergency Stop**: `terraform destroy` (⚠️ deletes everything!)
