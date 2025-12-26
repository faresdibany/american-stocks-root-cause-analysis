# Terraform Configuration for AWS Account: 040604762405
# Username: feldiban
# Generated: December 25, 2025

# General Configuration
aws_region  = "us-east-1"
environment = "prod"
project_name = "rca-pipeline"

# Notification Configuration
notification_emails = ["fares@yourdomain.com"]  # UPDATE THIS with your actual email

# VPC Configuration
vpc_cidr = "10.0.0.0/16"
availability_zones = [
  "us-east-1a",
  "us-east-1b",
  "us-east-1c"
]

# Enable high availability (set to false for cost savings during testing)
enable_multi_az = true

# RDS Configuration
rds_instance_class    = "db.t3.medium"
rds_allocated_storage = 100
rds_multi_az          = true
rds_database_name     = "rcapipeline"
rds_master_username   = "rcaadmin"

# ElastiCache Configuration
redis_node_type  = "cache.t3.micro"
redis_num_nodes  = 1

# Lambda Configuration
lambda_functions = {
  "pipeline-trigger" = {
    memory_size = 512
    timeout     = 30
    handler     = "handler.trigger_handler"
    runtime     = "python3.11"
  }
  "driver-analysis" = {
    memory_size = 2048
    timeout     = 300
    handler     = "handler.driver_analysis_handler"
    runtime     = "python3.11"
  }
  "quant-news-analysis" = {
    memory_size = 1024
    timeout     = 300
    handler     = "handler.quant_news_handler"
    runtime     = "python3.11"
  }
  "social-sentiment" = {
    memory_size = 1024
    timeout     = 300
    handler     = "handler.social_sentiment_handler"
    runtime     = "python3.11"
  }
  "merge-consolidate" = {
    memory_size = 1024
    timeout     = 180
    handler     = "handler.merge_handler"
    runtime     = "python3.11"
  }
  "generate-reports" = {
    memory_size = 512
    timeout     = 120
    handler     = "handler.reports_handler"
    runtime     = "python3.11"
  }
  "authorizer" = {
    memory_size = 256
    timeout     = 10
    handler     = "handler.auth_handler"
    runtime     = "python3.11"
  }
  "job-status" = {
    memory_size = 256
    timeout     = 10
    handler     = "handler.status_handler"
    runtime     = "python3.11"
  }
}

# ECS Configuration
ecs_task_definitions = {
  "advanced-quant" = {
    cpu    = 2048
    memory = 4096
    image  = "your-ecr-repo/advanced-quant:latest"  # UPDATE THIS
  }
  "nlg-generator" = {
    cpu    = 1024
    memory = 2048
    image  = "your-ecr-repo/nlg-generator:latest"  # UPDATE THIS
  }
}

# API Gateway Configuration
api_throttle_rate_limit  = 1000
api_throttle_burst_limit = 2000

# CloudWatch Configuration
cloudwatch_log_retention_days = 30

# Step Functions Configuration
step_functions_max_concurrent_executions = 10

# DynamoDB Configuration
dynamodb_billing_mode = "PAY_PER_REQUEST"  # or "PROVISIONED" for predictable costs

# S3 Configuration
s3_lifecycle_glacier_days = 90
s3_lifecycle_expiration_days = 365

# Cognito Configuration
cognito_mfa_configuration = "OPTIONAL"
cognito_password_minimum_length = 12

# Security Configuration
enable_waf = true
enable_encryption = true

# Monitoring Configuration
enable_detailed_monitoring = true
enable_xray_tracing = false  # Set to true for detailed tracing (additional cost)

# Cost Optimization
enable_timestream = false  # Set to true if you need time-series data storage

# Backup Configuration
enable_automated_backups = true
backup_retention_days = 7

# Tags
tags = {
  Project     = "RCA Pipeline"
  Environment = "Production"
  ManagedBy   = "Terraform"
  Owner       = "feldiban"
  Account     = "040604762405"
}
