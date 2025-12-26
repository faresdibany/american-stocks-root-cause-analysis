# Main Terraform configuration for RCA Pipeline SaaS Infrastructure
# Architecture: Serverless-first hybrid with Lambda, Step Functions, and ECS Fargate

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Uncomment for remote state management
  # backend "s3" {
  #   bucket         = "rca-pipeline-terraform-state"
  #   key            = "prod/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "rca-pipeline-terraform-locks"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "RCA-Pipeline"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "faresdibany"
    }
  }
}

# Random suffix for globally unique names
resource "random_id" "suffix" {
  byte_length = 4
}

locals {
  name_prefix = "rca-pipeline-${var.environment}"
  common_tags = {
    Application = "Stock Root Cause Analysis"
    Repository  = "american-stocks-root-cause-analysis"
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# VPC for ECS Fargate and RDS
module "vpc" {
  source = "./modules/vpc"

  name_prefix = local.name_prefix
  cidr_block  = var.vpc_cidr
  
  availability_zones = var.availability_zones
  
  tags = local.common_tags
}

# S3 for artifact storage
module "s3" {
  source = "./modules/s3"

  name_prefix = local.name_prefix
  suffix      = random_id.suffix.hex
  
  lifecycle_rules = var.s3_lifecycle_rules
  
  tags = local.common_tags
}

# DynamoDB for job tracking and caching
module "dynamodb" {
  source = "./modules/dynamodb"

  name_prefix = local.name_prefix
  
  tags = local.common_tags
}

# RDS PostgreSQL for structured data
module "rds" {
  source = "./modules/rds"

  name_prefix        = local.name_prefix
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnet_ids
  security_group_ids = [module.vpc.rds_security_group_id]
  
  instance_class     = var.rds_instance_class
  allocated_storage  = var.rds_allocated_storage
  multi_az           = var.rds_multi_az
  
  database_name      = var.rds_database_name
  master_username    = var.rds_master_username
  
  tags = local.common_tags
}

# ElastiCache Redis for rate limiting and caching
module "elasticache" {
  source = "./modules/elasticache"

  name_prefix        = local.name_prefix
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnet_ids
  security_group_ids = [module.vpc.redis_security_group_id]
  
  node_type          = var.redis_node_type
  num_cache_nodes    = var.redis_num_nodes
  
  tags = local.common_tags
}

# IAM roles and policies
module "iam" {
  source = "./modules/iam"

  name_prefix     = local.name_prefix
  account_id      = data.aws_caller_identity.current.account_id
  region          = data.aws_region.current.name
  
  s3_bucket_arn   = module.s3.artifacts_bucket_arn
  dynamodb_tables = module.dynamodb.table_arns
  
  tags = local.common_tags
}

# Lambda functions
module "lambda" {
  source = "./modules/lambda"

  name_prefix           = local.name_prefix
  lambda_role_arn       = module.iam.lambda_execution_role_arn
  
  vpc_config = {
    subnet_ids         = module.vpc.private_subnet_ids
    security_group_ids = [module.vpc.lambda_security_group_id]
  }
  
  environment_variables = {
    ARTIFACTS_BUCKET      = module.s3.artifacts_bucket_name
    JOBS_TABLE            = module.dynamodb.jobs_table_name
    CACHE_TABLE           = module.dynamodb.cache_table_name
    RDS_ENDPOINT          = module.rds.endpoint
    RDS_DATABASE          = var.rds_database_name
    REDIS_ENDPOINT        = module.elasticache.primary_endpoint
    SECRETS_ARN           = module.secrets.rds_secret_arn
    STATE_MACHINE_ARN     = module.step_functions.state_machine_arn
  }
  
  functions = var.lambda_functions
  
  tags = local.common_tags
  
  depends_on = [module.iam]
}

# Step Functions state machine
module "step_functions" {
  source = "./modules/step_functions"

  name_prefix           = local.name_prefix
  execution_role_arn    = module.iam.step_functions_role_arn
  
  lambda_function_arns = module.lambda.function_arns
  ecs_cluster_arn      = module.ecs.cluster_arn
  ecs_task_definitions = module.ecs.task_definition_arns
  
  sns_topic_arn        = module.sns.notifications_topic_arn
  
  tags = local.common_tags
  
  depends_on = [module.lambda, module.ecs]
}

# ECS Fargate for heavy workloads
module "ecs" {
  source = "./modules/ecs"

  name_prefix        = local.name_prefix
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnet_ids
  security_group_ids = [module.vpc.ecs_security_group_id]
  
  execution_role_arn = module.iam.ecs_execution_role_arn
  task_role_arn      = module.iam.ecs_task_role_arn
  
  task_definitions = var.ecs_task_definitions
  
  environment_variables = {
    ARTIFACTS_BUCKET = module.s3.artifacts_bucket_name
    RDS_ENDPOINT     = module.rds.endpoint
    RDS_DATABASE     = var.rds_database_name
    SECRETS_ARN      = module.secrets.rds_secret_arn
  }
  
  tags = local.common_tags
}

# API Gateway
module "api_gateway" {
  source = "./modules/api_gateway"

  name_prefix                = local.name_prefix
  stage_name                 = var.environment
  
  lambda_invoke_arn          = module.lambda.function_arns["pipeline-trigger"]
  authorizer_lambda_arn      = module.lambda.function_arns["authorizer"]
  authorizer_lambda_invoke_arn = module.lambda.function_invoke_arns["authorizer"]
  
  cognito_user_pool_arn      = module.cognito.user_pool_arn
  
  throttle_burst_limit       = var.api_throttle_burst_limit
  throttle_rate_limit        = var.api_throttle_rate_limit
  
  tags = local.common_tags
}

# Cognito for authentication
module "cognito" {
  source = "./modules/cognito"

  name_prefix         = local.name_prefix
  
  password_policy     = var.cognito_password_policy
  email_configuration = var.cognito_email_configuration
  
  tags = local.common_tags
}

# Secrets Manager for sensitive data
module "secrets" {
  source = "./modules/secrets"

  name_prefix = local.name_prefix
  
  rds_credentials = {
    username = var.rds_master_username
    password = module.rds.master_password
    endpoint = module.rds.endpoint
    database = var.rds_database_name
  }
  
  tags = local.common_tags
}

# SNS for notifications
module "sns" {
  source = "./modules/sns"

  name_prefix = local.name_prefix
  
  email_subscriptions = var.notification_emails
  
  tags = local.common_tags
}

# SQS for async job processing
module "sqs" {
  source = "./modules/sqs"

  name_prefix = local.name_prefix
  
  visibility_timeout = var.sqs_visibility_timeout
  max_receive_count  = var.sqs_max_receive_count
  
  tags = local.common_tags
}

# CloudWatch for monitoring
module "cloudwatch" {
  source = "./modules/cloudwatch"

  name_prefix = local.name_prefix
  
  lambda_function_names     = keys(module.lambda.function_arns)
  step_function_arn         = module.step_functions.state_machine_arn
  
  sns_alarm_topic_arn       = module.sns.alarms_topic_arn
  
  alarm_thresholds          = var.cloudwatch_alarm_thresholds
  
  tags = local.common_tags
}

# EventBridge for scheduled runs
module "eventbridge" {
  source = "./modules/eventbridge"

  name_prefix           = local.name_prefix
  
  state_machine_arn     = module.step_functions.state_machine_arn
  eventbridge_role_arn  = module.iam.eventbridge_role_arn
  
  scheduled_rules       = var.eventbridge_scheduled_rules
  
  tags = local.common_tags
}

# WAF for API protection
module "waf" {
  source = "./modules/waf"

  name_prefix     = local.name_prefix
  api_gateway_arn = module.api_gateway.api_arn
  
  rate_limit      = var.waf_rate_limit
  
  tags = local.common_tags
}

# TimeStream for time-series data (optional)
module "timestream" {
  source = "./modules/timestream"
  count  = var.enable_timestream ? 1 : 0

  name_prefix = local.name_prefix
  
  retention_period = var.timestream_retention_period
  
  tags = local.common_tags
}
