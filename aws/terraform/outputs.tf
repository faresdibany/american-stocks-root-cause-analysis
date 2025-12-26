# Output values from the infrastructure

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = module.vpc.private_subnet_ids
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = module.vpc.public_subnet_ids
}

# S3 Outputs
output "artifacts_bucket_name" {
  description = "Name of the S3 artifacts bucket"
  value       = module.s3.artifacts_bucket_name
}

output "artifacts_bucket_arn" {
  description = "ARN of the S3 artifacts bucket"
  value       = module.s3.artifacts_bucket_arn
}

# DynamoDB Outputs
output "jobs_table_name" {
  description = "Name of the DynamoDB jobs table"
  value       = module.dynamodb.jobs_table_name
}

output "cache_table_name" {
  description = "Name of the DynamoDB cache table"
  value       = module.dynamodb.cache_table_name
}

# RDS Outputs
output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "rds_database_name" {
  description = "Name of the RDS database"
  value       = var.rds_database_name
}

# ElastiCache Outputs
output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = module.elasticache.primary_endpoint
  sensitive   = true
}

# Lambda Outputs
output "lambda_function_arns" {
  description = "ARNs of Lambda functions"
  value       = module.lambda.function_arns
}

output "lambda_function_names" {
  description = "Names of Lambda functions"
  value       = module.lambda.function_names
}

# Step Functions Outputs
output "state_machine_arn" {
  description = "ARN of the Step Functions state machine"
  value       = module.step_functions.state_machine_arn
}

output "state_machine_name" {
  description = "Name of the Step Functions state machine"
  value       = module.step_functions.state_machine_name
}

# ECS Outputs
output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = module.ecs.cluster_arn
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = module.ecs.cluster_name
}

# API Gateway Outputs
output "api_gateway_url" {
  description = "URL of the API Gateway"
  value       = module.api_gateway.api_url
}

output "api_gateway_id" {
  description = "ID of the API Gateway"
  value       = module.api_gateway.api_id
}

output "api_gateway_stage" {
  description = "Stage name of the API Gateway"
  value       = module.api_gateway.stage_name
}

# Cognito Outputs
output "cognito_user_pool_id" {
  description = "ID of the Cognito User Pool"
  value       = module.cognito.user_pool_id
}

output "cognito_user_pool_arn" {
  description = "ARN of the Cognito User Pool"
  value       = module.cognito.user_pool_arn
}

output "cognito_client_id" {
  description = "ID of the Cognito User Pool Client"
  value       = module.cognito.client_id
  sensitive   = true
}

# Secrets Manager Outputs
output "rds_secret_arn" {
  description = "ARN of the RDS credentials secret"
  value       = module.secrets.rds_secret_arn
  sensitive   = true
}

# SNS Outputs
output "notifications_topic_arn" {
  description = "ARN of the SNS notifications topic"
  value       = module.sns.notifications_topic_arn
}

output "alarms_topic_arn" {
  description = "ARN of the SNS alarms topic"
  value       = module.sns.alarms_topic_arn
}

# SQS Outputs
output "job_queue_url" {
  description = "URL of the SQS job queue"
  value       = module.sqs.job_queue_url
}

output "dead_letter_queue_url" {
  description = "URL of the SQS dead letter queue"
  value       = module.sqs.dead_letter_queue_url
}

# CloudWatch Outputs
output "cloudwatch_log_group" {
  description = "Name of the CloudWatch log group"
  value       = module.cloudwatch.log_group_name
}

# IAM Outputs
output "lambda_execution_role_arn" {
  description = "ARN of the Lambda execution role"
  value       = module.iam.lambda_execution_role_arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role"
  value       = module.iam.ecs_task_role_arn
}

# TimeStream Outputs (conditional)
output "timestream_database_name" {
  description = "Name of the TimeStream database"
  value       = var.enable_timestream ? module.timestream[0].database_name : null
}

output "timestream_table_name" {
  description = "Name of the TimeStream table"
  value       = var.enable_timestream ? module.timestream[0].table_name : null
}

# Summary Output
output "deployment_summary" {
  description = "Summary of the deployment"
  value = {
    environment        = var.environment
    region             = var.aws_region
    api_url            = module.api_gateway.api_url
    artifacts_bucket   = module.s3.artifacts_bucket_name
    state_machine      = module.step_functions.state_machine_name
    user_pool_id       = module.cognito.user_pool_id
  }
}

# Connection Strings (for application configuration)
output "connection_info" {
  description = "Connection information for applications"
  value = {
    api_gateway_url    = module.api_gateway.api_url
    cognito_user_pool  = module.cognito.user_pool_id
    cognito_client_id  = module.cognito.client_id
    artifacts_bucket   = module.s3.artifacts_bucket_name
    jobs_table         = module.dynamodb.jobs_table_name
    cache_table        = module.dynamodb.cache_table_name
  }
  sensitive = true
}
