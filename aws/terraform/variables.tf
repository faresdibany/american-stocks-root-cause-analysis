# Variable definitions for RCA Pipeline Infrastructure

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones for multi-AZ deployment"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

# S3 Configuration
variable "s3_lifecycle_rules" {
  description = "S3 lifecycle rules for artifact management"
  type = list(object({
    id          = string
    enabled     = bool
    transitions = list(object({
      days          = number
      storage_class = string
    }))
    expiration = object({
      days = number
    })
  }))
  default = [
    {
      id      = "archive-old-reports"
      enabled = true
      transitions = [
        {
          days          = 30
          storage_class = "GLACIER"
        }
      ]
      expiration = {
        days = 365
      }
    }
  ]
}

# RDS Configuration
variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100
}

variable "rds_multi_az" {
  description = "Enable multi-AZ for RDS"
  type        = bool
  default     = true
}

variable "rds_database_name" {
  description = "Name of the PostgreSQL database"
  type        = string
  default     = "rcapipeline"
}

variable "rds_master_username" {
  description = "Master username for RDS"
  type        = string
  default     = "rcaadmin"
}

# ElastiCache Configuration
variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 1
}

# Lambda Configuration
variable "lambda_functions" {
  description = "Lambda function configurations"
  type = map(object({
    memory_size = number
    timeout     = number
    handler     = string
    runtime     = string
  }))
  default = {
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
      timeout     = 60
      handler     = "handler.merge_handler"
      runtime     = "python3.11"
    }
    "generate-reports" = {
      memory_size = 512
      timeout     = 60
      handler     = "handler.report_handler"
      runtime     = "python3.11"
    }
    "authorizer" = {
      memory_size = 256
      timeout     = 10
      handler     = "handler.authorizer_handler"
      runtime     = "python3.11"
    }
    "job-status" = {
      memory_size = 256
      timeout     = 10
      handler     = "handler.status_handler"
      runtime     = "python3.11"
    }
  }
}

# ECS Configuration
variable "ecs_task_definitions" {
  description = "ECS task definitions for heavy workloads"
  type = map(object({
    cpu    = number
    memory = number
    image  = string
  }))
  default = {
    "advanced-quant" = {
      cpu    = 2048
      memory = 4096
      image  = "rca-pipeline/advanced-quant:latest"
    }
    "nlg-generator" = {
      cpu    = 4096
      memory = 8192
      image  = "rca-pipeline/nlg-generator:latest"
    }
  }
}

# API Gateway Configuration
variable "api_throttle_burst_limit" {
  description = "API Gateway throttle burst limit"
  type        = number
  default     = 100
}

variable "api_throttle_rate_limit" {
  description = "API Gateway throttle rate limit (requests per second)"
  type        = number
  default     = 50
}

# Cognito Configuration
variable "cognito_password_policy" {
  description = "Password policy for Cognito"
  type = object({
    minimum_length    = number
    require_lowercase = bool
    require_numbers   = bool
    require_symbols   = bool
    require_uppercase = bool
  })
  default = {
    minimum_length    = 12
    require_lowercase = true
    require_numbers   = true
    require_symbols   = true
    require_uppercase = true
  }
}

variable "cognito_email_configuration" {
  description = "Email configuration for Cognito"
  type = object({
    email_sending_account = string
    source_arn            = string
  })
  default = {
    email_sending_account = "COGNITO_DEFAULT"
    source_arn            = ""
  }
}

# SQS Configuration
variable "sqs_visibility_timeout" {
  description = "SQS visibility timeout in seconds"
  type        = number
  default     = 900
}

variable "sqs_max_receive_count" {
  description = "Maximum receive count before moving to DLQ"
  type        = number
  default     = 3
}

# CloudWatch Configuration
variable "cloudwatch_alarm_thresholds" {
  description = "CloudWatch alarm thresholds"
  type = map(object({
    metric              = string
    threshold           = number
    comparison_operator = string
    evaluation_periods  = number
  }))
  default = {
    "lambda-errors" = {
      metric              = "Errors"
      threshold           = 10
      comparison_operator = "GreaterThanThreshold"
      evaluation_periods  = 1
    }
    "lambda-duration" = {
      metric              = "Duration"
      threshold           = 30000
      comparison_operator = "GreaterThanThreshold"
      evaluation_periods  = 2
    }
    "step-function-failed" = {
      metric              = "ExecutionsFailed"
      threshold           = 5
      comparison_operator = "GreaterThanThreshold"
      evaluation_periods  = 1
    }
  }
}

# EventBridge Configuration
variable "eventbridge_scheduled_rules" {
  description = "EventBridge scheduled rules"
  type = map(object({
    schedule_expression = string
    input               = string
    enabled             = bool
  }))
  default = {
    "daily-portfolio-check" = {
      schedule_expression = "cron(0 9 ? * MON-FRI *)"
      input = jsonencode({
        tickers = "AAPL,MSFT,NVDA,AMZN,GOOGL"
        period  = "1mo"
      })
      enabled = false
    }
  }
}

# WAF Configuration
variable "waf_rate_limit" {
  description = "WAF rate limit per IP (requests per 5 minutes)"
  type        = number
  default     = 2000
}

# TimeStream Configuration
variable "enable_timestream" {
  description = "Enable TimeStream for time-series data"
  type        = bool
  default     = false
}

variable "timestream_retention_period" {
  description = "TimeStream retention period in days"
  type        = number
  default     = 90
}

# Notification Configuration
variable "notification_emails" {
  description = "Email addresses for notifications"
  type        = list(string)
  default     = []
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}
