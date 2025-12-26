# DynamoDB Module - NoSQL storage for job tracking and caching

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Jobs Table - Track pipeline execution status
resource "aws_dynamodb_table" "jobs" {
  name           = "${var.name_prefix}-jobs"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "job_id"
  
  attribute {
    name = "job_id"
    type = "S"
  }

  attribute {
    name = "user_id"
    type = "S"
  }

  attribute {
    name = "created_at"
    type = "N"
  }

  attribute {
    name = "status"
    type = "S"
  }

  global_secondary_index {
    name            = "UserIdIndex"
    hash_key        = "user_id"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  global_secondary_index {
    name            = "StatusIndex"
    hash_key        = "status"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.name_prefix}-jobs"
    }
  )
}

# Cache Table - Store frequently accessed data
resource "aws_dynamodb_table" "cache" {
  name           = "${var.name_prefix}-cache"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "cache_key"
  
  attribute {
    name = "cache_key"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.name_prefix}-cache"
    }
  )
}

# API Usage Table - Track API calls for billing
resource "aws_dynamodb_table" "api_usage" {
  name           = "${var.name_prefix}-api-usage"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "usage_id"
  
  attribute {
    name = "usage_id"
    type = "S"
  }

  attribute {
    name = "user_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "N"
  }

  global_secondary_index {
    name            = "UserIdTimestampIndex"
    hash_key        = "user_id"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.name_prefix}-api-usage"
    }
  )
}

# Rate Limit Table - Track rate limits per user
resource "aws_dynamodb_table" "rate_limits" {
  name           = "${var.name_prefix}-rate-limits"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "identifier"
  
  attribute {
    name = "identifier"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.name_prefix}-rate-limits"
    }
  )
}

# Outputs
output "jobs_table_name" {
  description = "Name of the jobs table"
  value       = aws_dynamodb_table.jobs.name
}

output "jobs_table_arn" {
  description = "ARN of the jobs table"
  value       = aws_dynamodb_table.jobs.arn
}

output "cache_table_name" {
  description = "Name of the cache table"
  value       = aws_dynamodb_table.cache.name
}

output "cache_table_arn" {
  description = "ARN of the cache table"
  value       = aws_dynamodb_table.cache.arn
}

output "api_usage_table_name" {
  description = "Name of the API usage table"
  value       = aws_dynamodb_table.api_usage.name
}

output "api_usage_table_arn" {
  description = "ARN of the API usage table"
  value       = aws_dynamodb_table.api_usage.arn
}

output "rate_limits_table_name" {
  description = "Name of the rate limits table"
  value       = aws_dynamodb_table.rate_limits.name
}

output "rate_limits_table_arn" {
  description = "ARN of the rate limits table"
  value       = aws_dynamodb_table.rate_limits.arn
}

output "table_arns" {
  description = "Map of all table ARNs"
  value = {
    jobs        = aws_dynamodb_table.jobs.arn
    cache       = aws_dynamodb_table.cache.arn
    api_usage   = aws_dynamodb_table.api_usage.arn
    rate_limits = aws_dynamodb_table.rate_limits.arn
  }
}
