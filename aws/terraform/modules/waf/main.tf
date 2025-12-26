# WAF Module - Web Application Firewall
variable "name_prefix" {}
variable "api_gateway_arn" {}
variable "rate_limit" {}
variable "tags" { default = {} }

resource "aws_wafv2_web_acl" "main" {
  name  = "${var.name_prefix}-waf"
  scope = "REGIONAL"
  
  default_action {
    allow {}
  }
  
  rule {
    name     = "RateLimitRule"
    priority = 1
    
    action {
      block {}
    }
    
    statement {
      rate_based_statement {
        limit              = var.rate_limit
        aggregate_key_type = "IP"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.name_prefix}-rate-limit"
      sampled_requests_enabled   = true
    }
  }
  
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.name_prefix}-waf"
    sampled_requests_enabled   = true
  }
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-waf" })
}

resource "aws_wafv2_web_acl_association" "api_gateway" {
  resource_arn = var.api_gateway_arn
  web_acl_arn  = aws_wafv2_web_acl.main.arn
}

output "web_acl_id" {
  value = aws_wafv2_web_acl.main.id
}
