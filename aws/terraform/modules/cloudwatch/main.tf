# CloudWatch Module - Monitoring and alarms
variable "name_prefix" {}
variable "lambda_function_names" { type = list(string) }
variable "step_function_arn" {}
variable "sns_alarm_topic_arn" {}
variable "alarm_thresholds" {}
variable "tags" { default = {} }

resource "aws_cloudwatch_log_group" "main" {
  name              = "/rca-pipeline/${var.name_prefix}"
  retention_in_days = 30
  tags              = merge(var.tags, { Name = "${var.name_prefix}-logs" })
}

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  for_each = toset(var.lambda_function_names)
  
  alarm_name          = "${var.name_prefix}-${each.value}-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "Lambda function ${each.value} error rate"
  alarm_actions       = [var.sns_alarm_topic_arn]
  
  dimensions = {
    FunctionName = "${var.name_prefix}-${each.value}"
  }
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-${each.value}-alarm" })
}

output "log_group_name" {
  value = aws_cloudwatch_log_group.main.name
}
