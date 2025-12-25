# EventBridge Module - Scheduled jobs
variable "name_prefix" {}
variable "state_machine_arn" {}
variable "eventbridge_role_arn" {}
variable "scheduled_rules" {}
variable "tags" { default = {} }

resource "aws_cloudwatch_event_rule" "scheduled" {
  for_each = var.scheduled_rules
  
  name                = "${var.name_prefix}-${each.key}"
  description         = "Scheduled rule for ${each.key}"
  schedule_expression = each.value.schedule_expression
  is_enabled          = each.value.enabled
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-${each.key}" })
}

resource "aws_cloudwatch_event_target" "step_function" {
  for_each = var.scheduled_rules
  
  rule      = aws_cloudwatch_event_rule.scheduled[each.key].name
  target_id = "StepFunctionsTarget"
  arn       = var.state_machine_arn
  role_arn  = var.eventbridge_role_arn
  input     = each.value.input
}

output "rule_arns" {
  value = { for k, v in aws_cloudwatch_event_rule.scheduled : k => v.arn }
}
