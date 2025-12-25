# SNS Module - Notifications
variable "name_prefix" {}
variable "email_subscriptions" { type = list(string) }
variable "tags" { default = {} }

resource "aws_sns_topic" "notifications" {
  name = "${var.name_prefix}-notifications"
  tags = merge(var.tags, { Name = "${var.name_prefix}-notifications" })
}

resource "aws_sns_topic" "alarms" {
  name = "${var.name_prefix}-alarms"
  tags = merge(var.tags, { Name = "${var.name_prefix}-alarms" })
}

resource "aws_sns_topic_subscription" "email" {
  for_each  = toset(var.email_subscriptions)
  topic_arn = aws_sns_topic.notifications.arn
  protocol  = "email"
  endpoint  = each.value
}

output "notifications_topic_arn" {
  value = aws_sns_topic.notifications.arn
}

output "alarms_topic_arn" {
  value = aws_sns_topic.alarms.arn
}
