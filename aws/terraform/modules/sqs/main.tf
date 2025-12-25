# SQS Module - Job queue
variable "name_prefix" {}
variable "visibility_timeout" {}
variable "max_receive_count" {}
variable "tags" { default = {} }

resource "aws_sqs_queue" "dead_letter" {
  name = "${var.name_prefix}-dlq"
  message_retention_seconds = 1209600  # 14 days
  tags = merge(var.tags, { Name = "${var.name_prefix}-dlq" })
}

resource "aws_sqs_queue" "jobs" {
  name                      = "${var.name_prefix}-jobs"
  visibility_timeout_seconds = var.visibility_timeout
  message_retention_seconds = 86400  # 1 day
  
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dead_letter.arn
    maxReceiveCount     = var.max_receive_count
  })
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-jobs" })
}

output "job_queue_url" {
  value = aws_sqs_queue.jobs.url
}

output "dead_letter_queue_url" {
  value = aws_sqs_queue.dead_letter.url
}
