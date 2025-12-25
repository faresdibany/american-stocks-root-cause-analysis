# Secrets Manager Module
variable "name_prefix" {}
variable "rds_credentials" {}
variable "tags" { default = {} }

resource "aws_secretsmanager_secret" "rds" {
  name        = "${var.name_prefix}/rds-credentials"
  description = "RDS PostgreSQL credentials"
  tags        = merge(var.tags, { Name = "${var.name_prefix}-rds-secret" })
}

resource "aws_secretsmanager_secret_version" "rds" {
  secret_id = aws_secretsmanager_secret.rds.id
  secret_string = jsonencode(var.rds_credentials)
}

output "rds_secret_arn" {
  value = aws_secretsmanager_secret.rds.arn
}
