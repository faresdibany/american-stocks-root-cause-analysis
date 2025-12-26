# RDS Module - PostgreSQL database

variable "name_prefix" {}
variable "vpc_id" {}
variable "subnet_ids" {}
variable "security_group_ids" {}
variable "instance_class" {}
variable "allocated_storage" {}
variable "multi_az" {}
variable "database_name" {}
variable "master_username" {}
variable "tags" { default = {} }

resource "random_password" "master" {
  length  = 32
  special = true
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.name_prefix}-db-subnet"
  subnet_ids = var.subnet_ids
  tags       = merge(var.tags, { Name = "${var.name_prefix}-db-subnet" })
}

resource "aws_db_instance" "main" {
  identifier     = "${var.name_prefix}-db"
  engine         = "postgres"
  engine_version = "15.4"
  
  instance_class    = var.instance_class
  allocated_storage = var.allocated_storage
  storage_type      = "gp3"
  storage_encrypted = true
  
  db_name  = var.database_name
  username = var.master_username
  password = random_password.master.result
  
  multi_az               = var.multi_az
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = var.security_group_ids
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  
  skip_final_snapshot       = false
  final_snapshot_identifier = "${var.name_prefix}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  
  deletion_protection = true
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-db" })
}

output "endpoint" {
  value = aws_db_instance.main.endpoint
}

output "master_password" {
  value     = random_password.master.result
  sensitive = true
}

output "arn" {
  value = aws_db_instance.main.arn
}
