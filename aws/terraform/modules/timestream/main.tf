# TimeStream Module - Time-series database (optional)
variable "name_prefix" {}
variable "retention_period" {}
variable "tags" { default = {} }

resource "aws_timestreamwrite_database" "main" {
  database_name = "${var.name_prefix}-timeseries"
  tags          = merge(var.tags, { Name = "${var.name_prefix}-timeseries" })
}

resource "aws_timestreamwrite_table" "prices" {
  database_name = aws_timestreamwrite_database.main.database_name
  table_name    = "stock_prices"
  
  retention_properties {
    memory_store_retention_period_in_hours  = 24
    magnetic_store_retention_period_in_days = var.retention_period
  }
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-prices" })
}

resource "aws_timestreamwrite_table" "metrics" {
  database_name = aws_timestreamwrite_database.main.database_name
  table_name    = "pipeline_metrics"
  
  retention_properties {
    memory_store_retention_period_in_hours  = 24
    magnetic_store_retention_period_in_days = var.retention_period
  }
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-metrics" })
}

output "database_name" {
  value = aws_timestreamwrite_database.main.database_name
}

output "table_name" {
  value = aws_timestreamwrite_table.prices.table_name
}
