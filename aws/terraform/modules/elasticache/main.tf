# ElastiCache Redis Module
variable "name_prefix" {}
variable "vpc_id" {}
variable "subnet_ids" {}
variable "security_group_ids" {}
variable "node_type" {}
variable "num_cache_nodes" {}
variable "tags" { default = {} }

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.name_prefix}-redis-subnet"
  subnet_ids = var.subnet_ids
  tags       = merge(var.tags, { Name = "${var.name_prefix}-redis-subnet" })
}

resource "aws_elasticache_cluster" "main" {
  cluster_id           = "${var.name_prefix}-redis"
  engine               = "redis"
  engine_version       = "7.0"
  node_type            = var.node_type
  num_cache_nodes      = var.num_cache_nodes
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = var.security_group_ids
  
  snapshot_retention_limit = 5
  snapshot_window          = "03:00-05:00"
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-redis" })
}

output "primary_endpoint" {
  value = aws_elasticache_cluster.main.cache_nodes[0].address
}

output "port" {
  value = aws_elasticache_cluster.main.port
}
