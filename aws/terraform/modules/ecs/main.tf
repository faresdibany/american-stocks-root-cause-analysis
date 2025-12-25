# ECS Fargate Module
variable "name_prefix" {}
variable "vpc_id" {}
variable "subnet_ids" {}
variable "security_group_ids" {}
variable "execution_role_arn" {}
variable "task_role_arn" {}
variable "task_definitions" {}
variable "environment_variables" {}
variable "tags" { default = {} }

resource "aws_ecs_cluster" "main" {
  name = "${var.name_prefix}-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-cluster" })
}

resource "aws_cloudwatch_log_group" "ecs" {
  for_each = var.task_definitions
  
  name              = "/ecs/${var.name_prefix}/${each.key}"
  retention_in_days = 30
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-${each.key}-logs" })
}

resource "aws_ecs_task_definition" "tasks" {
  for_each = var.task_definitions
  
  family                   = "${var.name_prefix}-${each.key}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = each.value.cpu
  memory                   = each.value.memory
  execution_role_arn       = var.execution_role_arn
  task_role_arn            = var.task_role_arn
  
  container_definitions = jsonencode([{
    name  = each.key
    image = each.value.image
    
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs[each.key].name
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = "ecs"
      }
    }
    
    environment = [
      for k, v in var.environment_variables : { name = k, value = v }
    ]
  }])
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-${each.key}" })
}

data "aws_region" "current" {}

output "cluster_arn" {
  value = aws_ecs_cluster.main.arn
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "task_definition_arns" {
  value = { for k, v in aws_ecs_task_definition.tasks : k => v.arn }
}
