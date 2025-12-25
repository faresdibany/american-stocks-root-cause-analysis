# Lambda Module - Function definitions

variable "name_prefix" {}
variable "lambda_role_arn" {}
variable "vpc_config" {}
variable "environment_variables" {}
variable "functions" {}
variable "tags" { default = {} }

# Lambda functions will be created from deployment packages
# See ../lambda/README.md for building packages

resource "aws_lambda_function" "functions" {
  for_each = var.functions

  function_name = "${var.name_prefix}-${each.key}"
  role          = var.lambda_role_arn
  handler       = each.value.handler
  runtime       = each.value.runtime
  memory_size   = each.value.memory_size
  timeout       = each.value.timeout

  # Placeholder - replace with actual S3 bucket/key after building packages
  filename         = "${path.module}/../../lambda/dist/${each.key}.zip"
  source_code_hash = fileexists("${path.module}/../../lambda/dist/${each.key}.zip") ? filebase64sha256("${path.module}/../../lambda/dist/${each.key}.zip") : null

  vpc_config {
    subnet_ids         = var.vpc_config.subnet_ids
    security_group_ids = var.vpc_config.security_group_ids
  }

  environment {
    variables = var.environment_variables
  }

  tracing_config {
    mode = "Active"
  }

  tags = merge(var.tags, { Name = "${var.name_prefix}-${each.key}" })

  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
}

output "function_arns" {
  value = { for k, v in aws_lambda_function.functions : k => v.arn }
}

output "function_names" {
  value = { for k, v in aws_lambda_function.functions : k => v.function_name }
}

output "function_invoke_arns" {
  value = { for k, v in aws_lambda_function.functions : k => v.invoke_arn }
}
