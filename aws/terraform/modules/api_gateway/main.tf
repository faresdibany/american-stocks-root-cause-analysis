# API Gateway Module - REST API configuration

variable "name_prefix" {}
variable "stage_name" {}
variable "lambda_invoke_arn" {}
variable "authorizer_lambda_arn" {}
variable "authorizer_lambda_invoke_arn" {}
variable "cognito_user_pool_arn" {}
variable "throttle_burst_limit" {}
variable "throttle_rate_limit" {}
variable "tags" { default = {} }

resource "aws_api_gateway_rest_api" "main" {
  name        = "${var.name_prefix}-api"
  description = "RCA Pipeline API"
  
  endpoint_configuration {
    types = ["REGIONAL"]
  }
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-api" })
}

resource "aws_api_gateway_authorizer" "cognito" {
  name          = "${var.name_prefix}-cognito-authorizer"
  rest_api_id   = aws_api_gateway_rest_api.main.id
  type          = "COGNITO_USER_POOLS"
  provider_arns = [var.cognito_user_pool_arn]
}

# /analyze endpoint
resource "aws_api_gateway_resource" "analyze" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_rest_api.main.root_resource_id
  path_part   = "analyze"
}

resource "aws_api_gateway_method" "analyze_post" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_resource.analyze.id
  http_method   = "POST"
  authorization = "COGNITO_USER_POOLS"
  authorizer_id = aws_api_gateway_authorizer.cognito.id
}

resource "aws_api_gateway_integration" "analyze" {
  rest_api_id             = aws_api_gateway_rest_api.main.id
  resource_id             = aws_api_gateway_resource.analyze.id
  http_method             = aws_api_gateway_method.analyze_post.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = var.lambda_invoke_arn
}

# Deployment
resource "aws_api_gateway_deployment" "main" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  
  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.analyze.id,
      aws_api_gateway_method.analyze_post.id,
      aws_api_gateway_integration.analyze.id,
    ]))
  }
  
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_api_gateway_stage" "main" {
  deployment_id = aws_api_gateway_deployment.main.id
  rest_api_id   = aws_api_gateway_rest_api.main.id
  stage_name    = var.stage_name
  
  xray_tracing_enabled = true
  
  tags = merge(var.tags, { Name = "${var.name_prefix}-${var.stage_name}" })
}

resource "aws_api_gateway_method_settings" "main" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  stage_name  = aws_api_gateway_stage.main.stage_name
  method_path = "*/*"
  
  settings {
    throttling_burst_limit = var.throttle_burst_limit
    throttling_rate_limit  = var.throttle_rate_limit
    logging_level          = "INFO"
    data_trace_enabled     = true
    metrics_enabled        = true
  }
}

# Lambda permission
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = split(":", var.lambda_invoke_arn)[6]
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.main.execution_arn}/*/*"
}

output "api_id" {
  value = aws_api_gateway_rest_api.main.id
}

output "api_arn" {
  value = aws_api_gateway_rest_api.main.arn
}

output "api_url" {
  value = aws_api_gateway_stage.main.invoke_url
}

output "stage_name" {
  value = aws_api_gateway_stage.main.stage_name
}
