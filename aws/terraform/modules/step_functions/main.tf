# Step Functions Module - Pipeline orchestration

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "execution_role_arn" {
  description = "ARN of the execution role"
  type        = string
}

variable "lambda_function_arns" {
  description = "Map of Lambda function ARNs"
  type        = map(string)
}

variable "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  type        = string
}

variable "ecs_task_definitions" {
  description = "Map of ECS task definition ARNs"
  type        = map(string)
}

variable "sns_topic_arn" {
  description = "ARN of the SNS topic for notifications"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# State machine definition
resource "aws_sfn_state_machine" "rca_pipeline" {
  name     = "${var.name_prefix}-state-machine"
  role_arn = var.execution_role_arn

  definition = jsonencode({
    Comment = "RCA Pipeline - Root Cause Analysis for Stock Price Movements"
    StartAt = "ValidateInput"
    States = {
      ValidateInput = {
        Type = "Task"
        Resource = var.lambda_function_arns["pipeline-trigger"]
        ResultPath = "$.validation"
        Next = "DriverAnalysisMap"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath = "$.error"
          Next = "HandleError"
        }]
      }

      DriverAnalysisMap = {
        Type = "Map"
        ItemsPath = "$.tickers"
        MaxConcurrency = 10
        ResultPath = "$.driver_results"
        Iterator = {
          StartAt = "DriverAnalysis"
          States = {
            DriverAnalysis = {
              Type = "Task"
              Resource = var.lambda_function_arns["driver-analysis"]
              TimeoutSeconds = 300
              Retry = [
                {
                  ErrorEquals = ["States.TaskFailed", "States.Timeout"]
                  IntervalSeconds = 2
                  MaxAttempts = 3
                  BackoffRate = 2
                }
              ]
              End = true
            }
          }
        }
        Next = "QuantNewsAnalysis"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath = "$.error"
          Next = "HandleError"
        }]
      }

      QuantNewsAnalysis = {
        Type = "Task"
        Resource = var.lambda_function_arns["quant-news-analysis"]
        ResultPath = "$.quant_results"
        TimeoutSeconds = 300
        Retry = [
          {
            ErrorEquals = ["States.TaskFailed", "States.Timeout"]
            IntervalSeconds = 2
            MaxAttempts = 3
            BackoffRate = 2
          }
        ]
        Next = "SocialSentiment"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath = "$.error"
          Next = "HandleError"
        }]
      }

      SocialSentiment = {
        Type = "Task"
        Resource = var.lambda_function_arns["social-sentiment"]
        ResultPath = "$.social_results"
        TimeoutSeconds = 300
        Retry = [
          {
            ErrorEquals = ["States.TaskFailed", "States.Timeout"]
            IntervalSeconds = 2
            MaxAttempts = 3
            BackoffRate = 2
          }
        ]
        Next = "MergeConsolidate"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath = "$.error"
          Next = "HandleError"
        }]
      }

      MergeConsolidate = {
        Type = "Task"
        Resource = var.lambda_function_arns["merge-consolidate"]
        ResultPath = "$.merged_results"
        TimeoutSeconds = 60
        Next = "CheckAdvancedOptions"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath = "$.error"
          Next = "HandleError"
        }]
      }

      CheckAdvancedOptions = {
        Type = "Choice"
        Choices = [
          {
            Variable = "$.with_advanced_quant"
            BooleanEquals = true
            Next = "AdvancedQuantFargate"
          }
          {
            Variable = "$.with_nlg"
            BooleanEquals = true
            Next = "NLGFargate"
          }
        ]
        Default = "GenerateReports"
      }

      AdvancedQuantFargate = {
        Type = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster = var.ecs_cluster_arn
          TaskDefinition = var.ecs_task_definitions["advanced-quant"]
          LaunchType = "FARGATE"
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets = [".$$.Execution.Input.subnet_ids"]
              SecurityGroups = [".$$.Execution.Input.security_group_ids"]
              AssignPublicIp = "DISABLED"
            }
          }
          Overrides = {
            ContainerOverrides = [{
              Name = "advanced-quant"
              Environment = [
                {
                  Name = "JOB_ID"
                  Value = ".$$.Execution.Input.job_id"
                }
                {
                  Name = "TICKERS"
                  Value = ".$$.Execution.Input.tickers"
                }
              ]
            }]
          }
        }
        ResultPath = "$.advanced_quant_results"
        TimeoutSeconds = 3600
        Next = "CheckNLG"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath = "$.error"
          Next = "HandleError"
        }]
      }

      CheckNLG = {
        Type = "Choice"
        Choices = [{
          Variable = "$.with_nlg"
          BooleanEquals = true
          Next = "NLGFargate"
        }]
        Default = "GenerateReports"
      }

      NLGFargate = {
        Type = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster = var.ecs_cluster_arn
          TaskDefinition = var.ecs_task_definitions["nlg-generator"]
          LaunchType = "FARGATE"
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets = [".$$.Execution.Input.subnet_ids"]
              SecurityGroups = [".$$.Execution.Input.security_group_ids"]
              AssignPublicIp = "DISABLED"
            }
          }
          Overrides = {
            ContainerOverrides = [{
              Name = "nlg-generator"
              Environment = [
                {
                  Name = "JOB_ID"
                  Value = ".$$.Execution.Input.job_id"
                }
                {
                  Name = "TICKERS"
                  Value = ".$$.Execution.Input.tickers"
                }
              ]
            }]
          }
        }
        ResultPath = "$.nlg_results"
        TimeoutSeconds = 3600
        Next = "GenerateReports"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath = "$.error"
          Next = "HandleError"
        }]
      }

      GenerateReports = {
        Type = "Task"
        Resource = var.lambda_function_arns["generate-reports"]
        ResultPath = "$.report_results"
        TimeoutSeconds = 60
        Next = "NotifyUser"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath = "$.error"
          Next = "HandleError"
        }]
      }

      NotifyUser = {
        Type = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn = var.sns_topic_arn
          Message = {
            "default" = "RCA Pipeline completed successfully"
            "job_id.$" = "$.job_id"
            "status" = "COMPLETED"
            "artifacts.$" = "$.report_results.artifacts"
          }
          MessageStructure = "json"
        }
        Next = "Success"
      }

      Success = {
        Type = "Succeed"
      }

      HandleError = {
        Type = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn = var.sns_topic_arn
          Message = {
            "default" = "RCA Pipeline failed"
            "job_id.$" = "$.job_id"
            "status" = "FAILED"
            "error.$" = "$.error"
          }
          MessageStructure = "json"
        }
        Next = "Fail"
      }

      Fail = {
        Type = "Fail"
        Error = "PipelineExecutionError"
        Cause = "The RCA pipeline execution failed"
      }
    }
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.state_machine.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  tracing_configuration {
    enabled = true
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.name_prefix}-state-machine"
    }
  )
}

# CloudWatch Log Group for Step Functions
resource "aws_cloudwatch_log_group" "state_machine" {
  name              = "/aws/states/${var.name_prefix}"
  retention_in_days = 30

  tags = merge(
    var.tags,
    {
      Name = "${var.name_prefix}-state-machine-logs"
    }
  )
}

# Outputs
output "state_machine_arn" {
  description = "ARN of the state machine"
  value       = aws_sfn_state_machine.rca_pipeline.arn
}

output "state_machine_name" {
  description = "Name of the state machine"
  value       = aws_sfn_state_machine.rca_pipeline.name
}

output "state_machine_id" {
  description = "ID of the state machine"
  value       = aws_sfn_state_machine.rca_pipeline.id
}
