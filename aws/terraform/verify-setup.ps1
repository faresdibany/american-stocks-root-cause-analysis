# Terraform & AWS Setup Verification Script
# ==========================================

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "   RCA Pipeline - Setup Verification" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

# Check Terraform
Write-Host "Checking Terraform..." -ForegroundColor Yellow
$terraform = Get-Command terraform -ErrorAction SilentlyContinue
if ($terraform) {
    Write-Host "  OK - Terraform is installed" -ForegroundColor Green
    Write-Host "     Location: $($terraform.Source)"
    $version = terraform version -json 2>$null | ConvertFrom-Json
    if ($version) {
        Write-Host "     Version: $($version.terraform_version)" -ForegroundColor Green
    } else {
        $versionText = terraform version 2>$null | Select-Object -First 1
        Write-Host "     Version: $versionText" -ForegroundColor Green
    }
} else {
    Write-Host "  ERROR - Terraform not found" -ForegroundColor Red
    Write-Host "     Install from: https://www.terraform.io/downloads" -ForegroundColor Yellow
}

Write-Host ""

# Check AWS CLI
Write-Host "Checking AWS CLI..." -ForegroundColor Yellow
$aws = Get-Command aws -ErrorAction SilentlyContinue
if ($aws) {
    Write-Host "  OK - AWS CLI is installed" -ForegroundColor Green
    Write-Host "     Location: $($aws.Source)"
    $awsVersion = aws --version 2>&1
    Write-Host "     Version: $awsVersion" -ForegroundColor Green
} else {
    Write-Host "  ERROR - AWS CLI not found" -ForegroundColor Red
    Write-Host "     Install from: https://aws.amazon.com/cli/" -ForegroundColor Yellow
}

Write-Host ""

# Check AWS Configuration
Write-Host "Checking AWS Configuration..." -ForegroundColor Yellow
if (Test-Path "$env:USERPROFILE\.aws\credentials") {
    Write-Host "  OK - AWS credentials file exists" -ForegroundColor Green
    
    try {
        $identity = aws sts get-caller-identity --output json 2>&1 | ConvertFrom-Json
        Write-Host "  OK - AWS credentials are valid" -ForegroundColor Green
        Write-Host "     Account: $($identity.Account)"
        Write-Host "     User: $($identity.Arn.Split('/')[-1])"
    } catch {
        Write-Host "  WARNING - AWS credentials exist but could not verify" -ForegroundColor Yellow
        Write-Host "     Run: aws configure" -ForegroundColor Yellow
    }
} else {
    Write-Host "  INFO - AWS not configured yet" -ForegroundColor Yellow
    Write-Host "     Run: aws configure" -ForegroundColor Yellow
}

Write-Host ""

# Check Terraform Workspace
Write-Host "Checking Terraform Workspace..." -ForegroundColor Yellow
$terraformDir = "$(Get-Location)\aws\terraform"
if (Test-Path $terraformDir) {
    Write-Host "  OK - Terraform directory found: $terraformDir" -ForegroundColor Green
    
    if (Test-Path "$terraformDir\.terraform") {
        Write-Host "  OK - Terraform initialized" -ForegroundColor Green
    } else {
        Write-Host "  INFO - Terraform not yet initialized" -ForegroundColor Cyan
        Write-Host "     Run: terraform init" -ForegroundColor Cyan
    }
    
    if (Test-Path "$terraformDir\terraform.tfvars") {
        Write-Host "  OK - Configuration file exists" -ForegroundColor Green
    } else {
        Write-Host "  INFO - No terraform.tfvars file" -ForegroundColor Cyan
        Write-Host "     Copy: terraform.tfvars.example to terraform.tfvars" -ForegroundColor Cyan
    }
} else {
    Write-Host "  INFO - Not in terraform directory" -ForegroundColor Cyan
    Write-Host "     Navigate to: aws\terraform" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "================================================`n" -ForegroundColor Cyan

# Summary
$tfOk = Get-Command terraform -ErrorAction SilentlyContinue
$awsOk = Get-Command aws -ErrorAction SilentlyContinue
$awsConfigured = Test-Path "$env:USERPROFILE\.aws\credentials"

if ($tfOk -and $awsOk) {
    Write-Host "SUCCESS - All tools installed!" -ForegroundColor Green
    
    if ($awsConfigured) {
        Write-Host "SUCCESS - Ready to deploy infrastructure!" -ForegroundColor Green
        Write-Host "`nNext steps:" -ForegroundColor Cyan
        Write-Host "  cd aws\terraform"
        Write-Host "  terraform init"
        Write-Host "  terraform plan`n"
    } else {
        Write-Host "ACTION NEEDED - Configure AWS credentials first:" -ForegroundColor Yellow
        Write-Host "  aws configure`n"
    }
} else {
    Write-Host "ERROR - Some tools are missing. Please install them." -ForegroundColor Yellow
}
