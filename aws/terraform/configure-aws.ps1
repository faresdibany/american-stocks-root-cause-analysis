# AWS CLI Configuration Script
# Run this after installation to configure your AWS credentials

Write-Host "===========================================`n"
Write-Host "  AWS CLI Configuration Wizard`n"
Write-Host "===========================================`n"

# Check if AWS CLI is available
try {
    aws --version | Out-Null
    Write-Host "✅ AWS CLI detected`n" -ForegroundColor Green
} catch {
    Write-Host "❌ AWS CLI not found. Please install it first." -ForegroundColor Red
    exit 1
}

Write-Host "You'll need the following information:" -ForegroundColor Yellow
Write-Host "  1. AWS Access Key ID"
Write-Host "  2. AWS Secret Access Key"
Write-Host "  3. Default region (recommended: us-east-1)"
Write-Host "  4. Default output format (recommended: json)`n"

Write-Host "📝 Getting your credentials:" -ForegroundColor Cyan
Write-Host "  1. Go to: https://console.aws.amazon.com/iam/home#/security_credentials"
Write-Host "  2. Click 'Create access key'"
Write-Host "  3. Download the credentials`n"

$response = Read-Host "Do you have your AWS credentials ready? (y/n)"

if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host "`nStarting AWS configuration...`n" -ForegroundColor Green
    
    # Run aws configure
    aws configure
    
    Write-Host "`n===========================================`n" -ForegroundColor Green
    Write-Host "✅ Configuration complete!`n" -ForegroundColor Green
    
    # Test the configuration
    Write-Host "Testing configuration..." -ForegroundColor Cyan
    try {
        $identity = aws sts get-caller-identity --output json | ConvertFrom-Json
        Write-Host "`n✅ Successfully connected to AWS!" -ForegroundColor Green
        Write-Host "  Account: $($identity.Account)"
        Write-Host "  User ARN: $($identity.Arn)"
        Write-Host "  User ID: $($identity.UserId)`n"
    } catch {
        Write-Host "`n⚠️  Configuration saved, but couldn't verify connection." -ForegroundColor Yellow
        Write-Host "  Please check your credentials and network connection.`n"
    }
    
    Write-Host "===========================================`n" -ForegroundColor Green
    Write-Host "🚀 You're ready to use Terraform with AWS!`n"
    Write-Host "Next steps:"
    Write-Host "  1. cd aws\terraform"
    Write-Host "  2. terraform init"
    Write-Host "  3. terraform plan`n"
    
} else {
    Write-Host "`n📋 To configure later, run: aws configure`n" -ForegroundColor Yellow
    Write-Host "Or set environment variables:"
    Write-Host '  $env:AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY"'
    Write-Host '  $env:AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_KEY"'
    Write-Host '  $env:AWS_DEFAULT_REGION = "us-east-1"`n'
}
