# AWS Setup for Account 040604762405 (feldiban)
# ================================================

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "  AWS Account Setup" -ForegroundColor Cyan
Write-Host "  Account: 040604762405" -ForegroundColor Yellow
Write-Host "  Username: feldiban" -ForegroundColor Yellow
Write-Host "=========================================`n" -ForegroundColor Cyan

# Step 1: Get AWS Credentials
Write-Host "Step 1: Get Your AWS Credentials" -ForegroundColor Green
Write-Host "---------------------------------------"
Write-Host "1. Go to AWS Console: https://console.aws.amazon.com/"
Write-Host "2. Sign in as: feldiban"
Write-Host "3. Navigate to: IAM > Users > feldiban > Security credentials"
Write-Host "4. Click 'Create access key'"
Write-Host "5. Select 'Command Line Interface (CLI)'"
Write-Host "6. Download the credentials (CSV file)`n"

$ready = Read-Host "Do you have your Access Key ID and Secret Access Key? (y/n)"

if ($ready -ne 'y' -and $ready -ne 'Y') {
    Write-Host "`nPlease get your credentials first, then run this script again.`n" -ForegroundColor Yellow
    exit 0
}

# Step 2: Configure AWS CLI
Write-Host "`nStep 2: Configure AWS CLI" -ForegroundColor Green
Write-Host "---------------------------------------"
Write-Host "Enter your AWS credentials when prompted:`n" -ForegroundColor Cyan

$accessKey = Read-Host "AWS Access Key ID"
$secretKey = Read-Host "AWS Secret Access Key" -AsSecureString
$secretKeyPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secretKey))
$region = Read-Host "Default region (press Enter for us-east-1)"
if ([string]::IsNullOrWhiteSpace($region)) {
    $region = "us-east-1"
}
$output = Read-Host "Default output format (press Enter for json)"
if ([string]::IsNullOrWhiteSpace($output)) {
    $output = "json"
}

# Set AWS credentials
$env:AWS_ACCESS_KEY_ID = $accessKey
$env:AWS_SECRET_ACCESS_KEY = $secretKeyPlain
$env:AWS_DEFAULT_REGION = $region

# Also save to AWS credentials file
$awsDir = "$env:USERPROFILE\.aws"
New-Item -ItemType Directory -Force -Path $awsDir | Out-Null

$credentialsContent = @"
[default]
aws_access_key_id = $accessKey
aws_secret_access_key = $secretKeyPlain
"@

$configContent = @"
[default]
region = $region
output = $output
"@

$credentialsContent | Out-File -FilePath "$awsDir\credentials" -Encoding ASCII
$configContent | Out-File -FilePath "$awsDir\config" -Encoding ASCII

Write-Host "`n✅ AWS credentials saved" -ForegroundColor Green

# Step 3: Verify Connection
Write-Host "`nStep 3: Verify AWS Connection" -ForegroundColor Green
Write-Host "---------------------------------------"

try {
    $identity = aws sts get-caller-identity --output json | ConvertFrom-Json
    
    if ($identity.Account -eq "040604762405") {
        Write-Host "✅ Successfully connected to AWS!" -ForegroundColor Green
        Write-Host "   Account: $($identity.Account)" -ForegroundColor White
        Write-Host "   User ARN: $($identity.Arn)" -ForegroundColor White
        Write-Host "   User ID: $($identity.UserId)" -ForegroundColor White
    } else {
        Write-Host "⚠️  Connected, but to a different account: $($identity.Account)" -ForegroundColor Yellow
        Write-Host "   Expected: 040604762405" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Failed to verify AWS connection" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`nPlease check your credentials and try again.`n" -ForegroundColor Yellow
    exit 1
}

# Step 4: Update Terraform Config
Write-Host "`nStep 4: Update Terraform Configuration" -ForegroundColor Green
Write-Host "---------------------------------------"

$email = Read-Host "Enter your email for AWS notifications"

# Update terraform.tfvars
$tfvarsPath = ".\aws\terraform\terraform.tfvars"
if (Test-Path $tfvarsPath) {
    $content = Get-Content $tfvarsPath -Raw
    $content = $content -replace 'fares@yourdomain\.com', $email
    $content | Out-File -FilePath $tfvarsPath -Encoding UTF8 -NoNewline
    Write-Host "✅ Updated terraform.tfvars with your email" -ForegroundColor Green
} else {
    Write-Host "⚠️  terraform.tfvars not found at $tfvarsPath" -ForegroundColor Yellow
}

# Step 5: Ready to Deploy
Write-Host "`n=========================================" -ForegroundColor Green
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "=========================================`n" -ForegroundColor Green

Write-Host "You're ready to deploy the RCA Pipeline to AWS!`n"

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Navigate to terraform directory:"
Write-Host "     cd aws\terraform`n"
Write-Host "  2. Initialize Terraform:"
Write-Host "     terraform init`n"
Write-Host "  3. Review the deployment plan:"
Write-Host "     terraform plan`n"
Write-Host "  4. Deploy the infrastructure:"
Write-Host "     terraform apply`n"

Write-Host "⚠️  Important Notes:" -ForegroundColor Yellow
Write-Host "  • The deployment will create ~55 AWS resources"
Write-Host "  • Estimated cost: $150-300/month"
Write-Host "  • Deployment takes ~20-30 minutes"
Write-Host "  • Review terraform.tfvars before deploying"
Write-Host "  • Update ECR image URLs in terraform.tfvars`n"

$deploy = Read-Host "Would you like to navigate to the terraform directory now? (y/n)"

if ($deploy -eq 'y' -or $deploy -eq 'Y') {
    Set-Location -Path ".\aws\terraform"
    Write-Host "`nYou're now in the terraform directory." -ForegroundColor Green
    Write-Host "Run: terraform init`n" -ForegroundColor Cyan
}
