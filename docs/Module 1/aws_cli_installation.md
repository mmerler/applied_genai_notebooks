# AWS CLI Installation (Optional)

**Note**: This installation is optional and only required if you plan to train or deploy your models on AWS cloud infrastructure in later modules (specifically Module 7 and beyond).

⚠️ **Cost Warning**: Using AWS services may incur charges. While AWS offers a free tier, ML training and deployment can exceed free limits. Monitor your usage and set up billing alerts. All coursework can be completed locally at no cost.

---

## When You'll Need This

- **Module 7**: Optional AWS training and deployment of RNN models
- **Future modules**: Cloud-based model training and production deployments
- **Personal projects**: Deploying your own ML applications

If you prefer to work locally only, you can skip this installation entirely.

---

## Installation

### Windows

```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.msi" -o "AWSCLIV2.msi"
msiexec.exe /i AWSCLIV2.msi
```

### macOS

```bash
# Option 1: Direct installer
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /

# Option 2: Using Homebrew
brew install awscli
```

### Linux

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### Verify Installation

```bash
aws --version
```

Expected output:
```
aws-cli/2.15.0 Python/3.11.6 Darwin/23.1.0 exe/x86_64 prompt/off
```

---

## Configuration (When Ready to Use AWS)

### Prerequisites

You'll need:
1. AWS Account (free tier available)
2. AWS access keys from IAM console

### Quick Setup

```bash
aws configure
```

Enter when prompted:
- **AWS Access Key ID**: Your access key
- **AWS Secret Access Key**: Your secret key  
- **Default region**: `us-east-1` (recommended)
- **Default output format**: `json`

### Test Configuration

```bash
aws sts get-caller-identity
```

---

## Cost Management

### Set Up Billing Alerts

```bash
# Create billing alarm (after configuration)
aws cloudwatch put-metric-alarm \
  --alarm-name "Monthly-Spending-Alert" \
  --alarm-description "Alert when monthly spending exceeds $10" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --evaluation-periods 5 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold
```

### Approximate Costs for Course Activities

- **SageMaker Training** (`ml.p3.2xlarge`): ~$3-4/hour
- **ECS Deployment** (small): ~$10-15/month
- **ECR Storage**: ~$0.10/GB/month
- **Data Transfer**: Usually minimal for course work

**Recommendation**: Start with free tier resources and monitor usage closely.

---

## Alternative: Work Locally Only

If you prefer not to use AWS:
- ✅ All training can be done locally with your CPU/GPU
- ✅ Docker deployment works on any cloud provider
- ✅ You'll still learn the core ML concepts without cloud specifics
- ✅ Zero cost for coursework

The choice is yours! AWS experience is valuable for industry work, but local development is perfectly fine for learning the fundamentals.