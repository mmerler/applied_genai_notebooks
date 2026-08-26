# RNN Class Activity

!!! note "Related Marimo Notebook"
    This module builds upon concepts from the following interactive notebook:

    - [Module 6 Practical 2: Recurrent Neural Networks (RNN)](https://github.com/gurgentus/applied_genai_notebooks/blob/main/notebooks/Module_6_Practical_2_RNN.py)

As part of this activity we will update the text generation API developed in Module 3 to use an RNN. You will incorporate appropriate code from Module6-RNN notebook into your FastAPI implementation. 

---

## 1. Code Updates

Most of the activity code from Module 3 can be reused, but you need to modify the following lines:


```python
bigram_model = BigramModel(corpus)
```

Instead of the BigramModel you should use one of the autoregressive models (RNN, LSTM, GRU).

Similarly, replace the generate_text function/endpoint from the BigramModel with the analogous generate_with_rnn function/endpoint using the neural net.

```python
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    generated_text = # TODO
    return {"generated_text": generated_text}
```

## 2. Test the Text Generation functionality

Test the application using `uv run fastapi dev app/main.py` to make sure that the /generate_with_rnn api endpoint works correctly. Optionally, rebuild the Docker image if you're using containerization.

---

## 3. Optional: Train Your Model on AWS

For larger datasets or more complex models, you can train your RNN on AWS cloud infrastructure using Amazon SageMaker.

### SageMaker Training Script

Create a training script compatible with SageMaker:

```python
# train.py
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    return parser.parse_args()

def train_model(args):
    # TODO: Implement training loop based on Module 7 RNN notebook
    # 1. Load training data from args.data_dir
    # 2. Initialize RNN model (LSTM/GRU) 
    # 3. Set up loss function and optimizer
    # 4. Train for specified epochs
    # 5. Save model to args.model_dir
    
    print(f"Training completed. Model saved to {args.model_dir}")

if __name__ == '__main__':
    args = parse_args()
    train_model(args)
```

### Launch Training Job

```python
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

pytorch_estimator = PyTorch(
    entry_point='train.py',
    role=get_execution_role(),
    framework_version='1.8.0',
    py_version='py3',
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    hyperparameters={
        'epochs': 50,
        'batch-size': 64,
        'learning-rate': 0.001
    }
)

pytorch_estimator.fit({'training': 's3://your-bucket/training-data/'})
```

---

## 4. Optional: Deploy to AWS using ECS

Deploy your RNN text generation API to production using Amazon ECS (Elastic Container Service).

### Step 1: Push Docker Image to ECR

```bash
# Create ECR repository
aws ecr create-repository --repository-name rnn-text-gen

# Get login token and authenticate Docker
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push your image
docker tag sps-genai:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/rnn-text-gen:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/rnn-text-gen:latest
```

### Step 2: Create ECS Task Definition

```json
{
  "family": "rnn-text-gen-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "rnn-api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/rnn-text-gen:latest",
      "portMappings": [
        {
          "containerPort": 80,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rnn-text-gen",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Step 3: Deploy ECS Service with Load Balancer

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name rnn-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create ECS service
aws ecs create-service \
  --cluster rnn-cluster \
  --service-name rnn-text-gen-service \
  --task-definition rnn-text-gen-task:1 \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

### Step 4: Set Up Auto-Scaling (Optional)

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/rnn-cluster/rnn-text-gen-service \
  --min-capacity 1 \
  --max-capacity 5

# Create scaling policy based on CPU utilization
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/rnn-cluster/rnn-text-gen-service \
  --policy-name cpu-scaling-policy \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    }
  }'
```

### Step 5: Test Your Deployment

```bash
# Test the deployed API endpoint
curl -X POST http://your-load-balancer-url/generate_with_rnn \
  -H "Content-Type: application/json" \
  -d '{"start_word": "hello", "length": 10}'
```

### Cost Management

Set up billing alerts to monitor costs:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "RNN-Deployment-Cost-Alert" \
  --alarm-description "Alert when deployment costs exceed $50" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold
```

**Benefits of ECS Deployment:**
- Scalable and production-ready
- Automatic load balancing
- Health checks and auto-recovery
- Integration with other AWS services
- Cost-effective for consistent traffic