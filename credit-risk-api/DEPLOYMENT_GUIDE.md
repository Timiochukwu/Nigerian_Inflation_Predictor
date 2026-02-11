# 🚀 Deployment Guide

## Quick Deployment Options

### Option 1: Railway (Easiest - Free Tier)

1. **Sign up at [Railway.app](https://railway.app/)**

2. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

3. **Deploy:**
   ```bash
   cd credit-risk-api
   railway login
   railway init
   railway up
   ```

4. **Access your API:**
   - Railway will provide a URL like: `https://your-app.railway.app`
   - API docs: `https://your-app.railway.app/docs`

**Cost:** FREE (500 hours/month)

---

### Option 2: Render (Simple - Free Tier)

1. **Sign up at [Render.com](https://render.com/)**

2. **Create New Web Service:**
   - Connect your GitHub repository
   - Select **Web Service**

3. **Configure:**
   ```
   Name: credit-risk-api
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 4
   ```

4. **Deploy** - Automatic from main branch

**Cost:** FREE (750 hours/month)

---

### Option 3: Heroku (Traditional)

1. **Install Heroku CLI:**
   ```bash
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Create Procfile:**
   ```bash
   echo "web: uvicorn src.api.main:app --host 0.0.0.0 --port \$PORT" > Procfile
   ```

3. **Deploy:**
   ```bash
   heroku login
   heroku create your-credit-api
   git push heroku main
   ```

---

### Option 4: AWS (Production-Grade)

#### Using AWS App Runner (Easiest AWS option)

1. **Push Docker image to ECR:**
   ```bash
   # Build and tag image
   docker build -t credit-risk-api .

   # Create ECR repository
   aws ecr create-repository --repository-name credit-risk-api

   # Login to ECR
   aws ecr get-login-password --region us-east-1 | \
   docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.us-east-1.amazonaws.com

   # Tag and push
   docker tag credit-risk-api:latest <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/credit-risk-api:latest
   docker push <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/credit-risk-api:latest
   ```

2. **Create App Runner service** via AWS Console:
   - Source: ECR image
   - Port: 8000
   - Health check: /api/v1/health
   - Auto-scaling: 1-10 instances

#### Using AWS EC2 (Full Control)

```bash
# SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Clone repo and run
git clone <your-repo>
cd credit-risk-api
docker-compose up -d

# Set up nginx reverse proxy (optional)
sudo apt install nginx
# Configure /etc/nginx/sites-available/default
# Proxy port 80 → 8000
```

---

### Option 5: Google Cloud Run (Serverless)

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash

# Authenticate
gcloud auth login

# Build and deploy
gcloud run deploy credit-risk-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Your API will be at: https://credit-risk-api-<hash>.run.app
```

---

## Production Checklist

Before deploying to production:

### Security
- [ ] Change default `CORS` settings (restrict origins)
- [ ] Add API key authentication
- [ ] Enable HTTPS (most platforms do this automatically)
- [ ] Secure environment variables
- [ ] Add rate limiting

### Performance
- [ ] Set appropriate worker count (CPU cores × 2)
- [ ] Configure gunicorn/uvicorn properly
- [ ] Add caching for frequent predictions
- [ ] Set up CDN if serving static content

### Monitoring
- [ ] Set up health check alerts
- [ ] Add application metrics (Prometheus/Grafana)
- [ ] Configure log aggregation (ELK, CloudWatch)
- [ ] Set up error tracking (Sentry)

### Reliability
- [ ] Configure auto-scaling
- [ ] Set up load balancer (if multiple instances)
- [ ] Enable automatic restarts
- [ ] Add backup/failover

---

## Environment Variables for Production

```bash
# Required
APP_NAME=Credit Risk Scoring API
DEBUG=false
PORT=8000
WORKERS=4

# Model Configuration
MODEL_NAME=credit_risk_xgboost
MODEL_VERSION=v1

# Performance Tuning
MAX_BATCH_SIZE=1000
PREDICTION_TIMEOUT=30

# Monitoring (optional)
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

---

## Scaling Considerations

### Small Scale (0-100 requests/minute)
- Railway/Render free tier
- 1-2 workers
- $0-20/month

### Medium Scale (100-1000 requests/minute)
- Railway Pro or Render Standard
- 4-8 workers
- $20-100/month

### Large Scale (1000+ requests/minute)
- AWS/GCP with auto-scaling
- 10-50+ instances
- $100-1000+/month
- Consider adding:
  - Load balancer
  - Redis caching
  - Database for logging
  - CDN

---

## Testing Deployed API

```bash
# Replace with your deployed URL
export API_URL=https://your-app.railway.app

# Health check
curl $API_URL/api/v1/health

# Test prediction
curl -X POST "$API_URL/api/v1/predict" \
-H "Content-Type: application/json" \
-d '{
  "loanamount": 250000,
  "totaldue": 312500,
  "termdays": 180,
  "latitude_gps": 6.5244,
  "longitude_gps": 3.3792,
  "bank_name_clients": "GTBank",
  "bank_branch_clients": "Lagos",
  "employment_status_clients": "Permanent",
  "level_of_education_clients": "HND/BSc"
}'
```

---

## Troubleshooting

### API Not Starting
- Check logs: `docker logs <container-id>`
- Verify model files exist in `models/artifacts/`
- Ensure all dependencies installed

### Slow Predictions
- Reduce batch size
- Increase worker count
- Add caching layer
- Profile with `/api/v1/health` response times

### High Memory Usage
- Reduce worker count
- Use model quantization
- Clear old predictions from memory

---

## Support

For deployment issues:
1. Check application logs
2. Verify all environment variables are set
3. Test locally first with Docker
4. Review platform-specific documentation

---

**🎉 Once deployed, update your LinkedIn/resume with the live API URL!**
