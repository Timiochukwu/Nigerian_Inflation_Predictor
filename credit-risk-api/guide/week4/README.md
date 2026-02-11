# 🔴 Week 4: Deployment & Portfolio

**Ship it! Get your API live and add it to your portfolio**

---

## Overview

Take your working API and deploy it to the cloud so you can:
- ✅ Share a live demo link
- ✅ Add to your resume/LinkedIn
- ✅ Show employers in interviews
- ✅ Include in your portfolio

### What You'll Build

- ✅ `Dockerfile` - Container configuration
- ✅ `docker-compose.yml` - Local orchestration
- ✅ Cloud deployment (Railway/Render)
- ✅ Production README
- ✅ Deployment guide

### Daily Breakdown

| Day | Focus | Time | Deliverable |
|-----|-------|------|-------------|
| **Day 16** | Docker Setup | 45m | Working Dockerfile |
| **Day 17** | Docker Compose | 30m | Local container testing |
| **Day 18** | Cloud Deployment | 1h | **Live URL!** 🎉 |
| **Day 19** | Monitoring & Docs | 45m | Health checks, README |
| **Day 20** | Portfolio Polish | 1h | LinkedIn post, resume update |

**Total Time:** ~3-4 hours

---

## Learning Outcomes

1. **Docker & Containers**
   - Dockerfile creation
   - Multi-stage builds
   - Volume mounting

2. **Cloud Deployment**
   - Deploy to Railway/Render
   - Environment variables
   - Health checks

3. **Production Practices**
   - Monitoring & logging
   - Documentation
   - Portfolio presentation

4. **Career Skills**
   - Building in public
   - Showcasing work
   - Interview preparation

---

## Prerequisites

- ✅ Week 3 complete (working FastAPI)
- ✅ API tested locally
- ✅ Docker installed (optional for local testing)
- ✅ GitHub account

---

## Quick Deploy (Fastest Path)

**Deploy to Railway in 5 minutes:**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
cd credit-risk-api
railway init

# Deploy!
railway up
```

**Result:** Live URL like `https://credit-risk-api.railway.app` 🎉

---

## Day-by-Day Guide

### Day 16: Docker Setup

**Create `Dockerfile`:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health')"

# Run app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and test:**
```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

Visit: http://localhost:8000/docs

---

### Day 17: Docker Compose

**Create `docker-compose.yml`:**

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

**Run with compose:**
```bash
docker-compose up
```

**Benefits:**
- Easier local development
- Environment management
- Volume mounting for models

---

### Day 18: Cloud Deployment

Choose your platform:

#### Option A: Railway (Recommended)

**Why Railway?**
- ✅ Free tier (500 hours/month)
- ✅ Automatic HTTPS
- ✅ GitHub integration
- ✅ Easy CLI

**Steps:**
```bash
# Install & login
npm install -g @railway/cli
railway login

# Deploy
cd credit-risk-api
railway init
railway up

# Get your URL
railway open
```

**Your API is now live!** 🚀

#### Option B: Render

**Why Render?**
- ✅ Free tier (750 hours/month)
- ✅ Auto-deploy from GitHub
- ✅ Easy setup

**Steps:**
1. Push code to GitHub
2. Go to render.com → New Web Service
3. Connect your repo
4. Configure:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
5. Deploy!

**Your API is now live!** 🚀

---

### Day 19: Monitoring & Documentation

**Add health monitoring:**

Already built! Your `/api/v1/health` endpoint returns:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v1"
}
```

**Set up uptime monitoring (free):**
- UptimeRobot: https://uptimerobot.com
- Pingdom: https://www.pingdom.com
- StatusCake: https://www.statuscake.com

**Create production README:**

Already created! See `README_PRODUCTION.md` with:
- Overview & features
- Architecture diagram
- API documentation
- Deployment guide
- Performance metrics

**Create deployment guide:**

Already created! See `DEPLOYMENT_GUIDE.md` with:
- Railway deployment
- Render deployment
- AWS/GCP deployment
- Docker deployment

---

### Day 20: Portfolio Polish

This is where you turn your project into a career asset!

#### 1. Update GitHub Repository

**README.md should have:**
- [ ] Project title & description
- [ ] **Live demo link** (your deployed URL)
- [ ] Performance metrics (97.9% accuracy!)
- [ ] Tech stack
- [ ] Quick start guide
- [ ] API documentation link
- [ ] Screenshots

**Add to top of README:**
```markdown
# 🏦 Credit Risk Scoring API

**Production ML API for real-time credit risk assessment | 97.9% Accuracy**

🚀 **[Live Demo](https://your-app.railway.app/docs)** | 📚 **[API Docs](https://your-app.railway.app/docs)**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-97.9%25-orange.svg)]()
```

#### 2. Update LinkedIn

**Post template:**
```
🚀 Just deployed my Credit Risk Scoring API!

Built a production-grade machine learning API that:
✅ Predicts loan default risk in <100ms
✅ Achieves 97.9% accuracy with XGBoost
✅ Serves predictions via FastAPI
✅ Deployed with Docker on Railway

Tech Stack: Python, XGBoost, FastAPI, Docker, Railway

Try the live demo: [your-url]

Key features:
- 38 engineered features from customer loan history
- 99.87% ROC-AUC score
- RESTful API with interactive docs
- Production-ready with health checks & monitoring

This was a great learning experience in:
- Feature engineering for credit risk
- Building production ML pipelines
- REST API development
- Cloud deployment & MLOps

#MachineLearning #DataScience #FastAPI #Python #MLOps

[Add screenshot of Swagger UI]
```

#### 3. Update Resume

**Add to Projects section:**

```
Credit Risk Scoring API | Python, FastAPI, XGBoost, Docker
• Built production ML API predicting loan default risk with 97.9% accuracy
• Engineered 38 features from demographics and historical loan data
• Deployed containerized application on Railway with <100ms response time
• Implemented RESTful endpoints with automatic validation and documentation
• Tech: XGBoost, FastAPI, Pydantic, Docker, scikit-learn

Live Demo: [your-url]
```

#### 4. Prepare for Interviews

**Have ready:**
- Live demo URL
- GitHub repository link
- 2-minute project explanation
- Technical deep-dive (if asked)

**2-minute explanation template:**

"I built a credit risk scoring API that helps lenders assess loan applications in real-time. It uses XGBoost machine learning model trained on customer demographics and historical loan repayment data.

The system achieves 97.9% accuracy and can process predictions in under 100 milliseconds. I engineered 38 features from the raw data, handling challenges like class imbalance and missing historical data for new customers.

I deployed it as a REST API using FastAPI, containerized with Docker, and hosted on Railway. It includes health monitoring, comprehensive API documentation, and handles both single and batch predictions.

The project demonstrates my skills in feature engineering, model training, API development, and deployment - the full ML engineering lifecycle.

Here's the live demo: [show Swagger UI]"

#### 5. Portfolio Website

Add a project card:

```markdown
### Credit Risk Scoring API
Production ML API for loan default prediction

- **Tech:** XGBoost, FastAPI, Docker, Railway
- **Accuracy:** 97.9%
- **Latency:** <100ms
- **[Live Demo](your-url)** | **[GitHub](repo-url)**

![API Screenshot](screenshot.png)
```

---

## Success Criteria

By end of Week 4, you should have:

- [ ] API deployed to cloud (live URL)
- [ ] README with live demo link
- [ ] LinkedIn post published
- [ ] Resume updated
- [ ] Portfolio website updated
- [ ] Interview talking points prepared
- [ ] Health monitoring set up

---

## Your Deployed API

**What you have now:**

🌐 **Live URL:** `https://your-app.railway.app`

**Endpoints:**
- `GET /` - API info
- `GET /api/v1/health` - Health check
- `POST /api/v1/predict` - Single prediction
- `POST /api/v1/predict/batch` - Batch predictions
- `GET /api/v1/model/info` - Model metadata
- `GET /docs` - Interactive API docs

**Performance:**
- Accuracy: 97.9%
- ROC-AUC: 99.87%
- Response time: <100ms
- Uptime: 99%+

---

## Share Your Success!

**Tweet it:**
```
Just shipped my Credit Risk API! 🚀

✅ 97.9% accuracy
✅ <100ms predictions
✅ Deployed with Docker
✅ Live on Railway

Try it: [your-url]/docs

Built with #Python #FastAPI #XGBoost #MLOps

#100DaysOfCode #MachineLearning
```

**Add to portfolio sites:**
- Dev.to
- Medium
- Hashnode
- Your personal website

---

## Maintenance & Monitoring

### Weekly Checks
- Check uptime (Railway dashboard)
- Review error logs
- Test all endpoints

### Monthly Tasks
- Review model performance
- Update dependencies
- Check for security updates

### Optional Enhancements
- Add API authentication
- Implement rate limiting
- Add caching (Redis)
- Create frontend dashboard
- Set up CI/CD pipeline

---

## What Employers See

When you share this project, employers see:

✅ **Production-ready code** (not tutorial projects)
✅ **Live deployment** (you can ship!)
✅ **Clean documentation** (professional)
✅ **Modern tech stack** (FastAPI, Docker, cloud)
✅ **End-to-end ML** (data → model → API)
✅ **Performance metrics** (measurable results)

**This is portfolio gold!** 🏆

---

## Next Steps

### Immediate (This Week)
1. Deploy to Railway/Render
2. Update LinkedIn
3. Update resume
4. Start applying to ML Engineer roles

### Short-term (This Month)
1. Add authentication
2. Create frontend demo
3. Write blog post about building it
4. Build 2-3 more similar projects

### Long-term (3-6 Months)
1. Build portfolio of 5+ projects
2. Apply to 20-30 remote ML roles
3. Target: 10M-30M NGN/month positions
4. Land remote ML Engineer job! 🎯

---

## 🎉 Congratulations!

You've built a **production-grade ML system** from scratch!

**What you learned:**
- Feature engineering
- Model training & evaluation
- REST API development
- Containerization
- Cloud deployment
- Portfolio building

**What you have:**
- Live API (show it off!)
- GitHub project (recruiters love this)
- Portfolio piece (stand out)
- Interview material (talk about it)

**You're now equipped to:**
- Apply for ML Engineer roles
- Build more ML projects
- Ship production systems
- Earn 5M+ monthly! 🚀

---

**Ready to land that remote job?**

Start applying! Use this project as your portfolio centerpiece.

Good luck! 💪
