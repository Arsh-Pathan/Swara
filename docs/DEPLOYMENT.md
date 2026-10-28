# Swara Production Deployment Guide

This guide describes how to deploy, configure, scale, and monitor Swara in a production containerized environment.

---

## 1. Container Deployment Architecture

```
Internet / Client Requests
          │
          ▼
┌────────────────────────────────────────┐
│  Reverse Proxy (Nginx / Cloud ALB)     │
│  - SSL / TLS Termination (HTTPS)       │
│  - Rate Limiting (10 req/sec)          │
│  - Request Size Cap (50 MB)            │
└────────────────────────────────────────┘
          │ (Internal Network)
          ▼
┌────────────────────────────────────────┐
│  Swara Docker Container (port 8000)    │
│  - Non-root user (UID 10001)           │
│  - CPU Quota: 4 cores, RAM: 1024 MB    │
│  - Shared ONNX Runtime Engine          │
└────────────────────────────────────────┘
     │                      │
     ▼                      ▼
┌──────────────┐     ┌──────────────────┐
│ /models/cache│     │ /data/speakers   │
│ (Volume)     │     │ (Volume)         │
└──────────────┘     └──────────────────┘
```

---

## 2. Docker Compose Deployment

### 2.1 Configuration
Inspect or create `.env` from template:
```bash
cp infra/deployment/production.env.example .env
```

### 2.2 Launch Container Cluster
```bash
docker compose -f infra/compose/compose.yaml up -d --build
```

### 2.3 Verify Service Liveness & Readiness
```bash
# Check health
curl http://localhost:8000/health

# Check readiness & loaded models
curl http://localhost:8000/ready
```

---

## 3. High Availability & Scaling

Because Swara is stateless with respect to inference (all model weights are static ONNX sessions in shared memory), horizontal scaling is straightforward:

1. **Model Cache:** Mount the model directory `/app/models/cache` read-only across all worker replicas.
2. **Speaker Database:**
   - In single-node deployments, mount `/app/data/speakers` to a persistent Docker volume.
   - In multi-node Kubernetes / ECS clusters, mount `/app/data/speakers` on an NFS / Amazon EFS volume or sync via an object storage bucket (e.g. S3 / GCS).
3. **Number of Workers:**
   - On CPU, run 1 Uvicorn worker per container and scale via Docker Compose replicas or Kubernetes Pod replicas.
   - Allocate 2 to 4 CPU cores per replica.

---

## 4. Rollback Strategy

If a newly deployed image fails startup readiness:
```bash
# Revert to previous image tag
docker compose -f infra/compose/compose.yaml down
sed -i 's/swara:1.0.1/swara:1.0.0/g' infra/compose/compose.yaml
docker compose -f infra/compose/compose.yaml up -d
```
All persistent speaker profiles in `swara_speaker_data` remain intact across container image redeployments.
