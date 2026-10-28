# Swara Deployment & Infrastructure

This directory contains the production container, deployment assets, reverse proxy configurations, and monitoring hooks for **Swara**.

---

## Directory Layout

```
infra/
├── docker/
│   ├── Dockerfile             # Multi-stage minimal production image (non-root UID 10001)
│   └── .dockerignore          # Build context exclusion rules
├── compose/
│   └── compose.yaml           # Docker Compose specification with limits & volume mounts
├── deployment/
│   ├── production.env.example # Production environment variables template
│   └── nginx.conf             # Upstream reverse proxy, rate-limiting & security headers
├── scripts/
│   ├── setup.sh               # Automated local host setup script
│   └── healthcheck.sh         # Liveness/readiness verification script
├── monitoring/
│   └── prometheus.yml         # Prometheus metrics scrape configuration
└── security/
    └── DEPLOYMENT_SECURITY.md # Biometric privacy, threat model & hardening guide
```

---

## Quickstart: Docker Compose Deployment

### 1. Build and Run Container
```bash
docker compose -f infra/compose/compose.yaml up -d --build
```

### 2. Verify Container Health
```bash
bash infra/scripts/healthcheck.sh localhost 8000
```

### 3. Inspect Running Service Logs
```bash
docker compose -f infra/compose/compose.yaml logs -f
```

### 4. Stop Service
```bash
docker compose -f infra/compose/compose.yaml down
```

---

## Resource Sizing Guidelines

| Deployment Profile | Target Concurrency | CPU Allocation | RAM Limit | Real-Time Factor (RTF) |
|---|---|---|---|---|
| **Edge / Low-Memory** | 1 - 2 streams | 1 - 2 cores | 512 MB | ~0.02 |
| **Standard Production** | 5 - 10 concurrent streams | 4 cores | 1024 MB | ~0.01 |
| **High-Throughput Cluster** | 20+ streams | 8+ cores (or CUDA GPU) | 2048 MB | < 0.005 |
