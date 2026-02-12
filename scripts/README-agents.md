# PEST++ Distributed Agent Deployment

This directory contains scripts for deploying PEST++ agents across multiple machines on your local network.

## Execution Modes

The PEST++ calibration system supports three execution modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Local Process** | Parallel workers as local processes | Single machine, simple setup |
| **Local Containers** | Docker containers on main server | Single machine, isolated execution |
| **Hybrid Network** | Local containers + remote agents | Multiple machines, maximum throughput |

## Quick Start - Local Containers Only

If you only want to run containerized agents on the main server:

```bash
# Start 8 local container agents
docker compose -f docker-compose.pest.yml up -d --scale pest-local-agent=8
```

In the web UI:
1. Go to PEST++ Settings
2. Select "Local Container Agents"
3. Set the number of local agents
4. Start calibration

## Quick Start - Hybrid Network Mode

For distributed execution across multiple machines:

### 1. Configure Your Remote Machines

Create `agents.conf` from the example:

```bash
cp agents.conf.example agents.conf
```

Edit `agents.conf` to add your machines:

```
# Format: hostname_or_ip [username] [num_agents]
192.168.1.101 ubuntu 4
192.168.1.102 ubuntu 6
192.168.1.103 ubuntu 4
```

### 2. Set Up SSH Keys (One-Time Setup)

SSH key authentication is required for passwordless deployment:

```bash
# Set up keys for all machines in agents.conf
./setup-ssh-keys.sh --from-config

# Or set up specific machines
./setup-ssh-keys.sh user@192.168.1.101 user@192.168.1.102
```

### 3. Deploy Agents

Deploy to all machines:

```bash
# Linux/Mac
./deploy-agents.sh -m 192.168.1.100

# Windows PowerShell
.\deploy-agents.ps1 -ManagerIP 192.168.1.100
```

Options:
- `-m, --manager IP` - Main server IP (required)
- `-n, --num-agents N` - Default agents per machine (default: 4)
- `-b, --build` - Force rebuild Docker image
- `-s, --stop` - Stop all agents

### 4. Run a Calibration (Hybrid Mode)

1. In the web UI, go to PEST++ Settings
2. Select "Hybrid Network Mode"
3. Configure local container agents (e.g., 8 on main server)
4. Configure remote network agents (e.g., 12 from other machines)
5. Total agents = local + remote (e.g., 20)
6. Start the calibration

The UI shows:
- Local containers: agents running on the main server
- Remote agents: agents connecting from other machines

## File Overview

| File | Description |
|------|-------------|
| `deploy-agents.sh` | Main deployment script (Linux/Mac) |
| `deploy-agents.ps1` | Main deployment script (Windows PowerShell) |
| `setup-ssh-keys.sh` | SSH key setup helper |
| `agents.conf.example` | Example machine configuration |
| `agents.conf` | Your machine configuration (create this) |
| `start-agents.sh` | Manual agent startup script |
| `start-agents.bat` | Manual agent startup script (Windows) |
| `pest-agent-entrypoint.sh` | Docker container entrypoint |

## Recommended Agent Counts

Resource guidelines based on CPU cores and RAM:

| Machine Type | Cores | RAM | Recommended Agents |
|--------------|-------|-----|-------------------|
| 4-core Linux | 4 | 8GB | 3-4 |
| 8-core Linux | 8 | 16GB | 6-8 |
| 12-core workstation | 12 | 32GB | 10-12 |
| **Intel i7-14700** | 20 | 32GB | **14-16** |
| Intel i9-14900K | 24 | 64GB | 20-22 |
| WSL2 on Windows | varies | varies | CPU cores - 2 |

**Resource allocation per agent:**
- Memory: 2GB per agent (configurable via `AGENT_MEMORY_LIMIT`)
- CPU: 1 core per agent (configurable via `AGENT_CPU_LIMIT`)

For i7-14700 with 32GB RAM:
- 16 agents × 2GB = 32GB RAM
- 16 agents × 1 core = 16 cores (leaving 4 for system/manager)

## Troubleshooting

### SSH Connection Failed

1. Verify you can SSH manually: `ssh user@host`
2. Run `./setup-ssh-keys.sh user@host` to set up keys
3. Check firewall allows SSH (port 22)

### Docker Not Found

Install Docker on the remote machine:

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in
```

### Agents Not Connecting

1. Check firewall allows port 4004 from remote machines
2. Verify main server IP is correct in .env
3. Check agent logs: `ssh user@host 'docker logs pest-agent-1'`

### Image Build Fails

1. Check internet connectivity on remote machine
2. Try with `--build` flag to force rebuild
3. Check disk space: `ssh user@host 'df -h'`

## Manual Deployment

If the automated script doesn't work, you can deploy manually:

```bash
# On remote machine
mkdir -p /tmp/pest-agents
cd /tmp/pest-agents

# Copy files from main server (or use scp)
scp user@main-server:/path/to/model-app/Dockerfile.pest-agent .
scp user@main-server:/path/to/model-app/docker-compose.agent.yml .
scp -r user@main-server:/path/to/model-app/scripts .

# Create .env
cat > .env << EOF
MANAGER_HOST=192.168.1.100
MANAGER_PORT=4004
MINIO_HOST=192.168.1.100
MINIO_PORT=9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
NUM_AGENTS=4
EOF

# Build and run
docker build -f Dockerfile.pest-agent -t modflow-pest-agent:latest .
docker compose -f docker-compose.agent.yml up -d --scale pest-agent=4
```

## Architecture

### Hybrid Mode (Local Containers + Remote Agents)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Server (192.168.1.100)                  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Docker: API, Redis, Postgres, MinIO, Worker            │   │
│  │                                                         │   │
│  │  Worker Container                                       │   │
│  │    └── PEST++ Manager (listening on port 4004)         │   │
│  │                         ▲                               │   │
│  │    ┌────────────────────┴────────────────────┐         │   │
│  │    │         Docker Network                   │         │   │
│  │    ▼                                          ▼         │   │
│  │  ┌──────────────────────────────────────────────┐      │   │
│  │  │  Local Container Agents (pest-local-agent)   │      │   │
│  │  │  × 8-16 agents (docker-compose.pest.yml)     │      │   │
│  │  └──────────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │ TCP/IP (port 4004, exposed)
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Linux PC 1    │      │ Linux PC 2    │      │ Windows PC    │
│ 192.168.1.101 │      │ 192.168.1.102 │      │ (WSL2)        │
│ i7-14700 32GB │      │ i7-14700 32GB │      │ 192.168.1.103 │
│               │      │               │      │               │
│ pest-agent ×14│      │ pest-agent ×14│      │ pest-agent ×6 │
└───────────────┘      └───────────────┘      └───────────────┘

Total Agents: 8-16 local + 34 remote = 42-50 agents
```

### Local Containers Only Mode

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Server (192.168.1.100)                  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Worker Container                                       │   │
│  │    └── PEST++ Manager                                   │   │
│  │              ▲                                          │   │
│  │    ┌─────────┴─────────┐                               │   │
│  │    │  Docker Network   │                               │   │
│  │    ▼                   ▼                   ▼           │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ... ×N           │   │
│  │  │Agent 1 │  │Agent 2 │  │Agent N │                    │   │
│  │  │ 2GB    │  │ 2GB    │  │ 2GB    │                    │   │
│  │  └────────┘  └────────┘  └────────┘                    │   │
│  │  (pest-local-agent containers)                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  No external connections required                               │
└─────────────────────────────────────────────────────────────────┘
```
