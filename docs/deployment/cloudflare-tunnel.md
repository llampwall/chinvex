# Cloudflare Tunnel Setup

## Prerequisites

- Cloudflare account with a domain
- Gateway running locally on port 7778

## Installation

### Windows
```powershell
winget install cloudflare.cloudflared
```

### Verify
```bash
cloudflared --version
```

## Configuration

### 1. Login and create tunnel
```bash
cloudflared tunnel login
cloudflared tunnel create chinvex
```

### 2. Configure tunnel

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: chinvex
credentials-file: C:\Users\Jordan\.cloudflared\<tunnel-id>.json

ingress:
  - hostname: chinvex.yourdomain.com
    service: http://localhost:7778
  - service: http_status:404
```

### 3. Add DNS
```bash
cloudflared tunnel route dns chinvex chinvex.yourdomain.com
```

### 4. Test
```bash
cloudflared tunnel run chinvex
```

Visit `https://chinvex.yourdomain.com/health`

### 5. Run as service (PM2)

```bash
pm2 start "cloudflared tunnel run chinvex" --name chinvex-tunnel
pm2 start "chinvex gateway serve --port 7778" --name chinvex-gateway
pm2 save
pm2 startup
```

## Verification

```bash
cloudflared tunnel info chinvex
pm2 logs chinvex-tunnel
curl https://chinvex.yourdomain.com/health
```
