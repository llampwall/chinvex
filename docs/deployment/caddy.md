# Caddy Reverse Proxy Setup

## Prerequisites

- Caddy installed
- Gateway running on port 7778
- Domain with DNS pointing to server

## Caddyfile

Add to existing Caddyfile:

```
chinvex.yourdomain.com {
    reverse_proxy localhost:7778

    header {
        X-Content-Type-Options nosniff
        X-Frame-Options DENY
        X-XSS-Protection "1; mode=block"
        Strict-Transport-Security "max-age=31536000"
    }

    log {
        output file /var/log/caddy/chinvex.log
        format json
    }
}
```

## Start Services

```bash
pm2 start "chinvex gateway serve --port 7778" --name chinvex-gateway
caddy reload
```

## Verification

```bash
curl https://chinvex.yourdomain.com/health
curl -I https://chinvex.yourdomain.com/health
tail -f /var/log/caddy/chinvex.log
```
