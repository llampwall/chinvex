module.exports = {
  apps: [
    {
      name: "chinvex-tunnel",
      script: "cloudflared",
      args: "tunnel --protocol http2 run chinvex-gateway",
      autorestart: true,
      cron_restart: "0 */4 * * *",
      restart_delay: 5000
    },
    {
      name: "chinvex-gateway",
      script: "P:\\software\\chinvex\\.venv\\Scripts\\pythonw.exe",
      args: "-m chinvex.cli gateway serve --port 7778",
      cwd: "P:\\software\\chinvex",
      autorestart: true,
      restart_delay: 2000,
      windowsHide: true
    },
  ]
}
