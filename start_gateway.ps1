$env:CHINVEX_API_TOKEN = "KLsp7WJZo7kHaV6IB3W25-UN1tRUTmmenfLzwcJGvHc"
Write-Host "Token set to: $($env:CHINVEX_API_TOKEN.Substring(0,8))..."
python -m chinvex.cli gateway serve --port 7778
