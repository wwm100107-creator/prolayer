# Chạy backend và mở ra Internet qua Tailscale Funnel, để web trên Vercel gọi được.
# Tắt hẳn đường hầm: tailscale funnel reset
$ErrorActionPreference = "Stop"
$state = Join-Path $HOME ".cache\prolayer"
$python = Join-Path $state "venv\Scripts\python.exe"
$keyFile = Join-Path $state "access-key.txt"
$originsFile = Join-Path $state "origins.txt"

# Mã truy cập chỉ nằm trên máy này, không bao giờ đưa vào repo.
if (-not (Test-Path $keyFile)) {
    $bytes = New-Object byte[] 24
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    [Convert]::ToBase64String($bytes).Replace("+", "-").Replace("/", "_") | Set-Content $keyFile -NoNewline
}
if (-not (Test-Path $originsFile)) {
    Read-Host "Địa chỉ web trên Vercel (ví dụ https://prolayer.vercel.app)" | Set-Content $originsFile -NoNewline
}

$env:PROLAYER_ACCESS_KEY = (Get-Content $keyFile -Raw).Trim()
$env:PROLAYER_ORIGINS = (Get-Content $originsFile -Raw).Trim()

tailscale funnel --bg 8000
Write-Host "Mã truy cập để nhập trên web: $env:PROLAYER_ACCESS_KEY"
& $python -m uvicorn app.main:app --app-dir (Join-Path $PSScriptRoot "backend") --host 127.0.0.1 --port 8000
