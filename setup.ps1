# Cài môi trường lần đầu. Tải khoảng 5-8GB (PyTorch CUDA; model tải khi chạy lần đầu).
$ErrorActionPreference = "Stop"
# Venv nằm ở đường dẫn ngắn: PyTorch có file vượt giới hạn 260 ký tự của Windows nếu đặt sâu.
$venv = Join-Path $HOME ".cache\prolayer\venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) { uv venv --python 3.12 $venv }
uv pip install --python $python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $python -r (Join-Path $PSScriptRoot "backend\requirements.txt")

Push-Location (Join-Path $PSScriptRoot "frontend")
try {
    npm install
    npm run build
} finally {
    Pop-Location
}
Write-Host "Xong. Chạy .\run.ps1 rồi mở http://127.0.0.1:8000"
