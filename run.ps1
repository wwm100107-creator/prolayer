# Chạy web tại http://127.0.0.1:8000 (chỉ máy này truy cập được).
$python = Join-Path $HOME ".cache\prolayer\venv\Scripts\python.exe"
& $python -m uvicorn app.main:app --app-dir (Join-Path $PSScriptRoot "backend") --host 127.0.0.1 --port 8000
