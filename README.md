# ProLayer

Web tách một ảnh phẳng thành các layer chỉnh sửa được: phần tử, chữ, bóng đổ và nền đã vá. Xuất ra PSD, SVG, PNG hoặc ZIP. Giao diện chạy trên Vercel, AI chạy trên GPU của máy qua server Python.

## Chạy trên máy

Cần Windows, GPU NVIDIA, Python 3.12, `uv` và Node.js.

```bash
powershell -ExecutionPolicy Bypass -File setup.ps1
```

```bash
powershell -ExecutionPolicy Bypass -File run.ps1
```

Mở http://127.0.0.1:8000. Lần chạy đầu tải thêm model khoảng 1,6GB vào `~/.cache`.

## Dùng qua web trên Vercel

Vercel chỉ chứa giao diện, vì backend cần GPU và PyTorch. Máy có GPU phải bật và chạy:

```bash
powershell -ExecutionPolicy Bypass -File run-public.ps1
```

Script mở backend ra Internet qua Tailscale Funnel và in mã truy cập, nhập mã đó trên web khi được hỏi. Mã nằm ở `~/.cache/prolayer/access-key.txt`, xóa file rồi chạy lại để đổi mã.

| Cấu hình | Nơi đặt | Ý nghĩa |
| --- | --- | --- |
| `VITE_API_BASE` | Biến môi trường trên Vercel | Địa chỉ HTTPS của backend |
| `PROLAYER_ACCESS_KEY` | Máy chạy backend | Bắt buộc với mọi thao tác dùng GPU |
| `PROLAYER_ORIGINS` | Máy chạy backend | Các địa chỉ web được phép gọi API |

Tắt đường hầm:

```bash
tailscale funnel reset
```

## Cấu trúc

| Phần | Vai trò |
| --- | --- |
| `backend/app/models.py` | Nạp SAM 2, ViTMatte, LaMa, EasyOCR; điều phối GPU |
| `backend/app/pipeline.py` | Từ mask ra layer RGBA: trimap, matting, vá nền, khử nhiễm màu, phân rã cả ảnh |
| `backend/app/text.py` | Đọc chữ, khớp font, xóa chữ khỏi ảnh |
| `backend/app/shadow.py` | Tìm bóng đổ, tách thành layer multiply |
| `backend/app/vector.py` | Vector hóa layer màu phẳng bằng VTracer |
| `backend/app/main.py` | API HTTP, mã truy cập, CORS, quản lý phiên |
| `frontend/src/editor.ts` | Canvas Fabric.js, công cụ chọn, layer chữ |
| `frontend/src/exporter.ts` | Xuất PSD (ag-psd), SVG, PNG, ZIP |

Đổi model bằng biến môi trường `PROLAYER_SAM2` và `PROLAYER_VITMATTE`, ví dụ `facebook/sam2.1-hiera-small` nếu cần nhẹ hơn.

## Giới hạn hiện tại

- Thứ tự layer tính theo diện tích: phần tử nhỏ nằm trên phần tử lớn.
- Vá nền với lỗ rất lớn trên ảnh chụp có thể bị nhòe hoặc tối (giới hạn của LaMa).
- Vật bị che chỉ được hoàn thiện khi gần lồi hoặc phần khuất nằm trọn bên trong.
- Bóng đổ dùng mô hình vật lý, không phải mạng đã huấn luyện: tốt trên nền phẳng hoặc ít họa tiết, và tự bỏ qua trên nền nhiều vân như gỗ.
- OCR có thể sai dấu tiếng Việt; bấm đúp layer chữ để sửa. Font chỉ khớp trong 15 kiểu phổ biến của Windows.
- SVG: hình phẳng thành vector, chữ giữ là chữ; ảnh chụp, bóng đổ và nền được nhúng dưới dạng ảnh.
