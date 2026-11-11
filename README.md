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

Mở http://127.0.0.1:8000. Lần chạy đầu tải thêm model khoảng 5GB vào `~/.cache`.

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
| `backend/app/models.py` | Nạp SAM 2, ViTMatte, LaMa, Mask2Former, Depth Anything, CLIP, Stable Diffusion, EasyOCR; điều phối GPU |
| `backend/app/scene.py` | Nhận diện vật thể trong ảnh chụp, đặt tên layer, xếp thứ tự theo độ sâu |
| `backend/app/recognise.py` | Đặt tên cho vật chưa có tên bằng CLIP và danh sách khoảng 200 vật |
| `backend/app/complete.py` | Vẽ lại phần vật thể bị che bằng Stable Diffusion inpainting |
| `backend/app/pipeline.py` | Từ mask ra layer RGBA: trimap, matting, vá nền, khử nhiễm màu, phân rã cả ảnh |
| `backend/app/text.py` | Đọc chữ, khớp font, xóa chữ khỏi ảnh |
| `backend/app/shadow.py` | Tìm bóng đổ, tách thành layer multiply |
| `backend/app/vector.py` | Vector hóa layer màu phẳng bằng VTracer |
| `backend/app/main.py` | API HTTP, mã truy cập, CORS, quản lý phiên |
| `frontend/src/editor.ts` | Canvas Fabric.js, công cụ chọn, layer chữ |
| `frontend/src/exporter.ts` | Xuất PSD (ag-psd), SVG, PNG, ZIP |

Đổi model bằng biến môi trường `PROLAYER_SAM2`, `PROLAYER_VITMATTE`, `PROLAYER_PANOPTIC`, `PROLAYER_DEPTH`, `PROLAYER_RECOGNISER` và `PROLAYER_PAINTER`, ví dụ `facebook/sam2.1-hiera-small` nếu cần nhẹ hơn.

## Giới hạn hiện tại

- Ảnh thiết kế phẳng: tấm nền nằm dưới, hình vẽ nằm trên; hình chữ nhật hoặc elip bị khuyết một phần thì nằm dưới vật lấp chỗ khuyết đó. Ảnh chụp: xếp theo độ sâu, vật ở gần nằm trên.
- Một hình đều màu bị vật khác cắt thành nhiều mảnh được ghép lại thành một layer và lấp kín phần bị che. Hai hình giống hệt nhau cùng nằm trên một tấm nền (như hai bong bóng thoại) vẫn là hai layer.
- Ảnh minh họa hoặc thiết kế trên nền phẳng cho kết quả tốt nhất: mọi phần khác màu nền đều thành layer, nền được dựng lại sạch. Nhân vật có thể bị chia thành vài layer khi bị vật khác che ngang.
- Ảnh chụp được tách theo vật nguyên vẹn và đặt tên theo 133 loại vật quen thuộc (người, xe, cốc, mèo, bàn…). Vật không có trong danh sách chỉ được tách khi nổi hẳn lên so với chỗ nó nằm, như cái đĩa trên bàn; đồ vật xếp sát vào tường hoặc kệ thì vẫn nằm chung với nền. Mức "Chi tiết" chia tiếp mỗi vật thành từng phần. Vật bị bỏ sót có thể chọn bằng công cụ bên trái.
- Layer chữ luôn hiển thị đúng pixel gốc. Chỉ khi sửa nội dung trong ô "Nội dung chữ" mới đổi sang font gần giống nhất, được kéo giãn cho vừa khổ chữ cũ.
- OCR có thể thiếu dấu tiếng Việt, nhất là với font trang trí viết hoa; sửa trong ô nội dung.
- Phần bị che của một vật được vẽ lại bằng Stable Diffusion: vật được đặt riêng trên nền trơn, model vẽ tiếp theo tên vật, rồi SAM khoanh lại phần thuộc về vật. Tối đa 6 vật mỗi ảnh, mỗi vật thêm khoảng 2 giây. Phần đang nhìn thấy không bị sửa. Hình tròn, elip (đĩa, bánh xe), hình chữ nhật và các tấm nền phẳng được hoàn thiện bằng hình học vì chính xác hơn.
- Phần vẽ lại là phỏng đoán hợp lý, không phải sự thật: chi tiết bị khuất hoàn toàn như bàn tay hay khuôn mặt có thể sai. Vật bị cắt bởi mép ảnh chưa được vẽ tiếp.
- Tên layer chỉ được đặt khi model chắc chắn; vật không chắc giữ tên "Vật thể" hoặc "Phần tử".
- Vá nền với lỗ rất lớn trên ảnh chụp giữ đúng màu xung quanh nhưng có thể bị nhòe, vì LaMa không vẽ lại được chi tiết đã bị che.
- Bóng đổ dùng mô hình vật lý, không phải mạng đã huấn luyện: tốt trên nền phẳng hoặc ít họa tiết, và tự bỏ qua trên nền nhiều vân như gỗ.
- OCR có thể sai dấu tiếng Việt; bấm đúp layer chữ để sửa. Font chỉ khớp trong 15 kiểu phổ biến của Windows.
- SVG: hình phẳng thành vector, chữ giữ là chữ; ảnh chụp, bóng đổ và nền được nhúng dưới dạng ảnh.
