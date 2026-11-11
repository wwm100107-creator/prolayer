"""Names for cut-outs that Mask2Former has no class for.

CLIP compares each cut-out, shown alone on a plain backdrop, with a list of
everyday objects and of the things flat artwork is made of. A name is given
only when one word clearly wins: a wrong name is worse than none, both in the
layer list and as the hint for redrawing a hidden part.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .models import engine

log = logging.getLogger("prolayer")

# English for CLIP and Stable Diffusion, Vietnamese for the layer list.
WORDS = {
    "person": "Người", "man": "Người đàn ông", "woman": "Người phụ nữ", "child": "Em bé", "face": "Khuôn mặt",
    "cartoon character": "Nhân vật",
    "hand": "Bàn tay", "cat": "Mèo", "dog": "Chó", "bird": "Chim", "fish": "Cá", "horse": "Ngựa", "cow": "Bò",
    "butterfly": "Bướm", "rabbit": "Thỏ", "bear": "Gấu", "elephant": "Voi", "lion": "Sư tử", "monkey": "Khỉ",
    "tree": "Cây", "potted plant": "Chậu cây", "leaf": "Lá cây", "flower": "Hoa", "grass": "Cỏ", "mushroom": "Nấm",
    "cloud": "Đám mây", "sun": "Mặt trời", "moon": "Mặt trăng", "star": "Ngôi sao", "mountain": "Núi",
    "rock": "Đá", "fire": "Lửa", "water drop": "Giọt nước", "lightning bolt": "Tia sét", "rainbow": "Cầu vồng",
    "cup": "Cốc", "saucer": "Đĩa lót", "plate": "Đĩa", "bowl": "Bát", "bottle": "Chai", "drinking glass": "Ly",
    "spoon": "Thìa", "fork": "Nĩa", "knife": "Dao", "teapot": "Ấm trà", "pan": "Chảo", "jar": "Lọ",
    "apple": "Táo", "banana": "Chuối", "orange": "Cam", "bread": "Bánh mì", "cake": "Bánh ngọt", "pizza": "Pizza",
    "egg": "Trứng", "ice cream": "Kem", "hamburger": "Bánh hamburger", "candy": "Kẹo",
    "table": "Bàn", "chair": "Ghế", "sofa": "Sofa", "bed": "Giường", "shelf": "Kệ", "lamp": "Đèn",
    "door": "Cửa", "window": "Cửa sổ", "vase": "Bình hoa", "pillow": "Gối", "box": "Hộp", "basket": "Giỏ",
    "bucket": "Xô", "ladder": "Thang", "umbrella": "Ô", "candle": "Nến", "mirror": "Gương", "picture frame": "Khung ảnh",
    "computer monitor": "Màn hình", "laptop": "Laptop", "keyboard": "Bàn phím", "computer mouse": "Chuột máy tính",
    "phone": "Điện thoại", "tablet": "Máy tính bảng", "camera": "Máy ảnh", "television": "Tivi",
    "headphones": "Tai nghe", "microphone": "Micro", "speaker": "Loa", "printer": "Máy in", "robot": "Robot",
    "book": "Sách", "notebook": "Sổ tay", "sheet of paper": "Tờ giấy", "clipboard": "Bảng kẹp giấy",
    "pen": "Bút", "pencil": "Bút chì", "paintbrush": "Cọ vẽ", "scissors": "Kéo", "ruler": "Thước",
    "envelope": "Phong bì", "folder": "Bìa hồ sơ", "calendar": "Lịch", "clock": "Đồng hồ", "briefcase": "Cặp tài liệu",
    "whiteboard": "Bảng trắng", "presentation board": "Bảng trình chiếu", "tripod": "Chân máy", "easel": "Giá vẽ",
    "speech bubble": "Bong bóng thoại", "bar chart": "Biểu đồ cột", "pie chart": "Biểu đồ tròn",
    "line chart": "Biểu đồ đường", "arrow": "Mũi tên", "button": "Nút bấm", "logo": "Logo", "icon": "Biểu tượng",
    "banner": "Băng rôn", "ribbon": "Dải ruy băng", "badge": "Huy hiệu", "label": "Nhãn", "check mark": "Dấu tích",
    "heart": "Trái tim", "light bulb": "Bóng đèn", "gear": "Bánh răng", "magnifying glass": "Kính lúp",
    "map pin": "Ghim bản đồ", "globe": "Quả địa cầu", "map": "Bản đồ", "flag": "Cờ", "trophy": "Cúp",
    "medal": "Huy chương", "crown": "Vương miện", "gift box": "Hộp quà", "balloon": "Bóng bay", "bell": "Chuông",
    "key": "Chìa khóa", "lock": "Ổ khóa", "shield": "Khiên", "target": "Bia ngắm", "puzzle piece": "Mảnh ghép",
    "coin": "Đồng xu", "banknote": "Tiền giấy", "credit card": "Thẻ ngân hàng", "wallet": "Ví",
    "shopping cart": "Xe đẩy hàng", "shopping bag": "Túi mua sắm", "price tag": "Thẻ giá",
    "car": "Ô tô", "bus": "Xe buýt", "truck": "Xe tải", "bicycle": "Xe đạp", "motorcycle": "Xe máy",
    "airplane": "Máy bay", "rocket": "Tên lửa", "boat": "Thuyền", "train": "Tàu hỏa", "wheel": "Bánh xe",
    "house": "Ngôi nhà", "building": "Tòa nhà", "bridge": "Cầu", "tent": "Lều", "fence": "Hàng rào",
    "road sign": "Biển báo", "street light": "Đèn đường", "bench": "Ghế dài",
    "helmet": "Mũ bảo hiểm", "hat": "Mũ", "glasses": "Kính", "shirt": "Áo", "dress": "Váy", "trousers": "Quần",
    "shoe": "Giày", "backpack": "Ba lô", "handbag": "Túi xách", "watch": "Đồng hồ đeo tay", "ring": "Nhẫn",
    "necktie": "Cà vạt", "glove": "Găng tay", "scarf": "Khăn quàng",
    "ball": "Quả bóng", "guitar": "Đàn guitar", "piano": "Đàn piano", "drum": "Trống", "dice": "Xúc xắc",
    "toy": "Đồ chơi", "teddy bear": "Gấu bông", "kite": "Diều", "hammer": "Búa", "wrench": "Cờ lê",
    "syringe": "Ống tiêm", "pill": "Viên thuốc", "stethoscope": "Ống nghe", "toothbrush": "Bàn chải đánh răng",
    # Not objects: these only keep shapes and scraps from being given a name.
    "plain rectangle": "", "plain circle": "", "abstract shape": "", "shadow": "", "blank background": "",
    "text": "", "pattern": "",
}
TEMPLATES = ("a photo of a {}", "an illustration of a {}")
CROP_SIDE = 224
BACKDROP = 128
# The winning word must hold this much of the probability, and lead by this much.
MIN_SHARE = 0.55
MIN_LEAD = 0.15
# Words for one kind of thing split the probability between them, so they count together.
KINDS = (
    ("person", "man", "woman", "child", "cartoon character"),
    ("saucer", "plate", "bowl"),
    ("whiteboard", "presentation board"),
)


def _cutout(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The object alone on grey, centred in a square."""
    ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    x0, y0, x1, y1 = int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1
    crop = np.where(mask[y0:y1, x0:x1, None], rgb[y0:y1, x0:x1], BACKDROP).astype(np.uint8)
    side = int(max(x1 - x0, y1 - y0) * 1.15) + 2
    square = np.full((side, side, 3), BACKDROP, np.uint8)
    top, left = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
    square[top : top + y1 - y0, left : left + x1 - x0] = crop
    return cv2.resize(square, (CROP_SIDE, CROP_SIDE), interpolation=cv2.INTER_AREA)


def recognise(rgb: np.ndarray, masks: list[np.ndarray]) -> list[tuple[str, str] | None]:
    """For each mask, (English word, Vietnamese name), or None when unsure."""
    if not masks:
        return []
    words = list(WORDS)
    shares = engine.match_words([_cutout(rgb, mask) for mask in masks], words, TEMPLATES)
    kind_of = {word: kind for kind in KINDS for word in kind}
    out: list[tuple[str, str] | None] = []
    for share in shares:
        best = int(np.argmax(share))
        word = words[best]
        kind = kind_of.get(word, (word,))
        together = float(sum(share[words.index(other)] for other in kind))
        rival = max(float(share[i]) for i, other in enumerate(words) if other not in kind)
        log.debug("recognise %s %.2f, rival %.2f", word, together, rival)
        sure = together >= MIN_SHARE and together - rival >= MIN_LEAD
        out.append((word, WORDS[word]) if sure and WORDS[word] else None)
    return out
