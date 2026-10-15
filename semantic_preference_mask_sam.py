import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import cv2

from transformers import CLIPModel, CLIPProcessor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from adaptive_semantic_weight import adaptive_semantic_weight


# ============================================================
# 1. 分类类别
# ============================================================

OBJECT_LABELS = [
    # human
    "person", "face", "human body", "hand", "hair", "portrait",

    # animals
    "dog", "cat", "bird", "horse", "fish", "animal",

    # nature
    "sky", "cloud", "mountain", "tree", "forest",
    "river", "water", "grass", "flower", "beach",

    # vehicles
    "car", "truck", "train", "boat", "bicycle", "bus",

    # buildings
    "building", "architecture", "cityscape", "house",

    # indoor
    "sofa", "chair", "table", "bed", "lamp",
    "wall", "floor", "window"
]

STYLE_LABELS = [
    "texture", "pattern", "background",
    "painting", "artwork", "brush stroke",
    "abstract region", "color region"
]

# ================== Global Cache ==================

_SAM_MODEL = None
_SAM_GENERATOR = None

SAM_MASK_CACHE = {}        # content_id -> masks
TEXT_FEAT_CACHE = {}       # label -> clip text feat
STYLE_IMG_FEAT_CACHE = {}  # style_id -> clip image feat


# ============================================================
# 2. 初始化 CLIP
# ============================================================

CLIP_PATH = "models/clip/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(CLIP_PATH).cuda()
clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH)


# ============================================================
# 3. 稳定版分类函数
# ============================================================

@torch.no_grad()
def stable_clip_classify(pil_img):
    """
    两阶段分类：
      1. 先在 OBJECT_LABELS 中找物体
      2. 若相似度都很低 → fallback 到 STYLE_LABELS
    """

    # ---------- Step A: Object-first 分类 ----------
    inputs = clip_processor(text=OBJECT_LABELS, images=pil_img,
                            return_tensors="pt", padding=True).to("cuda")

    logits = clip_model(**inputs).logits_per_image
    probs = logits.softmax(dim=1)[0]
    max_prob, idx = probs.max(dim=0)

    # 阈值：≤0.28 认为不可靠（经验值，非常稳）
    if max_prob.item() > 0.28:
        cls = OBJECT_LABELS[idx]
        return cls, max_prob.item(), False  # not fallback

    # ---------- Step B: fallback 到 style 类别 ----------
    inputs2 = clip_processor(text=STYLE_LABELS, images=pil_img,
                             return_tensors="pt", padding=True).to("cuda")
    logits2 = clip_model(**inputs2).logits_per_image
    probs2 = logits2.softmax(dim=1)[0]
    max_prob2, idx2 = probs2.max(dim=0)

    cls2 = STYLE_LABELS[idx2]
    return cls2, max_prob2.item(), True  # fallback=True

@torch.no_grad()
def stable_clip_classify_topk(
        pil_img,
        topk_obj=3,
        obj_thresh=0.18,
        topk_style=2
):
    """
    为风格图提取多语义集合（style_class_list）
    """
    style_classes = []

    # ---------- Object labels ----------
    inputs = clip_processor(
        text=OBJECT_LABELS,
        images=pil_img,
        return_tensors="pt",
        padding=True
    ).to("cuda")

    logits = clip_model(**inputs).logits_per_image
    probs = logits.softmax(dim=1)[0]

    vals, idxs = torch.topk(probs, k=topk_obj)

    for v, idx in zip(vals, idxs):
        if v.item() >= obj_thresh:
            style_classes.append(OBJECT_LABELS[idx])

    # ---------- Fallback: style labels ----------
    if len(style_classes) == 0:
        inputs2 = clip_processor(
            text=STYLE_LABELS,
            images=pil_img,
            return_tensors="pt",
            padding=True
        ).to("cuda")

        logits2 = clip_model(**inputs2).logits_per_image
        probs2 = logits2.softmax(dim=1)[0]

        vals2, idxs2 = torch.topk(probs2, k=topk_style)
        for idx in idxs2:
            style_classes.append(STYLE_LABELS[idx])

    # 至少保证一个
    if len(style_classes) == 0:
        style_classes.append("background")

    return list(set(style_classes))



# ============================================================
# 4. 稳定的语义偏好掩码
# ============================================================

def get_semantic_preference_mask_sam(
        content_pil,
        style_pil,
        device="cuda"
):
    print("\n========== Stable SAM–CLIP Semantic Mask ==========")

    img_np = np.array(content_pil)
    H, W = img_np.shape[:2]

    # ---------- 4.1 SAM ----------
    sam = sam_model_registry["vit_h"](
        checkpoint="models/sam_vit_h_4b8939.pth"
    ).cuda()

    mask_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=200
    )

    masks = mask_gen.generate(img_np)

    # ---------- 4.2 风格图主类别 ----------
    # style_class, sp, sf = stable_clip_classify(style_pil)
    # print(f"[Style Class] {style_class}  (prob={sp:.3f}, fallback={sf})")
    style_class_list = stable_clip_classify_topk(style_pil)

    print(f"[Style Classes] {style_class_list}")

    final_mask = torch.zeros((1,1,H,W), device=device)

    # ---------- 4.3 遍历所有 SAM 区域 ----------
    for i, m in enumerate(masks):
        mask = m["segmentation"]
        if mask.sum() < 200:
            continue

        x,y,w,h = cv2.boundingRect(mask.astype(np.uint8))
        crop = content_pil.crop((x, y, x+w, y+h))

        crop_class, cp, cf = stable_clip_classify(crop)

        # --- 自适应语义权重 ---
        weight = adaptive_semantic_weight(
            crop_pil=crop,
            style_pil=style_pil,
            crop_class=crop_class,
            style_class_list=style_class_list
        )

        print(f"Region {i:02d}:  class={crop_class:15s}, prob={cp:.3f}, "
              f"fallback={cf}, weight={weight:.3f}")

        mask_crop = torch.tensor(mask[y:y+h, x:x+w].astype(np.float32),
                                 device=device)[None,None]

        final_mask[:,:,y:y+h, x:x+w] += mask_crop * weight

    # 正规化
    final_mask = torch.clamp(final_mask, 0, 1)

    # ---------- 可视化 ----------
    mask_np = final_mask[0,0].detach().cpu().numpy()
    mask_norm = (mask_np * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
    cv2.imwrite("semantic_mask_heatmap.png", heatmap)

    print("Saved → semantic_mask_heatmap.png")
    print("====================================================\n")

    return final_mask
