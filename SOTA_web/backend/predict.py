import cv2
import json
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# ========================================
# 설정
# ========================================
MODEL_PATH = './model/sephiria_item_model.keras'
CLASSES_PATH = './model/classes.pickle'
IMG_SIZE = 128
BORDER_COLOR_BGR = (52, 32, 36)

# ========================================
# 유틸리티 함수
# ========================================


def non_max_suppression(boxes, overlap_thresh=0.3):
    """중복 슬롯 제거"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    pick = []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    area = boxes[:, 2] * boxes[:, 3]
    idxs = np.argsort(y1)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
    return boxes[pick].tolist()

# ========================================
# 슬롯 검출
# ========================================
def find_inventory_slots(image):
    """
    해상도 적응형 슬롯 검출
    - 테두리 색상 기반 마스크 생성
    - 형태학적 연산으로 내부 채우기
    - 하이브리드 윤곽선 검출 (테두리 + 반전)
    """
    print("\n=== 슬롯 검색 (적응형 하이브리드) ===")
    
    img_h, img_w = image.shape[:2]
    print(f"이미지 크기: {img_w}x{img_h}")
    
    # 이미지 크기 기반 슬롯 크기 추정
    estimated_slot_size = img_w * 0.08
    min_size = int(estimated_slot_size * 1.0)
    max_size = int(estimated_slot_size * 2.0)
    
    print(f"슬롯 크기 범위: {min_size} ~ {max_size}px")
    
    # 테두리 색상 마스크
    tolerance = 15
    lower = np.array([max(0, c - tolerance) for c in BORDER_COLOR_BGR])
    upper = np.array([min(255, c + tolerance) for c in BORDER_COLOR_BGR])
    
    mask = cv2.inRange(image, lower, upper)
    
    # 적응형 커널 크기
    kernel_small = max(3, int(img_w / 400))
    kernel_fill = max(5, int(img_w / 250))
    kernel_s = np.ones((kernel_small, kernel_small), np.uint8)
    kernel_f = np.ones((kernel_fill, kernel_fill), np.uint8)
    
    # 적응형 반복 횟수
    iterations_close = max(2, int(img_w / 600))
    iterations_fill = max(3, int(img_w / 400))
    
    print(f"커널: {kernel_small}x{kernel_small}, {kernel_fill}x{kernel_fill}")
    print(f"반복: close={iterations_close}, fill={iterations_fill}")
    
    # 노이즈 제거 및 내부 채우기
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_s, iterations=iterations_close)
    mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_f, iterations=iterations_fill)
    
    # 반전 마스크 (검은 부분 = 슬롯 내부)
    mask_inv = cv2.bitwise_not(mask_filled)
    
    candidates = []

    # 방법 A: 테두리 윤곽선 검출
    edges = cv2.Canny(mask, 50, 150)
    contours_A, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count_a = 0
    for cnt in contours_A:
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / float(h) if h > 0 else 0
        
        if min_size < w < max_size and min_size < h < max_size and 0.75 < ar < 1.25:
            candidates.append([x, y, w, h])
            count_a += 1

    # 방법 B: 반전 마스크 윤곽선 검출
    contours_B, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count_b = 0
    for cnt in contours_B:
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / float(h) if h > 0 else 0
        
        if min_size < w < max_size and min_size < h < max_size and 0.75 < ar < 1.25:
            candidates.append([x, y, w, h])
            count_b += 1

    print(f"🔹 방법 A(테두리): {count_a}개")
    print(f"🔹 방법 B(반전): {count_b}개")
    
    # 멀티스케일 폴백
    if len(candidates) == 0:
        print("⚠️ 슬롯을 못 찾았습니다.")
        print("\n멀티스케일 검색 시도...")
        
        for scale_factor in [0.6, 0.8, 1.0, 1.2, 1.5]:
            min_s = int(estimated_slot_size * scale_factor * 0.7)
            max_s = int(estimated_slot_size * scale_factor * 1.5)
            
            temp_candidates = []
            for cnt in contours_A + contours_B:
                x, y, w, h = cv2.boundingRect(cnt)
                if min_s < w < max_s and min_s < h < max_s:
                    temp_candidates.append([x, y, w, h])
            
            if len(temp_candidates) > 0:
                print(f"  스케일 {scale_factor:.1f}: {len(temp_candidates)}개 발견 → 사용")
                candidates = temp_candidates
                break

    # 중복 제거
    final_slots = non_max_suppression(candidates, overlap_thresh=0.3)
    
    if isinstance(final_slots, np.ndarray):
        final_slots = final_slots.tolist()
    
    # 정렬 (위→아래, 좌→우)
    row_gap = max(10, int(estimated_slot_size * 0.2))
    final_slots = sorted(final_slots, key=lambda s: (s[1] // row_gap, s[0]))
    
    print(f"✅ 최종: {len(final_slots)}개")
    

    return final_slots

# ========================================
# TTA (Test Time Augmentation)
# ========================================
def predict_with_tta(model, roi, class_names, tablet_classes, n_augmentations=5):
    predictions = []
    
    # 원본 예측
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    roi_input = preprocess_input(roi_resized.astype(np.float32))[None, ...]
    
    pred_original = model.predict(roi_input, verbose=0)[0]
    predictions.append(pred_original)
    
    top_class = class_names[np.argmax(pred_original)]
    is_tablet = top_class in tablet_classes
    
    for _ in range(n_augmentations - 1):
        aug = roi.copy()
        angle = np.random.uniform(-5, 5)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        brightness = np.random.randint(-15, 15)
        aug = np.clip(aug.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
        
        aug_rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
        aug_resized = cv2.resize(aug_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        aug_input = preprocess_input(aug_resized.astype(np.float32))[None, ...]
        predictions.append(model.predict(aug_input, verbose=0)[0])
        
    tta_type = "Small"
    
    return np.mean(predictions, axis=0), roi_resized, tta_type

# ========================================
# 메인 실행
# ========================================
print("=" * 60)
model = load_model(MODEL_PATH)
with open(CLASSES_PATH, 'rb') as f:
    class_names = pickle.load(f)
print("=" * 60)

# ========================================
# 아이템 분석
# ========================================
def predict_inventory(file):
    """메인 예측 함수 (app.py에서 호출)"""
    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    slots = find_inventory_slots(frame)
    
    results = []
    for idx, (x, y, w, h) in enumerate(slots):
        roi = frame[y:y+h, x:x+w]
        prediction = predict_with_tta(roi)
        
        top3_indices = np.argsort(prediction)[-3:][::-1]
        
        results.append({
            "idx": idx,
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "prediction": class_names[top3_indices[0]],
            "confidence": float(prediction[top3_indices[0]]),
            "top3": [class_names[i] for i in top3_indices]
        })
    
    return {
        "slots": results,
        "all_items": sorted(class_names)
    }