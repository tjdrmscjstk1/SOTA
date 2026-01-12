import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# ========================================
# 설정
# ========================================
SCREENSHOT_PATH = './CNN/test6.png'
MODEL_PATH = './CNN/sephiria_item_model.keras'
CLASSES_PATH = './CNN/classes.pickle'
IMG_SIZE = 128
BORDER_COLOR_BGR = (52, 32, 36)

os.makedirs('./debug_test_rois', exist_ok=True)
os.makedirs('./debug_test_processed', exist_ok=True)

# ========================================
# 유틸리티 함수
# ========================================
def load_tablet_classes(tablets_folder='./assets/tablets'):
    """석판 클래스 목록 로드"""
    tablet_classes = set()
    if os.path.exists(tablets_folder):
        for filename in os.listdir(tablets_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                class_name = os.path.splitext(filename)[0]
                tablet_classes.add(class_name)
    print(f"✅ 석판: {len(tablet_classes)}개")
    return tablet_classes

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
    cv2.imwrite('./CNN/debug_mask_raw.png', mask)
    
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
    cv2.imwrite('./CNN/debug_mask_filled.png', mask_filled)
    
    # 반전 마스크 (검은 부분 = 슬롯 내부)
    mask_inv = cv2.bitwise_not(mask_filled)
    cv2.imwrite('./CNN/debug_mask_inv.png', mask_inv)
    
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
        
        if len(candidates) == 0:
            print("-> 디버그 파일:")
            print("   debug_mask_raw.png")
            print("   debug_mask_filled.png")
            print("   debug_mask_inv.png")
            return []

    # 중복 제거
    final_slots = non_max_suppression(candidates, overlap_thresh=0.3)
    
    if isinstance(final_slots, np.ndarray):
        final_slots = final_slots.tolist()
    
    # 정렬 (위→아래, 좌→우)
    row_gap = max(10, int(estimated_slot_size * 0.2))
    final_slots = sorted(final_slots, key=lambda s: (s[1] // row_gap, s[0]))
    
    print(f"✅ 최종: {len(final_slots)}개")

    # 디버그 시각화
    vis = image.copy()
    for i, (x, y, w, h) in enumerate(final_slots):
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font_scale = max(0.3, img_w / 1000)
        cv2.putText(vis, str(i), (x+5, y+int(h*0.3)), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
    
    cv2.imwrite('./CNN/debug_slots_hybrid.png', vis)
    print("-> debug_slots_hybrid.png 저장")

    return final_slots

# ========================================
# TTA (Test Time Augmentation)
# ========================================
def predict_with_tta(model, roi, class_names, tablet_classes, n_augmentations=5):
    """
    증강 기반 예측으로 정확도 향상
    - 석판: 90도 단위 회전
    - 일반 아이템: 미세 회전 및 밝기 변화
    """
    predictions = []
    
    # 원본 예측
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    roi_input = preprocess_input(roi_resized.astype(np.float32))[None, ...]
    
    pred_original = model.predict(roi_input, verbose=0)[0]
    predictions.append(pred_original)
    
    top_class = class_names[np.argmax(pred_original)]
    is_tablet = top_class in tablet_classes
    
    # 증강 예측
    if is_tablet:
        # 석판: 90도 단위 회전
        for angle in [90, 180, 270]:
            h, w = roi.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            aug = cv2.warpAffine(roi, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            brightness = np.random.randint(-20, 20)
            aug = np.clip(aug.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
            
            aug_rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
            aug_resized = cv2.resize(aug_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            aug_input = preprocess_input(aug_resized.astype(np.float32))[None, ...]
            predictions.append(model.predict(aug_input, verbose=0)[0])
        
        tta_type = "Rotation"
    else:
        # 일반 아이템: 미세 변화
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
tablet_classes = load_tablet_classes('./assets/tablets')
print("=" * 60)

frame = cv2.imread(SCREENSHOT_PATH)
if frame is None:
    print("❌ 이미지 없음")
    exit()

print(f"\n이미지: {frame.shape[1]}x{frame.shape[0]}")

# 슬롯 검출
slots = find_inventory_slots(frame)

if len(slots) == 0:
    print("❌ 슬롯 없음")
    exit()

print(f"\n✅ {len(slots)}개 슬롯 발견")

# ========================================
# 아이템 분석
# ========================================
output_frame = frame.copy()

print("\n" + "=" * 60)
print("분석 중...")
print("=" * 60)

for idx, (x, y, w, h) in enumerate(slots):
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        continue
    
    cv2.imwrite(f'./CNN/debug_test_rois/slot_{idx}.png', roi)
    
    try:
        prediction, roi_final, tta_type = predict_with_tta(model, roi, class_names, tablet_classes)
        cv2.imwrite(f'./CNN/debug_test_processed/slot_{idx}_final.png', 
                    cv2.cvtColor(roi_final, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"[{idx}] ❌ 오류: {e}")
        continue
    
    # 결과 처리
    pred_idx = np.argmax(prediction)
    confidence = prediction[pred_idx]
    label = class_names[pred_idx]

    # 출력 (50% 미만일 때만 Top-3)
    if confidence < 0.5:
        top3 = np.argsort(prediction)[-3:][::-1]
        print(f"[{idx}] {label} ({confidence*100:.0f}%) [{tta_type}]")
        print(f"     Top-3: ", end="")
        for ti in top3:
            print(f"{class_names[ti]}({prediction[ti]*100:.0f}%) ", end="")
        print()
    else:
        print(f"[{idx}] {label} ({confidence*100:.0f}%) [{tta_type}]")
    
    # 시각화
    color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
    cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
    cv2.rectangle(output_frame, (x, y-20), (x+100, y), (0, 0, 0), -1)
    cv2.putText(output_frame, label[:12], (x+2, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

cv2.imwrite('./CNN/inventory_result.png', output_frame)
print("\n✅ 분석 완료: inventory_result.png")

cv2.imshow('Result', output_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()