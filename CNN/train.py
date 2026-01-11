import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. 설정 ---
BASE_PATH = '../assets'
IMG_SIZE = 128  # [수정 1] 해상도를 128로 올림 (정확도 상승 요인)
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
AUGMENT_COUNT = 150 # [수정 2] 이미지 1장을 150장으로 불림 (필수!)

FOLDER_CONFIG = {
    'artifacts': './slot_artifacts.png',
    'tablets': './slot_tablets.png',
    'empty': './slot_empty.png'
}

# --- 2. 배경 이미지 로드 및 캐싱 ---
def load_background(bg_path, target_size):
    """배경 이미지를 로드하고 크기 조정"""
    if not os.path.exists(bg_path):
        print(f"경고: 배경 이미지 {bg_path}를 찾을 수 없습니다. 기본 회색 배경 사용")
        # artifacts는 (59, 58, 73), tablets는 (59, 59, 88)
        if 'artifacts' in bg_path:
            return np.full((target_size, target_size, 3), (59, 58, 73), dtype=np.uint8)
        elif 'empty' in bg_path:
            return np.full((target_size, target_size, 3), (98, 49, 68), dtype=np.uint8)
        else:
            return np.full((target_size, target_size, 3), (59, 59, 88), dtype=np.uint8)
    
    bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if bg_img is None:
        print(f"경고: 배경 이미지 {bg_path} 로드 실패. 기본 회색 배경 사용")
        if 'artifacts' in bg_path:
            return np.full((target_size, target_size, 3), (59, 58, 73), dtype=np.uint8)
        elif 'empty' in bg_path:
            return np.full((target_size, target_size, 3), (98, 49, 68), dtype=np.uint8)
        else:
            return np.full((target_size, target_size, 3), (59, 59, 88), dtype=np.uint8)
    
    # 배경을 target_size에 맞게 리사이즈
    bg_resized = cv2.resize(bg_img, (target_size, target_size))
    return bg_resized

# 배경 이미지 미리 로드 (원본 이미지 크기로)
background_cache = {}
# --- 2. 강력한 데이터 증강 함수 (폴더별 다른 전략) ---
def augment_image(image, folder_type='artifacts'):
    """
    folder_type='tablets': 석판 (360도 회전, UI 없음)
    folder_type='artifacts': 아티팩트 (회전 없음, UI 있음)
    folder_type='empty': 빈 슬롯 (회전 없음, UI 없음)
    """
    aug_img = image.copy()
    h, w = aug_img.shape[:2]

    # 1. 폴더별 회전 전략
    if folder_type == 'tablets':
        angle = np.random.choice([-180, -90, 0, 90, 180])
    else:
        angle = np.random.uniform(-1, 1)
    
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 2. 밝기 변형
    value = np.random.randint(-30, 30)
    aug_img = np.clip(aug_img.astype(np.int16) + value, 0, 255).astype(np.uint8)

    # 3. [추가] 이미지 품질 저하 (해상도 낮은 스크린샷 대응)
    quality_degradation = np.random.rand()
    
    if quality_degradation > 0.5:  # 50% 확률로 품질 저하
        degradation_type = np.random.choice(['blur', 'downsample', 'combined'])
        
        if degradation_type == 'blur':
            # 가우시안 블러 (스크린샷 압축 효과)
            kernel_size = 3
            aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)
        
        elif degradation_type == 'downsample':
            # 다운샘플링 후 업샘플링 (해상도 저하)
            scale = np.random.uniform(0.7, 0.9)  # 70~90%로 축소
            small_h, small_w = int(h * scale), int(w * scale)
            
            # 축소
            small = cv2.resize(aug_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
            # 다시 확대 (뭉개짐 효과)
            aug_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        elif degradation_type == 'combined':
            # 여러 효과 조합 (가장 현실적)
            # 1) 다운샘플링
            scale = np.random.uniform(0.7, 0.9)
            small_h, small_w = int(h * scale), int(w * scale)
            small = cv2.resize(aug_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
            aug_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 2) 약간의 블러
            if np.random.rand() > 0.5:
                aug_img = cv2.GaussianBlur(aug_img, (3, 3), 0)

    # 4. UI 그리기 (기존)
    if folder_type == 'artifacts':
        if np.random.rand() > 0.7:
            box_size = np.random.randint(15, 25)
            pt1 = (w - box_size, h - box_size)
            pt2 = (w, h)
            cv2.rectangle(aug_img, pt1, pt2, (20, 20, 20), -1) 
            cv2.putText(aug_img, str(np.random.randint(1, 9)), (w - box_size + 5, h - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if folder_type == 'empty':
        if np.random.rand() > 0.4:
            font_scale = np.random.uniform(0.6, 1.0)
            text_x = np.random.randint(0, 10)
            text_y = np.random.randint(10, 20)
            cv2.putText(aug_img, "+1", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return aug_img.astype(np.uint8)

# --- 3. 데이터 로드 ---
data = []
labels = []

print(f"이미지 로드 및 {AUGMENT_COUNT}배 증강 시작 (해상도: {IMG_SIZE}x{IMG_SIZE})...")

for folder_name, bg_path in FOLDER_CONFIG.items():
    folder_path = os.path.join(BASE_PATH, folder_name)
    if not os.path.exists(folder_path): 
        print(f"경고: {folder_path} 폴더가 없습니다.")
        continue
    
    print(f"\n=== {folder_name} 폴더 처리 중 ===")
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(VALID_EXTENSIONS):
            label = os.path.splitext(filename)[0]
            
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None: 
                print(f"  경고: {filename} 로드 실패")
                continue

            # [핵심 수정] 투명 배경 합성 - 배경 이미지 사용
            if img.shape[2] == 4:
                # 배경 이미지 로드 (원본 이미지 크기에 맞춤)
                if bg_path not in background_cache:
                    background_cache[bg_path] = load_background(bg_path, img.shape[0])
                
                # 배경을 현재 이미지 크기에 맞게 조정
                bg = cv2.resize(background_cache[bg_path], (img.shape[1], img.shape[0]))
                
                alpha = img[:, :, 3]
                rgb = img[:, :, :3]
                alpha_f = alpha[:, :, np.newaxis] / 255.0
                
                base_img = (rgb * alpha_f + bg * (1 - alpha_f)).astype(np.uint8)
            else:
                base_img = img

            # 원본 1장 저장
            img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            data.append(img_resized)
            labels.append(label)

            # [수정] 증강 시 폴더 정보 전달
            for _ in range(AUGMENT_COUNT - 1):
                aug = augment_image(base_img, folder_type=folder_name)  # [변경!]
                aug_rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
                aug_resized = cv2.resize(aug_rgb, (IMG_SIZE, IMG_SIZE))
                data.append(aug_resized)
                labels.append(label)
    
    # [추가] 폴더별 로그
    if folder_name == 'tablets':
        print(f"-> {folder_name}: 360도 회전 적용, UI 없음, 배경: {bg_path}")
    else:  # empty
        print(f"-> {folder_name}: 거의 회전 없음, UI 없음, 배경: {bg_path}")

data = preprocess_input(np.array(data, dtype=np.float32))
labels = np.array(labels)

print(f"\n-> 최종 데이터 개수: {len(data)}장")
print(f"-> 클래스 개수: {len(np.unique(labels))}개")

# [추가] 학습 데이터 샘플 저장
import random
os.makedirs('./debug_train_samples', exist_ok=True)

unique_labels = np.unique(labels)
for label in unique_labels:
    indices = np.where(labels == label)[0]
    sample_idx = random.choice(indices)
    
    # 정규화 전 이미지 복원 (0~255)
    img_to_save = (data[sample_idx] * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(f'./debug_train_samples/{label}_sample.png', img_bgr)

print("학습 샘플 이미지 저장 완료: ./debug_train_samples/")

# --- 4. 학습 준비 ---
le = LabelEncoder()
labels_enc = le.fit_transform(labels)
labels_onehot = to_categorical(labels_enc)

with open('classes.pickle', 'wb') as f:
    pickle.dump(le.classes_, f)

# 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(data, labels_onehot, test_size=0.2, random_state=42)

# --- 5. 모델 설계 (GPU 사용 설정) ---
# MobileNetV2 로드 (알파값은 모델 크기 조절, 1.0이 기본)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), alpha=1.0)
base_model.trainable = False 

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base_model,
    layers.GlobalAveragePooling2D(), # Flatten 대신 이걸 쓰는 게 MobileNet엔 더 좋습니다
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(le.classes_), activation='softmax', dtype='float32') # mixed_precision 쓸 때 dtype 명시
])
# 학습률 설정
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 6. 학습 실행 ---
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("학습 시작...")
history = model.fit(X_train, y_train, 
                    epochs=15, 
                    batch_size=32, 
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop])

model.save('sephiria_item_model.keras')
print("저장 완료!")