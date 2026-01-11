import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2

# --- 1. 설정 ---
BASE_PATH = './assets'
IMG_SIZE = 128
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
AUGMENT_COUNT = 100

# [수정] 배경을 이미지 파일로 변경
FOLDER_CONFIG = {
    'artifacts': './CNN/slot_artifacts.png',
    'tablets': './CNN/slot_tablets.png',
    'empty': './CNN/slot_empty.png'
}

# --- 2. 배경 이미지 로드 및 캐싱 ---
def load_background(bg_path, target_size):
    """배경 이미지를 로드하고 크기 조정"""
    if not os.path.exists(bg_path):
        print(f"경고: 배경 이미지 {bg_path}를 찾을 수 없습니다.")
        if 'artifacts' in bg_path:
            return np.full((target_size, target_size, 3), (59, 58, 73), dtype=np.uint8)
        elif 'empty' in bg_path:
            return np.full((target_size, target_size, 3), (98, 49, 68), dtype=np.uint8)
        else:
            return np.full((target_size, target_size, 3), (59, 59, 88), dtype=np.uint8)
    
    bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if bg_img is None:
        print(f"경고: 배경 이미지 {bg_path} 로드 실패")
        if 'artifacts' in bg_path:
            return np.full((target_size, target_size, 3), (59, 58, 73), dtype=np.uint8)
        elif 'empty' in bg_path:
            return np.full((target_size, target_size, 3), (98, 49, 68), dtype=np.uint8)
        else:
            return np.full((target_size, target_size, 3), (59, 59, 88), dtype=np.uint8)
    
    bg_resized = cv2.resize(bg_img, (target_size, target_size))
    return bg_resized

background_cache = {}

# --- 3. 강력한 데이터 증강 함수 ---
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
        angle = np.random.uniform(-180, 180)
    else:
        angle = np.random.uniform(-5, 5)
    
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 2. 밝기 변형
    value = np.random.randint(-30, 30)
    aug_img = np.clip(aug_img.astype(np.int16) + value, 0, 255).astype(np.uint8)

    # 3. [추가] 이미지 품질 저하 (해상도 낮은 스크린샷 대응)
    if np.random.rand() > 0.3:  # 70% 확률
        degradation_type = np.random.choice(['blur', 'downsample', 'jpeg', 'combined'])
        
        if degradation_type == 'blur':
            kernel_size = np.random.choice([3, 5])
            aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)
        
        elif degradation_type == 'downsample':
            scale = np.random.uniform(0.5, 0.8)
            small_h, small_w = int(h * scale), int(w * scale)
            small = cv2.resize(aug_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
            aug_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        elif degradation_type == 'jpeg':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(60, 85)]
            _, encimg = cv2.imencode('.jpg', aug_img, encode_param)
            aug_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        
        elif degradation_type == 'combined':
            scale = np.random.uniform(0.6, 0.9)
            small_h, small_w = int(h * scale), int(w * scale)
            small = cv2.resize(aug_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
            aug_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            
            if np.random.rand() > 0.5:
                aug_img = cv2.GaussianBlur(aug_img, (3, 3), 0)

    # 4. UI 그리기
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

# --- 4. 데이터 로드 ---
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

            # 투명 배경 합성
            if img.shape[2] == 4:
                if bg_path not in background_cache:
                    background_cache[bg_path] = load_background(bg_path, img.shape[0])
                
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

            # 증강
            for _ in range(AUGMENT_COUNT - 1):
                aug = augment_image(base_img, folder_type=folder_name)
                aug_rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
                aug_resized = cv2.resize(aug_rgb, (IMG_SIZE, IMG_SIZE))
                data.append(aug_resized)
                labels.append(label)
    
    # 폴더별 로그
    if folder_name == 'tablets':
        print(f"-> {folder_name}: 360도 회전, UI 없음, 배경: {bg_path}")
    else:
        print(f"-> {folder_name}: 거의 회전 없음, 배경: {bg_path}")

data = np.array(data) / 255.0
labels = np.array(labels)

print(f"\n-> 최종 데이터 개수: {len(data)}장")
print(f"-> 클래스 개수: {len(np.unique(labels))}개")

# --- 5. 학습 준비 ---
le = LabelEncoder()
labels_enc = le.fit_transform(labels)
labels_onehot = to_categorical(labels_enc)

with open('./CNN/classes.pickle', 'wb') as f:
    pickle.dump(le.classes_, f)

# 석판 클래스도 저장
tablet_classes = []
tablets_folder = os.path.join(BASE_PATH, 'tablets')
if os.path.exists(tablets_folder):
    for filename in os.listdir(tablets_folder):
        if filename.lower().endswith(VALID_EXTENSIONS):
            class_name = os.path.splitext(filename)[0]
            tablet_classes.append(class_name)

with open('./CNN/tablet_classes.pickle', 'wb') as f:
    pickle.dump(tablet_classes, f)
print(f"석판 클래스 {len(tablet_classes)}개 저장")

X_train, X_val, y_train, y_val = train_test_split(data, labels_onehot, test_size=0.2, random_state=42)

# --- 6. 모델 설계 (개선: 더 큰 모델 + Fine-tuning) ---
print("\n" + "=" * 60)
print("모델 구성 중...")
print("=" * 60)

# [개선] MobileNetV2 alpha 증가 (1.0 → 1.4, 더 큰 모델)
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3), 
    alpha=1.4  # 1.0 → 1.4 (모델 크기 40% 증가)
)
base_model.trainable = False  # 처음엔 동결

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),  # 배치 정규화
    layers.Dense(512, activation='relu'),  # 256 → 512
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),  # 추가 레이어
    layers.Dropout(0.3),
    layers.Dense(len(le.classes_), activation='softmax', dtype='float32')
])

print(f"✅ 모델 구성 완료 (MobileNetV2 alpha=1.4)")
print(f"   총 파라미터: {model.count_params():,}개")

# --- 7. 콜백 설정 ---
# Early Stopping
early_stop_stage1 = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Learning Rate Scheduler
reduce_lr_stage1 = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Model Checkpoint (최고 모델 저장)
checkpoint_stage1 = ModelCheckpoint(
    'best_model_stage1.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# --- 8. 1단계 학습 (Feature Extractor) ---
print("\n" + "=" * 60)
print("1단계 학습 시작 (Feature Extractor)")
print("=" * 60)

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history1 = model.fit(
    X_train, y_train,
    epochs=30,  # Early stopping이 알아서 멈춤
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop_stage1, reduce_lr_stage1, checkpoint_stage1],
    verbose=1
)

print(f"✅ 1단계 완료 (최고 검증 정확도: {max(history1.history['val_accuracy'])*100:.2f}%)")

# --- 9. 2단계 학습 (Fine-tuning) ---
print("\n" + "=" * 60)
print("2단계 학습 시작 (Fine-tuning)")
print("=" * 60)

# MobileNetV2 상위 레이어 해동
base_model.trainable = True
unfrozen_layers = 0
for layer in base_model.layers:
    if unfrozen_layers < len(base_model.layers) - 30:
        layer.trainable = False
    else:
        layer.trainable = True
        unfrozen_layers += 1

print(f"✅ {unfrozen_layers}개 레이어 학습 가능")

# 낮은 learning rate로 재컴파일
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),  # 10배 낮춤
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 2단계 콜백
early_stop_stage2 = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_stage2 = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-8,
    verbose=1
)

checkpoint_stage2 = ModelCheckpoint(
    'best_model_stage2.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history2 = model.fit(
    X_train, y_train,
    epochs=40,  # Fine-tuning은 더 오래
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop_stage2, reduce_lr_stage2, checkpoint_stage2],
    verbose=1
)

print(f"✅ 2단계 완료 (최고 검증 정확도: {max(history2.history['val_accuracy'])*100:.2f}%)")

# --- 10. 최종 모델 저장 ---
model.save('./CNN/sephiria_item_model.keras')
print("\n" + "=" * 60)
print("학습 완료!")
print("=" * 60)
print("저장된 파일:")
print("  - ./CNN/sephiria_item_model.keras (최종 모델)")
print("  - ./CNN/best_model_stage1.keras (1단계 최고 모델)")
print("  - ./CNN/best_model_stage2.keras (2단계 최고 모델)")
print("  - ./CNN/classes.pickle")
print("  - ./CNN/tablet_classes.pickle")

# --- 11. 학습 곡선 시각화 ---
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 정확도
stage1_epochs = len(history1.history['accuracy'])
axes[0].plot(history1.history['accuracy'], label='Stage 1 Train', linewidth=2)
axes[0].plot(history1.history['val_accuracy'], label='Stage 1 Val', linewidth=2)
axes[0].plot(range(stage1_epochs, stage1_epochs + len(history2.history['accuracy'])),
             history2.history['accuracy'], label='Stage 2 Train', linewidth=2)
axes[0].plot(range(stage1_epochs, stage1_epochs + len(history2.history['val_accuracy'])),
             history2.history['val_accuracy'], label='Stage 2 Val', linewidth=2)
axes[0].axvline(x=stage1_epochs, color='r', linestyle='--', label='Fine-tuning Start')
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 손실
axes[1].plot(history1.history['loss'], label='Stage 1 Train', linewidth=2)
axes[1].plot(history1.history['val_loss'], label='Stage 1 Val', linewidth=2)
axes[1].plot(range(stage1_epochs, stage1_epochs + len(history2.history['loss'])),
             history2.history['loss'], label='Stage 2 Train', linewidth=2)
axes[1].plot(range(stage1_epochs, stage1_epochs + len(history2.history['val_loss'])),
             history2.history['val_loss'], label='Stage 2 Val', linewidth=2)
axes[1].axvline(x=stage1_epochs, color='r', linestyle='--', label='Fine-tuning Start')
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./CNN/training_history.png', dpi=150)
print("\n✅ 학습 곡선 저장: ./CNN/training_history.png")

# 최종 성능 출력
final_val_acc = history2.history['val_accuracy'][-1]
best_val_acc = max(history2.history['val_accuracy'])
print(f"\n최종 검증 정확도: {final_val_acc*100:.2f}%")
print(f"최고 검증 정확도: {best_val_acc*100:.2f}%")