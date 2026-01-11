import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# 설정
IMG_PATH = './assets/artifacts/sky_blue_planet.png' # 테스트할 이미지 경로
MODEL_PATH = './CNN/sephiria_item_model.keras'
CLASSES_PATH = './CNN/classes.pickle'
IMG_SIZE = 128

# 로드
model = load_model(MODEL_PATH)
with open('./CNN/classes.pickle', 'rb') as f:
    classes = pickle.load(f)

# 이미지 읽기
img = cv2.imread(IMG_PATH, cv2.IMREAD_UNCHANGED)

# 투명 배경 처리 (PNG 파일 테스트 시 필요)
if img.shape[2] == 4:
    alpha = img[:, :, 3] / 255.0
    rgb = img[:, :, :3]
    bg = np.full(rgb.shape, (50, 50, 50), dtype=np.uint8) # 임시 배경색
    img = (rgb * alpha[:,:,None] + bg * (1-alpha[:,:,None])).astype(np.uint8)

# 전처리
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_input = img / 255.0
img_input = np.expand_dims(img_input, axis=0)

# 예측
pred = model.predict(img_input)
idx = np.argmax(pred)
print(f"결과: {classes[idx]} (확률: {pred[0][idx]*100:.2f}%)")

cv2.imshow('Test', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()