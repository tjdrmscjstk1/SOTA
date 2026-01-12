import json
import os
from typing import List, Dict, Optional, Set
import sys

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..models.tablet import Tablet, PositionEffect, ShapeEffect, Rarity, Restriction


class CNNBridge:
    """
    CNN 감지 시스템과의 통합 브릿지

    워크플로우:
    1. 사용자가 스크린샷 제공
    2. CNN이 인벤토리의 석판/아이템 감지
    3. 브릿지가 감지 결과를 Tablet 객체로 변환
    4. GA 최적화기가 최적 배치 탐색
    """

    def __init__(self,
                 tablet_json_path: str = None,
                 classes_pickle_path: str = None):
        if tablet_json_path is None:
            tablet_json_path = os.path.join(PROJECT_ROOT, 'tablet.json')
        if classes_pickle_path is None:
            classes_pickle_path = os.path.join(PROJECT_ROOT, 'CNN', 'classes.pickle')

        self.tablet_data = self._load_tablet_data(tablet_json_path)
        self.class_names = self._load_class_names(classes_pickle_path)
        self.name_to_tablet = {t['name']: t for t in self.tablet_data}

        # 영어 파일명 -> 한국어 이름 매핑
        self.english_to_korean = self._build_name_mapping()

    def _load_tablet_data(self, path: str) -> List[Dict]:
        """tablet.json 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_class_names(self, path: str) -> List[str]:
        """classes.pickle 로드"""
        try:
            import pickle
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"경고: {path} 파일을 찾을 수 없습니다.")
            return []

    def _build_name_mapping(self) -> Dict[str, str]:
        """영어 파일명 -> 한국어 석판 이름 매핑 생성"""
        mapping = {}
        for tablet in self.tablet_data:
            if 'image_url' in tablet:
                # "assets/tablets/chivalry.png" -> "chivalry"
                filename = tablet['image_url'].split('/')[-1]
                filename = os.path.splitext(filename)[0]
                mapping[filename] = tablet['name']
        return mapping

    def detect_tablets_from_screenshot(self, screenshot_path: str) -> List[Tablet]:
        """
        스크린샷에서 CNN 감지 실행 후 Tablet 객체 리스트 반환

        inven_test.py 연동:
        - 감지 함수 임포트
        - 슬롯 탐지 실행
        - 분류 실행
        - 예측을 Tablet 객체로 매핑
        """
        # CNN 경로 추가
        cnn_path = os.path.join(PROJECT_ROOT, 'CNN')
        if cnn_path not in sys.path:
            sys.path.insert(0, cnn_path)

        try:
            import cv2
            import numpy as np
            import pickle
            from tensorflow.keras.models import load_model

            # 모델 로드
            model_path = os.path.join(cnn_path, 'sephiria_item_model.keras')
            classes_path = os.path.join(cnn_path, 'classes.pickle')

            model = load_model(model_path)
            with open(classes_path, 'rb') as f:
                class_names = pickle.load(f)

            # 석판 클래스 로드
            tablet_classes = self._load_tablet_class_names()

            # 이미지 로드 및 처리
            image = cv2.imread(screenshot_path)
            if image is None:
                print(f"오류: 이미지를 로드할 수 없습니다: {screenshot_path}")
                return []

            # 슬롯 탐지 (간소화된 버전)
            slots = self._find_inventory_slots(image)

            detected_tablets = []

            for idx, (x, y, w, h) in enumerate(slots):
                roi = image[y:y+h, x:x+w]
                prediction = self._predict_item(model, roi, class_names)

                pred_idx = prediction.argmax()
                label = class_names[pred_idx]
                confidence = prediction[pred_idx]

                # 높은 신뢰도의 석판 감지만 사용
                if confidence > 0.5 and label in tablet_classes:
                    tablet = self.create_tablet_from_name(label)
                    if tablet:
                        detected_tablets.append(tablet)

            return detected_tablets

        except ImportError as e:
            print(f"CNN 모듈 임포트 오류: {e}")
            print("CNN 연동 없이 수동 석판 목록을 사용하세요.")
            return []

    def _find_inventory_slots(self, image) -> List[tuple]:
        """간소화된 슬롯 탐지 (inven_test.py 로직 기반)"""
        import cv2
        import numpy as np

        BORDER_COLOR_BGR = (52, 32, 36)
        img_h, img_w = image.shape[:2]

        # 슬롯 크기 추정 (이미지 너비의 8%)
        estimated_slot_size = img_w * 0.08
        min_size = int(estimated_slot_size * 0.7)
        max_size = int(estimated_slot_size * 1.5)

        # 테두리 색상 마스크
        tolerance = 15
        lower = np.array([max(0, c - tolerance) for c in BORDER_COLOR_BGR])
        upper = np.array([min(255, c + tolerance) for c in BORDER_COLOR_BGR])

        mask = cv2.inRange(image, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 컨투어 탐지
        edges = cv2.Canny(mask, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / float(h) if h > 0 else 0

            if min_size < w < max_size and min_size < h < max_size and 0.75 < ar < 1.25:
                candidates.append((x, y, w, h))

        return candidates

    def _predict_item(self, model, roi, class_names):
        """단일 아이템 예측"""
        import cv2
        import numpy as np
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        IMG_SIZE = 128
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        roi_input = preprocess_input(roi_resized.astype(np.float32))[None, ...]

        return model.predict(roi_input, verbose=0)[0]

    def _load_tablet_class_names(self) -> Set[str]:
        """석판 클래스 이름 목록 로드"""
        tablets_folder = os.path.join(PROJECT_ROOT, 'assets', 'tablets')
        tablet_classes = set()

        if os.path.exists(tablets_folder):
            for filename in os.listdir(tablets_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    class_name = os.path.splitext(filename)[0]
                    tablet_classes.add(class_name)

        return tablet_classes

    def create_tablet_from_name(self, english_name: str) -> Optional[Tablet]:
        """영어 파일명으로 Tablet 객체 생성"""
        korean_name = self.english_to_korean.get(english_name)
        if not korean_name:
            return None

        tablet_data = self.name_to_tablet.get(korean_name)
        if not tablet_data:
            return None

        return self._convert_to_tablet_object(tablet_data)

    def _convert_to_tablet_object(self, data: Dict) -> Tablet:
        """JSON 석판 데이터를 Tablet 객체로 변환"""
        position_effects = []
        shape_effects = []

        for effect in data.get('effects', []):
            if 'pos' in effect:
                effect_type = effect.get('type', 'level_add')
                value = effect.get('value', 0)

                # restriction_remove 처리
                if 'restriction_remove' in effect:
                    effect_type = 'restriction_remove'
                    value = effect['restriction_remove']

                position_effects.append(PositionEffect(
                    dx=effect['pos'][0],
                    dy=effect['pos'][1],
                    effect_type=effect_type,
                    value=value
                ))
            elif 'shape' in effect:
                shape_effects.append(ShapeEffect(
                    shape=effect['shape'],
                    effect_type=effect.get('type', 'level_add'),
                    value=effect.get('level_add', effect.get('value', 0))
                ))

        props = data.get('properties', {})
        restriction_str = props.get('restriction')

        # restriction 변환
        restriction = None
        if restriction_str:
            restriction = Restriction(restriction_str)

        # rotatable 또는 can_rotate 속성 확인
        rotatable = props.get('rotatable', False) or props.get('can_rotate', False)

        # rarity 변환
        rarity_str = props.get('rarity', '일반')
        try:
            rarity = Rarity(rarity_str)
        except ValueError:
            rarity = Rarity.NORMAL

        return Tablet(
            id=data['id'],
            name=data['name'],
            rotatable=rotatable,
            rarity=rarity,
            restriction=restriction,
            position_effects=position_effects,
            shape_effects=shape_effects
        )

    def get_all_tablets(self) -> List[Tablet]:
        """tablet.json에서 모든 석판 로드"""
        return [self._convert_to_tablet_object(t) for t in self.tablet_data]

    def get_tablets_by_names(self, names: List[str]) -> List[Tablet]:
        """이름 목록으로 석판 로드 (영어 또는 한국어)"""
        tablets = []
        for name in names:
            # 먼저 영어 이름으로 시도
            tablet = self.create_tablet_from_name(name)
            if tablet:
                tablets.append(tablet)
                continue

            # 한국어 이름으로 시도
            if name in self.name_to_tablet:
                tablet = self._convert_to_tablet_object(self.name_to_tablet[name])
                tablets.append(tablet)

        return tablets
