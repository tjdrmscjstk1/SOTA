from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from enum import Enum


class Rarity(Enum):
    """석판 희귀도"""
    NORMAL = "일반"
    ADVANCED = "고급"
    RARE = "희귀"
    LEGENDARY = "전설"


class Restriction(Enum):
    """배치 제한 타입"""
    NONE = "none"
    BOTTOM = "bottom"      # 하단에만 배치 가능
    TOP = "top"            # 상단에만 배치 가능
    LEFT_RIGHT = "left_right"  # 좌우 인접 불가


@dataclass
class PositionEffect:
    """위치 기반 효과 (석판 위치 기준 상대 좌표)"""
    dx: int                    # 상대 x 오프셋 (-1=왼쪽, +1=오른쪽)
    dy: int                    # 상대 y 오프셋 (+1=위, -1=아래)
    effect_type: str           # "level_add" 또는 "restriction_remove"
    value: Union[int, bool]    # 레벨 값 또는 제한 해제 여부


@dataclass
class ShapeEffect:
    """형태 기반 효과 (행/열/대각선 전체 영향)"""
    shape: str                 # "row", "column", "diagonal", "top", "bottom"
    effect_type: str           # "level_add"
    value: int


@dataclass
class Tablet:
    """석판 클래스"""
    id: str
    name: str
    rotatable: bool
    rarity: Rarity
    restriction: Optional[Restriction]
    position_effects: List[PositionEffect] = field(default_factory=list)
    shape_effects: List[ShapeEffect] = field(default_factory=list)

    # 런타임 상태
    rotation: int = 0  # 0, 90, 180, 270도 (rotatable=True일 때만 유효)

    def get_rotated_effects(self) -> List[PositionEffect]:
        """회전이 적용된 위치 효과 반환"""
        if not self.rotatable or self.rotation == 0:
            return self.position_effects

        rotated = []
        for effect in self.position_effects:
            dx, dy = self._rotate_offset(effect.dx, effect.dy, self.rotation)
            rotated.append(PositionEffect(dx, dy, effect.effect_type, effect.value))
        return rotated

    @staticmethod
    def _rotate_offset(dx: int, dy: int, degrees: int) -> Tuple[int, int]:
        """좌표를 시계 방향으로 회전"""
        if degrees == 90:
            return (dy, -dx)
        elif degrees == 180:
            return (-dx, -dy)
        elif degrees == 270:
            return (-dy, dx)
        return (dx, dy)

    def copy(self) -> 'Tablet':
        """석판 복사본 생성"""
        return Tablet(
            id=self.id,
            name=self.name,
            rotatable=self.rotatable,
            rarity=self.rarity,
            restriction=self.restriction,
            position_effects=self.position_effects.copy(),
            shape_effects=self.shape_effects.copy(),
            rotation=self.rotation
        )

    def __repr__(self) -> str:
        return f"Tablet({self.id}, {self.name}, rot={self.rotation})"
