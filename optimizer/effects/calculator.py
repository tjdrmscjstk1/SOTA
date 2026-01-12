from typing import List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..models.grid import Grid
    from ..models.tablet import Tablet


class EffectCalculator:
    """
    효과 계산 엔진

    좌표계 변환:
    - tablet.json: x=수평(-1=왼쪽, +1=오른쪽), y=수직(+1=위, -1=아래)
    - 그리드: row=0부터 아래로 증가, col=0부터 오른쪽으로 증가

    변환 공식: grid(row, col) + effect(dx, dy) -> grid(row - dy, col + dx)
    """

    def __init__(self, grid: 'Grid'):
        self.grid = grid

    def calculate_total_levels(self) -> np.ndarray:
        """
        모든 셀의 레벨 보너스 계산
        2D 배열로 총 레벨 반환

        처리 순서:
        1. 1차: restriction_remove 효과 적용
        2. 2차: level_add 효과 적용
        """
        # 모든 보너스 초기화
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                self.grid.cells[r, c].level_bonus = 0
                self.grid.cells[r, c].restriction_removed = False

        # 1차: 제한 해제 효과 적용
        self._apply_restriction_removals()

        # 2차: 레벨 효과 적용
        self._apply_level_effects()

        # 결과 매트릭스 생성
        result = np.zeros((self.grid.rows, self.grid.cols), dtype=int)
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                result[r, c] = self.grid.cells[r, c].total_level

        return result

    def _apply_restriction_removals(self):
        """모든 석판에서 restriction_remove 효과 적용"""
        for r, c, tablet in self.grid.get_all_placements():
            for effect in tablet.get_rotated_effects():
                if effect.effect_type == "restriction_remove":
                    target_row = r - effect.dy  # y를 row로 변환
                    target_col = c + effect.dx  # x를 col로 변환

                    if self.grid.is_in_bounds(target_row, target_col):
                        self.grid.cells[target_row, target_col].restriction_removed = True

    def _apply_level_effects(self):
        """모든 level_add 효과 적용"""
        for r, c, tablet in self.grid.get_all_placements():
            # 위치 기반 효과
            for effect in tablet.get_rotated_effects():
                if effect.effect_type == "level_add":
                    target_row = r - effect.dy
                    target_col = c + effect.dx

                    if self.grid.is_in_bounds(target_row, target_col):
                        self.grid.cells[target_row, target_col].level_bonus += effect.value

            # 형태 기반 효과
            for effect in tablet.shape_effects:
                affected_cells = self._get_shape_cells(r, c, effect.shape)
                for tr, tc in affected_cells:
                    self.grid.cells[tr, tc].level_bonus += effect.value

    def _get_shape_cells(self, row: int, col: int, shape: str) -> List[Tuple[int, int]]:
        """형태 효과에 영향받는 모든 셀 좌표 반환"""
        cells = []

        if shape == "row":
            # 같은 행의 모든 셀 (자기 자신 제외)
            for c in range(self.grid.cols):
                if c != col:
                    cells.append((row, c))

        elif shape == "column":
            # 같은 열의 모든 셀 (자기 자신 제외)
            for r in range(self.grid.rows):
                if r != row:
                    cells.append((r, col))

        elif shape == "diagonal":
            # 양쪽 대각선 (자기 자신 제외)
            for offset in range(1, max(self.grid.rows, self.grid.cols)):
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = row + dr * offset, col + dc * offset
                    if self.grid.is_in_bounds(nr, nc):
                        cells.append((nr, nc))

        elif shape == "top":
            # 현재 행 위의 모든 셀
            for r in range(row):
                for c in range(self.grid.cols):
                    cells.append((r, c))

        elif shape == "bottom":
            # 현재 행 아래의 모든 셀
            for r in range(row + 1, self.grid.rows):
                for c in range(self.grid.cols):
                    cells.append((r, c))

        return cells

    def get_fitness_score(self) -> int:
        """모든 셀 레벨의 총합 (GA 적합도) 계산"""
        levels = self.calculate_total_levels()
        return int(np.sum(levels))

    def get_level_matrix(self) -> np.ndarray:
        """레벨 매트릭스 반환 (calculate_total_levels의 별칭)"""
        return self.calculate_total_levels()
