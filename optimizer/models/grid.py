from dataclasses import dataclass, field
from typing import Optional, List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .tablet import Tablet, Restriction


@dataclass
class Cell:
    """그리드의 단일 셀"""
    row: int
    col: int
    tablet: Optional['Tablet'] = None
    base_level: int = 0
    level_bonus: int = 0
    restriction_removed: bool = False

    @property
    def total_level(self) -> int:
        return self.base_level + self.level_bonus

    def clear(self):
        """셀 상태 초기화"""
        self.tablet = None
        self.level_bonus = 0
        self.restriction_removed = False


@dataclass
class Grid:
    """게임 보드 표현"""
    rows: int
    cols: int
    cells: np.ndarray = field(init=False)  # Cell 객체의 2D 배열

    def __post_init__(self):
        self.cells = np.empty((self.rows, self.cols), dtype=object)
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r, c] = Cell(row=r, col=c)

    def place_tablet(self, tablet: 'Tablet', row: int, col: int) -> bool:
        """석판을 위치에 배치 (유효하면 True 반환)"""
        if not self.is_in_bounds(row, col):
            return False
        if self.cells[row, col].tablet is not None:
            return False
        self.cells[row, col].tablet = tablet
        return True

    def remove_tablet(self, row: int, col: int) -> Optional['Tablet']:
        """위치에서 석판 제거 후 반환"""
        if not self.is_in_bounds(row, col):
            return None
        tablet = self.cells[row, col].tablet
        self.cells[row, col].tablet = None
        return tablet

    def is_valid_placement(self, tablet: 'Tablet', row: int, col: int) -> bool:
        """제한 조건을 고려하여 배치가 유효한지 확인"""
        from .tablet import Restriction

        if tablet.restriction is None or tablet.restriction == Restriction.NONE:
            return True

        cell = self.cells[row, col]
        if cell.restriction_removed:
            return True

        restriction = tablet.restriction

        if restriction == Restriction.BOTTOM and row != self.rows - 1:
            return False
        if restriction == Restriction.TOP and row != 0:
            return False
        if restriction == Restriction.LEFT_RIGHT:
            # 좌우 경계에만 배치 가능
            if col != 0 and col != self.cols - 1:
                return False

        return True

    def get_all_placements(self) -> List[Tuple[int, int, 'Tablet']]:
        """배치된 모든 석판의 (row, col, tablet) 목록 반환"""
        placements = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cells[r, c].tablet is not None:
                    placements.append((r, c, self.cells[r, c].tablet))
        return placements

    def clear(self):
        """모든 석판 제거 및 레벨 초기화"""
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r, c].clear()

    def is_in_bounds(self, row: int, col: int) -> bool:
        """좌표가 그리드 범위 내인지 확인"""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def copy(self) -> 'Grid':
        """그리드 깊은 복사"""
        new_grid = Grid(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                old_cell = self.cells[r, c]
                new_cell = new_grid.cells[r, c]
                new_cell.base_level = old_cell.base_level
                new_cell.level_bonus = old_cell.level_bonus
                new_cell.restriction_removed = old_cell.restriction_removed
                if old_cell.tablet:
                    new_cell.tablet = old_cell.tablet.copy()
        return new_grid

    def __repr__(self) -> str:
        lines = [f"Grid({self.rows}x{self.cols})"]
        for r in range(self.rows):
            row_str = "|"
            for c in range(self.cols):
                cell = self.cells[r, c]
                if cell.tablet:
                    name = cell.tablet.name[:3]
                else:
                    name = "---"
                row_str += f" {name:^5} |"
            lines.append(row_str)
        return "\n".join(lines)
