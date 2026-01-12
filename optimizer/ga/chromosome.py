from dataclasses import dataclass, field
from typing import List, Tuple, Optional, TYPE_CHECKING
import random
import copy

if TYPE_CHECKING:
    from ..models.tablet import Tablet
    from ..models.grid import Grid


@dataclass
class Gene:
    """단일 유전자 - 석판 배치 정보"""
    tablet_idx: int            # 사용 가능한 석판 리스트의 인덱스 (-1 = 빈 셀)
    position: Tuple[int, int]  # (row, col) 그리드 위치
    rotation: int              # 0, 90, 180, 270 (rotatable 석판만 유효)

    def copy(self) -> 'Gene':
        return Gene(self.tablet_idx, self.position, self.rotation)


@dataclass
class Chromosome:
    """
    염색체 인코딩 전략:
    - 순열 기반: 각 유전자가 그리드 셀을 표현
    - 유전자 값: (tablet_index, rotation)
    - 길이: rows * cols
    - 제약: 각 석판은 한 번만 사용 가능
    """
    genes: List[Gene]
    grid_rows: int
    grid_cols: int
    fitness: float = 0.0

    @classmethod
    def create_random(cls,
                      grid_rows: int,
                      grid_cols: int,
                      available_tablets: List['Tablet']) -> 'Chromosome':
        """랜덤 유효 염색체 생성"""
        total_cells = grid_rows * grid_cols
        genes = []

        # 석판 인덱스를 셔플하여 랜덤 위치에 배정
        tablet_indices = list(range(len(available_tablets)))
        random.shuffle(tablet_indices)

        # 셀에 석판 배정 (석판 수보다 셀이 많으면 일부 셀은 비어있음)
        for pos_idx in range(total_cells):
            row = pos_idx // grid_cols
            col = pos_idx % grid_cols

            if pos_idx < len(tablet_indices):
                tablet_idx = tablet_indices[pos_idx]
                tablet = available_tablets[tablet_idx]
                rotation = random.choice([0, 90, 180, 270]) if tablet.rotatable else 0
            else:
                tablet_idx = -1  # 빈 셀
                rotation = 0

            genes.append(Gene(tablet_idx, (row, col), rotation))

        return cls(genes=genes, grid_rows=grid_rows, grid_cols=grid_cols)

    def to_grid(self, grid: 'Grid', tablets: List['Tablet']) -> 'Grid':
        """염색체를 그리드 배치로 변환"""
        grid.clear()
        for gene in self.genes:
            if gene.tablet_idx >= 0 and gene.tablet_idx < len(tablets):
                tablet = tablets[gene.tablet_idx].copy()
                tablet.rotation = gene.rotation
                grid.place_tablet(tablet, gene.position[0], gene.position[1])
        return grid

    def is_valid(self, tablets: List['Tablet']) -> bool:
        """염색체가 유효한 배치를 나타내는지 확인"""
        used_tablets = set()
        used_positions = set()

        for gene in self.genes:
            # 위치 유일성 확인
            if gene.position in used_positions:
                return False
            used_positions.add(gene.position)

            # 석판 유일성 확인
            if gene.tablet_idx >= 0:
                if gene.tablet_idx in used_tablets:
                    return False
                used_tablets.add(gene.tablet_idx)

        return True

    def copy(self) -> 'Chromosome':
        """염색체 깊은 복사"""
        new_genes = [gene.copy() for gene in self.genes]
        return Chromosome(
            genes=new_genes,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            fitness=self.fitness
        )

    def get_tablet_positions(self) -> dict:
        """석판 인덱스 -> 위치 매핑 반환"""
        return {gene.tablet_idx: gene.position
                for gene in self.genes if gene.tablet_idx >= 0}

    def __repr__(self) -> str:
        placed = sum(1 for g in self.genes if g.tablet_idx >= 0)
        return f"Chromosome(fitness={self.fitness:.2f}, placed={placed}/{len(self.genes)})"
