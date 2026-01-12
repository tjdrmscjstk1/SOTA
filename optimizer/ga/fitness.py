from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.grid import Grid
    from ..models.tablet import Tablet, Rarity
    from .chromosome import Chromosome
    from ..effects.calculator import EffectCalculator


class FitnessEvaluator:
    """
    적합도 함수 설계:

    주요 목표: 모든 셀 레벨의 총합 최대화

    패널티 요소:
    - 제한 위반 배치: -1000 per violation
    - 미사용 전설 석판: -50
    - 미사용 희귀 석판: -20

    보너스 요소:
    - 시너지 탐지 (향후 확장)
    """

    RESTRICTION_PENALTY = -1000
    UNUSED_LEGENDARY_PENALTY = -50
    UNUSED_RARE_PENALTY = -20

    def __init__(self, grid: 'Grid', tablets: List['Tablet']):
        self.grid = grid
        self.tablets = tablets
        self._calculator = None

    @property
    def calculator(self):
        """지연 로딩 EffectCalculator"""
        if self._calculator is None:
            from ..effects.calculator import EffectCalculator
            self._calculator = EffectCalculator(self.grid)
        return self._calculator

    def evaluate(self, chromosome: 'Chromosome') -> float:
        """염색체의 적합도 점수 계산"""
        # 염색체를 그리드로 디코딩
        chromosome.to_grid(self.grid, self.tablets)

        # 기본 적합도: 모든 레벨의 합
        base_fitness = self.calculator.get_fitness_score()

        # 제한 위반 패널티
        penalty = self._calculate_restriction_penalty(chromosome)

        # 희귀도 보너스/패널티
        rarity_bonus = self._calculate_rarity_bonus(chromosome)

        total_fitness = base_fitness + penalty + rarity_bonus
        chromosome.fitness = total_fitness

        return total_fitness

    def _calculate_restriction_penalty(self, chromosome: 'Chromosome') -> float:
        """잘못된 위치에 배치된 석판에 대한 패널티"""
        penalty = 0
        for gene in chromosome.genes:
            if gene.tablet_idx >= 0 and gene.tablet_idx < len(self.tablets):
                tablet = self.tablets[gene.tablet_idx]
                row, col = gene.position

                if not self.grid.is_valid_placement(tablet, row, col):
                    penalty += self.RESTRICTION_PENALTY
        return penalty

    def _calculate_rarity_bonus(self, chromosome: 'Chromosome') -> float:
        """희귀/전설 석판 사용 장려"""
        from ..models.tablet import Rarity

        bonus = 0
        used_indices = {g.tablet_idx for g in chromosome.genes if g.tablet_idx >= 0}

        for idx, tablet in enumerate(self.tablets):
            if idx not in used_indices:
                if tablet.rarity == Rarity.LEGENDARY:
                    bonus += self.UNUSED_LEGENDARY_PENALTY
                elif tablet.rarity == Rarity.RARE:
                    bonus += self.UNUSED_RARE_PENALTY

        return bonus

    def get_detailed_score(self, chromosome: 'Chromosome') -> dict:
        """상세 점수 분해 반환"""
        chromosome.to_grid(self.grid, self.tablets)

        base_fitness = self.calculator.get_fitness_score()
        penalty = self._calculate_restriction_penalty(chromosome)
        rarity_bonus = self._calculate_rarity_bonus(chromosome)

        return {
            'base_fitness': base_fitness,
            'restriction_penalty': penalty,
            'rarity_bonus': rarity_bonus,
            'total': base_fitness + penalty + rarity_bonus
        }
