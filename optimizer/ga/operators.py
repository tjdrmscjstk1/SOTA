import random
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.tablet import Tablet
    from .chromosome import Chromosome, Gene


class GeneticOperators:
    """
    유전 연산자:

    교차 전략:
    - PMX (Partially Mapped Crossover): 순열 문제에 적합
    - 위치 기반 교차: 절대 위치 유지

    돌연변이 전략:
    - 스왑 돌연변이: 두 석판 위치 교환
    - 회전 돌연변이: 석판 회전 변경
    - 스크램블 돌연변이: 유전자 부분집합 랜덤화
    """

    @staticmethod
    def pmx_crossover(parent1: 'Chromosome',
                      parent2: 'Chromosome') -> Tuple['Chromosome', 'Chromosome']:
        """부분 매핑 교차 (PMX)"""
        from .chromosome import Chromosome, Gene

        size = len(parent1.genes)

        # 교차점 선택
        cx1, cx2 = sorted(random.sample(range(size), 2))

        # 자식 생성
        child1_genes = [None] * size
        child2_genes = [None] * size

        # 부모로부터 세그먼트 복사
        for i in range(cx1, cx2):
            child1_genes[i] = parent1.genes[i].copy()
            child2_genes[i] = parent2.genes[i].copy()

        # 나머지 위치 채우기 (석판 유일성 유지)
        GeneticOperators._fill_pmx(child1_genes, parent2.genes, cx1, cx2)
        GeneticOperators._fill_pmx(child2_genes, parent1.genes, cx1, cx2)

        child1 = Chromosome(
            genes=child1_genes,
            grid_rows=parent1.grid_rows,
            grid_cols=parent1.grid_cols
        )
        child2 = Chromosome(
            genes=child2_genes,
            grid_rows=parent1.grid_rows,
            grid_cols=parent1.grid_cols
        )

        return child1, child2

    @staticmethod
    def _fill_pmx(child_genes: List, parent_genes: List, cx1: int, cx2: int):
        """PMX 자식 유전자 채우기"""
        from .chromosome import Gene

        used_tablets = {g.tablet_idx for g in child_genes if g is not None and g.tablet_idx >= 0}

        parent_idx = 0
        for i in range(len(child_genes)):
            if child_genes[i] is None:
                # 부모에서 미사용 석판 찾기
                while parent_idx < len(parent_genes):
                    p_gene = parent_genes[parent_idx]
                    if p_gene.tablet_idx == -1 or p_gene.tablet_idx not in used_tablets:
                        # 현재 인덱스의 위치에 부모의 석판 배치
                        child_genes[i] = Gene(
                            tablet_idx=p_gene.tablet_idx,
                            position=parent_genes[i].position,
                            rotation=p_gene.rotation
                        )
                        if p_gene.tablet_idx >= 0:
                            used_tablets.add(p_gene.tablet_idx)
                        parent_idx += 1
                        break
                    parent_idx += 1

                # 남은 위치에 빈 셀 배치
                if child_genes[i] is None:
                    child_genes[i] = Gene(
                        tablet_idx=-1,
                        position=parent_genes[i].position,
                        rotation=0
                    )

    @staticmethod
    def swap_mutation(chromosome: 'Chromosome', mutation_rate: float = 0.1) -> 'Chromosome':
        """스왑 돌연변이: 두 석판의 위치 교환"""
        if random.random() < mutation_rate:
            genes = chromosome.genes
            if len(genes) >= 2:
                idx1, idx2 = random.sample(range(len(genes)), 2)

                # 석판 인덱스와 회전 교환 (위치는 유지)
                genes[idx1].tablet_idx, genes[idx2].tablet_idx = \
                    genes[idx2].tablet_idx, genes[idx1].tablet_idx
                genes[idx1].rotation, genes[idx2].rotation = \
                    genes[idx2].rotation, genes[idx1].rotation

        return chromosome

    @staticmethod
    def rotation_mutation(chromosome: 'Chromosome',
                          tablets: List['Tablet'],
                          mutation_rate: float = 0.15) -> 'Chromosome':
        """회전 돌연변이: 회전 가능 석판의 회전 변경"""
        for gene in chromosome.genes:
            if random.random() < mutation_rate and gene.tablet_idx >= 0:
                if gene.tablet_idx < len(tablets):
                    tablet = tablets[gene.tablet_idx]
                    if tablet.rotatable:
                        gene.rotation = random.choice([0, 90, 180, 270])

        return chromosome

    @staticmethod
    def scramble_mutation(chromosome: 'Chromosome', mutation_rate: float = 0.05) -> 'Chromosome':
        """스크램블 돌연변이: 연속 구간의 석판들을 섞음"""
        if random.random() < mutation_rate:
            genes = chromosome.genes
            if len(genes) >= 3:
                # 스크램블 구간 선택
                start = random.randint(0, len(genes) - 2)
                end = random.randint(start + 1, len(genes))

                # 구간 내 석판 인덱스와 회전 추출
                segment_data = [(genes[i].tablet_idx, genes[i].rotation)
                                for i in range(start, end)]
                random.shuffle(segment_data)

                # 셔플된 데이터 재배치
                for i, (tablet_idx, rotation) in enumerate(segment_data):
                    genes[start + i].tablet_idx = tablet_idx
                    genes[start + i].rotation = rotation

        return chromosome

    @staticmethod
    def tournament_selection(population: List['Chromosome'],
                             tournament_size: int = 3) -> 'Chromosome':
        """토너먼트 선택"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda c: c.fitness)

    @staticmethod
    def roulette_selection(population: List['Chromosome']) -> 'Chromosome':
        """룰렛 휠 선택"""
        # 적합도가 음수일 수 있으므로 최소값 이동
        min_fitness = min(c.fitness for c in population)
        adjusted_fitness = [c.fitness - min_fitness + 1 for c in population]
        total = sum(adjusted_fitness)

        pick = random.uniform(0, total)
        current = 0
        for i, fit in enumerate(adjusted_fitness):
            current += fit
            if current > pick:
                return population[i]

        return population[-1]
