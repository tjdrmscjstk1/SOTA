from typing import List, Callable, Optional
from dataclasses import dataclass, field
import random

from .chromosome import Chromosome
from .fitness import FitnessEvaluator
from .operators import GeneticOperators
from ..models.grid import Grid
from ..models.tablet import Tablet


@dataclass
class GAConfig:
    """GA 하이퍼파라미터"""
    population_size: int = 100
    generations: int = 500
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    rotation_mutation_rate: float = 0.15
    scramble_mutation_rate: float = 0.05
    elitism_count: int = 5
    tournament_size: int = 3
    early_stop_generations: int = 50  # N세대 동안 개선 없으면 중단


class GeneticAlgorithm:
    """
    유전 알고리즘 메인 구현:
    - 엘리트 보존: 상위 N개 개체 유지
    - 토너먼트 선택
    - PMX 교차
    - 다중 돌연변이 연산자
    - 조기 종료
    """

    def __init__(self,
                 config: GAConfig,
                 grid: Grid,
                 tablets: List[Tablet],
                 fitness_evaluator: Optional[FitnessEvaluator] = None):
        self.config = config
        self.grid = grid
        self.tablets = tablets

        if fitness_evaluator is None:
            self.fitness_evaluator = FitnessEvaluator(grid, tablets)
        else:
            self.fitness_evaluator = fitness_evaluator

        self.operators = GeneticOperators()

        self.population: List[Chromosome] = []
        self.best_solution: Optional[Chromosome] = None
        self.generation_stats: List[dict] = []

    def initialize_population(self):
        """초기 랜덤 인구 생성"""
        self.population = []
        for _ in range(self.config.population_size):
            chromosome = Chromosome.create_random(
                self.grid.rows,
                self.grid.cols,
                self.tablets
            )
            self.population.append(chromosome)

        # 초기 인구 평가
        for chromosome in self.population:
            self.fitness_evaluator.evaluate(chromosome)

        self._update_best()

    def evolve(self, callback: Optional[Callable] = None) -> Chromosome:
        """GA 진화 실행"""
        self.initialize_population()

        no_improvement_count = 0
        best_fitness = self.best_solution.fitness

        for generation in range(self.config.generations):
            new_population = []

            # 엘리트 보존: 상위 N개 개체 유지
            sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
            elite = [c.copy() for c in sorted_pop[:self.config.elitism_count]]
            new_population.extend(elite)

            # 나머지 인구 생성
            while len(new_population) < self.config.population_size:
                # 선택
                parent1 = self.operators.tournament_selection(
                    self.population, self.config.tournament_size)
                parent2 = self.operators.tournament_selection(
                    self.population, self.config.tournament_size)

                # 교차
                if random.random() < self.config.crossover_rate:
                    child1, child2 = self.operators.pmx_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # 돌연변이
                child1 = self.operators.swap_mutation(child1, self.config.mutation_rate)
                child2 = self.operators.swap_mutation(child2, self.config.mutation_rate)
                child1 = self.operators.rotation_mutation(
                    child1, self.tablets, self.config.rotation_mutation_rate)
                child2 = self.operators.rotation_mutation(
                    child2, self.tablets, self.config.rotation_mutation_rate)
                child1 = self.operators.scramble_mutation(
                    child1, self.config.scramble_mutation_rate)
                child2 = self.operators.scramble_mutation(
                    child2, self.config.scramble_mutation_rate)

                # 평가
                self.fitness_evaluator.evaluate(child1)
                self.fitness_evaluator.evaluate(child2)

                new_population.extend([child1, child2])

            # 인구 크기 조정
            self.population = new_population[:self.config.population_size]
            self._update_best()

            # 통계 추적
            stats = {
                'generation': generation,
                'best_fitness': self.best_solution.fitness,
                'avg_fitness': sum(c.fitness for c in self.population) / len(self.population),
                'min_fitness': min(c.fitness for c in self.population)
            }
            self.generation_stats.append(stats)

            # 진행 콜백
            if callback:
                callback(stats)

            # 조기 종료 확인
            if self.best_solution.fitness > best_fitness:
                best_fitness = self.best_solution.fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.config.early_stop_generations:
                print(f"조기 종료: {generation}세대 (개선 없음 {no_improvement_count}세대)")
                break

        return self.best_solution

    def _update_best(self):
        """최고 솔루션 업데이트"""
        current_best = max(self.population, key=lambda c: c.fitness)
        if self.best_solution is None or current_best.fitness > self.best_solution.fitness:
            self.best_solution = current_best.copy()

    def get_convergence_data(self) -> List[dict]:
        """수렴 데이터 반환 (시각화용)"""
        return self.generation_stats
