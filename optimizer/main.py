#!/usr/bin/env python3
"""
Sephiria Tablet Inventory Optimizer
유전 알고리즘 기반 석판 배치 최적화

사용법:
    python -m optimizer.main --rows 6 --cols 6
    python -m optimizer.main --rows 6 --cols 6 --screenshot ./CNN/test1.png
    python -m optimizer.main --rows 6 --cols 6 --tablets 기사도 건조 근사
"""

import argparse
import json
import os
import sys

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optimizer.models.grid import Grid
from optimizer.models.tablet import Tablet
from optimizer.effects.calculator import EffectCalculator
from optimizer.ga.algorithm import GeneticAlgorithm, GAConfig
from optimizer.ga.fitness import FitnessEvaluator
from optimizer.integration.cnn_bridge import CNNBridge


def main():
    parser = argparse.ArgumentParser(
        description='세피리아 석판 인벤토리 최적화기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 6x6 그리드에서 모든 석판으로 최적화
  python -m optimizer.main --rows 6 --cols 6

  # 스크린샷에서 석판 감지 후 최적화
  python -m optimizer.main --rows 6 --cols 6 --screenshot ./CNN/test1.png

  # 특정 석판만 사용
  python -m optimizer.main --rows 4 --cols 4 --tablets 기사도 건조 근사

  # GA 파라미터 조정
  python -m optimizer.main --rows 6 --cols 6 --generations 1000 --population 200
        """
    )

    parser.add_argument('--rows', type=int, required=True, help='그리드 행 수')
    parser.add_argument('--cols', type=int, required=True, help='그리드 열 수')
    parser.add_argument('--screenshot', type=str, help='인벤토리 스크린샷 경로')
    parser.add_argument('--tablets', type=str, nargs='+', help='사용할 석판 이름 목록')
    parser.add_argument('--output', type=str, default='optimal_placement.json', help='출력 파일 경로')
    parser.add_argument('--generations', type=int, default=500, help='GA 세대 수')
    parser.add_argument('--population', type=int, default=100, help='인구 크기')
    parser.add_argument('--early-stop', type=int, default=50, help='조기 종료 세대 수')
    parser.add_argument('--quiet', action='store_true', help='진행 상황 출력 안 함')

    args = parser.parse_args()

    print("=" * 60)
    print("세피리아 석판 인벤토리 최적화기")
    print("=" * 60)

    # CNN 브릿지 초기화
    bridge = CNNBridge()

    # 최적화할 석판 가져오기
    if args.screenshot:
        print(f"\n스크린샷 분석 중: {args.screenshot}")
        tablets = bridge.detect_tablets_from_screenshot(args.screenshot)
        print(f"-> {len(tablets)}개 석판 감지")
    elif args.tablets:
        print(f"\n지정된 석판 로드 중: {args.tablets}")
        tablets = bridge.get_tablets_by_names(args.tablets)
        print(f"-> {len(tablets)}개 석판 로드")
    else:
        print("\n모든 석판 로드 중...")
        tablets = bridge.get_all_tablets()
        print(f"-> {len(tablets)}개 석판 로드")

    if not tablets:
        print("오류: 최적화할 석판이 없습니다.")
        sys.exit(1)

    # 그리드 및 최적화기 생성
    grid = Grid(rows=args.rows, cols=args.cols)
    fitness_evaluator = FitnessEvaluator(grid, tablets)

    config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        early_stop_generations=args.early_stop
    )

    ga = GeneticAlgorithm(config, grid, tablets, fitness_evaluator)

    # 진행 콜백
    def progress_callback(stats):
        if not args.quiet and stats['generation'] % 50 == 0:
            print(f"세대 {stats['generation']:4d}: "
                  f"최고={stats['best_fitness']:8.2f}, "
                  f"평균={stats['avg_fitness']:8.2f}")

    # 최적화 실행
    print(f"\n최적화 시작...")
    print(f"  그리드 크기: {args.rows}x{args.cols}")
    print(f"  석판 수: {len(tablets)}")
    print(f"  세대 수: {args.generations}")
    print(f"  인구 크기: {args.population}")
    print()

    best = ga.evolve(callback=progress_callback)

    # 결과 포맷팅
    result = format_result(best, tablets, grid)

    # JSON 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 결과 출력
    print("\n" + "=" * 60)
    print("최적화 완료!")
    print("=" * 60)
    print(f"최고 적합도: {best.fitness}")
    print(f"총 레벨 합계: {result['total_level']}")
    print(f"배치된 석판: {len(result['placements'])}개")
    print(f"결과 저장: {args.output}")

    # 상세 점수
    detailed = fitness_evaluator.get_detailed_score(best)
    print(f"\n점수 분해:")
    print(f"  기본 레벨: {detailed['base_fitness']}")
    print(f"  제한 패널티: {detailed['restriction_penalty']}")
    print(f"  희귀도 보너스: {detailed['rarity_bonus']}")

    # 시각적 그리드 출력
    print_visual_grid(best, tablets, grid)


def format_result(chromosome, tablets, grid) -> dict:
    """결과를 JSON 직렬화 가능한 dict로 포맷"""
    chromosome.to_grid(grid, tablets)
    calculator = EffectCalculator(grid)
    levels = calculator.calculate_total_levels()

    placements = []
    for gene in chromosome.genes:
        if gene.tablet_idx >= 0 and gene.tablet_idx < len(tablets):
            tablet = tablets[gene.tablet_idx]
            placements.append({
                'tablet_id': tablet.id,
                'tablet_name': tablet.name,
                'position': {'row': gene.position[0], 'col': gene.position[1]},
                'rotation': gene.rotation
            })

    return {
        'fitness': chromosome.fitness,
        'grid_size': {'rows': grid.rows, 'cols': grid.cols},
        'placements': placements,
        'level_matrix': levels.tolist(),
        'total_level': int(levels.sum())
    }


def print_visual_grid(chromosome, tablets, grid):
    """그리드 ASCII 시각화 출력"""
    chromosome.to_grid(grid, tablets)
    calculator = EffectCalculator(grid)
    levels = calculator.calculate_total_levels()

    print("\n" + "=" * 60)
    print("최적 배치 그리드:")
    print("=" * 60)

    # 석판 이름 출력
    col_width = 8
    header = "|" + "|".join(f" {c:^{col_width-2}} " for c in range(grid.cols)) + "|"
    separator = "+" + "+".join("-" * col_width for _ in range(grid.cols)) + "+"

    print(separator)
    for r in range(grid.rows):
        row_str = "|"
        for c in range(grid.cols):
            cell = grid.cells[r, c]
            if cell.tablet:
                name = cell.tablet.name[:col_width-2]
            else:
                name = "----"
            row_str += f" {name:^{col_width-2}} |"
        print(row_str)
        print(separator)

    print("\n레벨 보너스:")
    print(separator)
    for r in range(grid.rows):
        row_str = "|"
        for c in range(grid.cols):
            level = levels[r, c]
            row_str += f" {level:^{col_width-2}} |"
        print(row_str)
        print(separator)


if __name__ == '__main__':
    main()
