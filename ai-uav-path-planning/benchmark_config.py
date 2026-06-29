from __future__ import annotations

import random
from dataclasses import dataclass

from path_algorithms import Grid, Point

ROWS = 24
COLS = 32
START: Point = (2, 2)
GOAL: Point = (21, 29)


@dataclass(frozen=True)
class BenchmarkScenario:
    """批量对比实验中的内置静态场景。"""

    name: str
    obstacle_ratio: float
    seed: int
    description: str


BENCHMARK_SCENARIOS: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario("普通障碍", 0.12, 2026061701, "障碍较少，用于验证三种算法的基础可达性和路径质量。"),
    BenchmarkScenario("密集障碍", 0.22, 2026061702, "障碍明显增加，用于观察搜索范围、转折次数和综合代价变化。"),
    BenchmarkScenario("高密障碍", 0.30, 2026061703, "复杂度较高，用于测试算法在高障碍比例下的稳定性。"),
)


def clone_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def corridor_points(start: Point, goal: Point) -> list[Point]:
    """生成一条基础连通走廊，保证内置实验具有可复现的可行路径。"""
    row, col = start
    gr, gc = goal
    points = [(row, col)]
    while row != gr:
        row += 1 if row < gr else -1
        points.append((row, col))
    while col != gc:
        col += 1 if col < gc else -1
        points.append((row, col))
    return points


def make_benchmark_grid(rows: int, cols: int, start: Point, goal: Point, obstacle_ratio: float, seed: int) -> Grid:
    """按照固定随机种子生成批量实验地图。"""
    rng = random.Random(seed)
    grid: Grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            point = (row, col)
            if point in (start, goal):
                continue
            if rng.random() < obstacle_ratio:
                grid[row][col] = 1

    # 保留一条可复现实验通道，避免随机障碍导致整张地图不可达，影响三算法公平比较。
    for row, col in corridor_points(start, goal):
        grid[row][col] = 0
        for nr, nc in ((row + 1, col), (row, col + 1), (row - 1, col), (row, col - 1)):
            if 0 <= nr < rows and 0 <= nc < cols:
                grid[nr][nc] = 0
    return grid
