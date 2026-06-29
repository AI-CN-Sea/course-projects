from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

from benchmark_config import BENCHMARK_SCENARIOS, COLS, GOAL, ROWS, START, clone_grid, make_benchmark_grid
from export_utils import export_result_package
from path_algorithms import Grid, Point, SearchResult, astar, dijkstra, improved_astar

ALGORITHMS: tuple[tuple[str, Callable[[Grid, Point, Point], SearchResult]], ...] = (
    ("Dijkstra", dijkstra),
    ("A*", astar),
    ("改进A*", improved_astar),
)


def point_to_text(point: Point) -> str:
    return f"({point[0]},{point[1]})"


def path_to_text(path: list[Point]) -> str:
    return ";".join(point_to_text(point) for point in path)


def count_obstacles(grid: Grid) -> int:
    return sum(1 for row in grid for value in row if value == 1)


def run_with_average_time(
    func: Callable[[Grid, Point, Point], SearchResult],
    grid: Grid,
    start: Point,
    goal: Point,
    repeats: int = 30,
) -> SearchResult:
    """统计平均运行时间。

    路径、搜索格数、转折次数等指标取一次确定性搜索结果；运行时间单独重复搜索多次求平均。
    这样可以减少毫秒级计时抖动，并避免把 GUI 动画时间算入算法运行时间。
    """
    repeats = max(1, repeats)
    representative = func(clone_grid(grid), start, goal)
    total_time = 0.0
    for _ in range(repeats):
        timed = func(clone_grid(grid), start, goal)
        total_time += timed.runtime_ms
    return replace(representative, runtime_ms=total_time / repeats)


def make_record(
    result: SearchResult,
    grid: Grid,
    start: Point,
    goal: Point,
    test_id: str,
    source: str,
    scenario: str,
    scenario_description: str,
    obstacle_ratio_setting: str,
    seed: int | str,
) -> dict[str, object]:
    obstacles = count_obstacles(grid)
    total_cells = len(grid) * len(grid[0]) if grid and grid[0] else 1
    return {
        "实验编号": test_id,
        "数据来源": source,
        "实验场景": scenario,
        "场景说明": scenario_description,
        "随机种子": seed,
        "算法": result.algorithm,
        "是否成功": "是" if result.success else "否",
        "起点": point_to_text(start),
        "终点": point_to_text(goal),
        "设定障碍比例": obstacle_ratio_setting,
        "实际障碍数量": obstacles,
        "实际障碍比例": f"{obstacles / total_cells:.3f}",
        "路径长度": result.distance,
        "转折次数": result.turn_count,
        "综合代价": f"{result.route_cost:.2f}",
        "搜索格数": result.searched_nodes,
        "扩展状态数": result.expanded_states,
        "平均运行时间/ms": f"{result.runtime_ms:.3f}",
        "路径节点序列": path_to_text(result.path),
    }


def run(output_dir: Path = Path("results"), repeats: int = 30) -> list[Path]:
    records: list[dict[str, object]] = []
    for scene_index, scenario in enumerate(BENCHMARK_SCENARIOS, start=1):
        grid = make_benchmark_grid(ROWS, COLS, START, GOAL, scenario.obstacle_ratio, scenario.seed)
        for _name, func in ALGORITHMS:
            result = run_with_average_time(func, grid, START, GOAL, repeats=repeats)
            records.append(
                make_record(
                    result=result,
                    grid=grid,
                    start=START,
                    goal=GOAL,
                    test_id=f"B{scene_index:03d}",
                    source="内置三场景批量对比实验",
                    scenario=scenario.name,
                    scenario_description=scenario.description,
                    obstacle_ratio_setting=f"{scenario.obstacle_ratio:.2f}",
                    seed=scenario.seed,
                )
            )
    return export_result_package(records, prefix="batch_experiment", output_dir=output_dir, export_png_charts=True)


if __name__ == "__main__":
    outputs = run(Path("results"), repeats=30)
    for output in outputs:
        print(f"saved {output}")
