from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from time import perf_counter
from typing import Callable, Iterable

Point = tuple[int, int]
Grid = list[list[int]]
Direction = tuple[int, int] | None
State = tuple[Point, Direction]
TURN_PENALTY = 0.6
INF = 10**18


@dataclass(frozen=True)
class SearchResult:
    """一次路径搜索的完整结果。

    distance: 路径步数，等于 len(path)-1。
    searched_nodes: 可视化层面的唯一搜索格子数。
    expanded_states: 算法层面实际弹出的状态数。改进 A* 使用“位置+方向”作为状态，
        因此 expanded_states 可能大于 searched_nodes。
    route_cost: 综合代价 = 路径步数 + 转折次数 * TURN_PENALTY。
    """

    algorithm: str
    success: bool
    path: list[Point]
    visited_order: list[Point]
    distance: int
    searched_nodes: int
    runtime_ms: float
    turn_count: int
    route_cost: float
    expanded_states: int
    message: str = ""


def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def in_bounds(grid: Grid, point: Point) -> bool:
    if not grid or not grid[0]:
        return False
    row, col = point
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])


def is_free(grid: Grid, point: Point) -> bool:
    row, col = point
    return grid[row][col] == 0


def is_rectangular_grid(grid: Grid) -> bool:
    return bool(grid) and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)


def neighbors(grid: Grid, point: Point) -> Iterable[Point]:
    """四邻域移动：上、右、下、左。每一步代价为 1。"""
    row, col = point
    for candidate in ((row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1)):
        if in_bounds(grid, candidate) and is_free(grid, candidate):
            yield candidate


def direction_between(a: Point, b: Point) -> tuple[int, int]:
    return b[0] - a[0], b[1] - a[1]


def count_turns(path: list[Point]) -> int:
    if len(path) < 3:
        return 0
    turns = 0
    previous = direction_between(path[0], path[1])
    for index in range(1, len(path) - 1):
        current = direction_between(path[index], path[index + 1])
        if current != previous:
            turns += 1
        previous = current
    return turns


def route_cost(path: list[Point], turn_penalty: float = TURN_PENALTY) -> float:
    if not path:
        return -1.0
    return max(0, len(path) - 1) + count_turns(path) * turn_penalty


def reconstruct_path(parent: dict[Point, Point], start: Point, goal: Point) -> list[Point]:
    if goal != start and goal not in parent:
        return []
    point = goal
    path = [point]
    while point != start:
        point = parent[point]
        path.append(point)
    path.reverse()
    return path


def reconstruct_state_path(parent: dict[State, State], goal_state: State) -> list[Point]:
    state = goal_state
    states = [state]
    while state in parent:
        state = parent[state]
        states.append(state)
    states.reverse()
    return [point for point, _ in states]


def astar(
    grid: Grid,
    start: Point,
    goal: Point,
    heuristic: Callable[[Point, Point], float] = manhattan,
) -> SearchResult:
    """原始 A*。

    评价函数 f(n)=g(n)+h(n)。在四邻域栅格中，默认曼哈顿距离是可采纳启发函数。
    """
    start_time = perf_counter()
    valid, message = _valid_problem(grid, start, goal)
    if not valid:
        return _failed_result("A*", start_time, message=message)

    open_heap: list[tuple[float, int, Point]] = []
    heappush(open_heap, (heuristic(start, goal), 0, start))
    parent: dict[Point, Point] = {}
    g_score: dict[Point, int] = {start: 0}
    closed: set[Point] = set()
    visited_order: list[Point] = []
    counter = 0

    while open_heap:
        _, _, current = heappop(open_heap)
        if current in closed:
            continue

        closed.add(current)
        visited_order.append(current)

        if current == goal:
            return _success_result("A*", start_time, reconstruct_path(parent, start, goal), visited_order, len(visited_order))

        for nxt in neighbors(grid, current):
            if nxt in closed:
                continue
            tentative = g_score[current] + 1
            if tentative < g_score.get(nxt, INF):
                parent[nxt] = current
                g_score[nxt] = tentative
                counter += 1
                heappush(open_heap, (tentative + heuristic(nxt, goal), counter, nxt))

    return _failed_result("A*", start_time, visited_order, expanded_states=len(visited_order), message="目标点不可达")


def improved_astar(
    grid: Grid,
    start: Point,
    goal: Point,
    heuristic: Callable[[Point, Point], float] = manhattan,
    turn_penalty: float = TURN_PENALTY,
) -> SearchResult:
    """改进 A*：在搜索状态中加入方向，并把转弯惩罚加入 g 代价。

    普通 A* 的状态只有位置 (row, col)，改进 A* 的状态是 ((row, col), direction)。
    这样同一个格子从不同方向进入时会被区别对待，算法会优先选择转弯更少、飞行更平滑的路径。
    """
    start_time = perf_counter()
    valid, message = _valid_problem(grid, start, goal)
    if not valid:
        return _failed_result("改进A*", start_time, message=message)

    start_state: State = (start, None)
    open_heap: list[tuple[float, int, State]] = []
    heappush(open_heap, (heuristic(start, goal), 0, start_state))
    parent: dict[State, State] = {}
    g_score: dict[State, float] = {start_state: 0.0}
    closed_states: set[State] = set()
    visited_points: set[Point] = set()
    visited_order: list[Point] = []
    counter = 0

    while open_heap:
        _, _, current_state = heappop(open_heap)
        if current_state in closed_states:
            continue

        current, current_direction = current_state
        closed_states.add(current_state)
        if current not in visited_points:
            visited_points.add(current)
            visited_order.append(current)

        if current == goal:
            path = reconstruct_state_path(parent, current_state)
            return _success_result("改进A*", start_time, path, visited_order, len(closed_states), turn_penalty)

        for nxt in neighbors(grid, current):
            next_direction = direction_between(current, nxt)
            extra_turn_cost = 0.0
            if current_direction is not None and next_direction != current_direction:
                extra_turn_cost = turn_penalty
            next_state: State = (nxt, next_direction)
            tentative = g_score[current_state] + 1.0 + extra_turn_cost
            if tentative < g_score.get(next_state, INF):
                parent[next_state] = current_state
                g_score[next_state] = tentative
                counter += 1
                heappush(open_heap, (tentative + heuristic(nxt, goal), counter, next_state))

    return _failed_result(
        "改进A*",
        start_time,
        visited_order,
        expanded_states=len(closed_states),
        message="目标点不可达",
    )


def dijkstra(grid: Grid, start: Point, goal: Point) -> SearchResult:
    """Dijkstra 算法：不使用启发函数，作为对比基线。"""
    start_time = perf_counter()
    valid, message = _valid_problem(grid, start, goal)
    if not valid:
        return _failed_result("Dijkstra", start_time, message=message)

    open_heap: list[tuple[int, int, Point]] = []
    heappush(open_heap, (0, 0, start))
    parent: dict[Point, Point] = {}
    dist_score: dict[Point, int] = {start: 0}
    closed: set[Point] = set()
    visited_order: list[Point] = []
    counter = 0

    while open_heap:
        dist, _, current = heappop(open_heap)
        if current in closed:
            continue

        closed.add(current)
        visited_order.append(current)

        if current == goal:
            return _success_result("Dijkstra", start_time, reconstruct_path(parent, start, goal), visited_order, len(visited_order))

        for nxt in neighbors(grid, current):
            if nxt in closed:
                continue
            tentative = dist + 1
            if tentative < dist_score.get(nxt, INF):
                parent[nxt] = current
                dist_score[nxt] = tentative
                counter += 1
                heappush(open_heap, (tentative, counter, nxt))

    return _failed_result("Dijkstra", start_time, visited_order, expanded_states=len(visited_order), message="目标点不可达")


def _success_result(
    algorithm: str,
    start_time: float,
    path: list[Point],
    visited_order: list[Point],
    expanded_states: int,
    turn_penalty: float = TURN_PENALTY,
) -> SearchResult:
    turns = count_turns(path)
    return SearchResult(
        algorithm=algorithm,
        success=True,
        path=path,
        visited_order=visited_order,
        distance=max(0, len(path) - 1),
        searched_nodes=len(visited_order),
        runtime_ms=(perf_counter() - start_time) * 1000,
        turn_count=turns,
        route_cost=route_cost(path, turn_penalty),
        expanded_states=expanded_states,
        message="搜索成功",
    )


def _valid_problem(grid: Grid, start: Point, goal: Point) -> tuple[bool, str]:
    if not is_rectangular_grid(grid):
        return False, "地图为空或不是规则矩形网格"
    if not in_bounds(grid, start):
        return False, "起点不在地图范围内"
    if not in_bounds(grid, goal):
        return False, "终点不在地图范围内"
    if not is_free(grid, start):
        return False, "起点被障碍物占用"
    if not is_free(grid, goal):
        return False, "终点被障碍物占用"
    return True, ""


def _failed_result(
    algorithm: str,
    start_time: float,
    visited_order: list[Point] | None = None,
    expanded_states: int | None = None,
    message: str = "搜索失败",
) -> SearchResult:
    visited = visited_order or []
    return SearchResult(
        algorithm=algorithm,
        success=False,
        path=[],
        visited_order=visited,
        distance=-1,
        searched_nodes=len(visited),
        runtime_ms=(perf_counter() - start_time) * 1000,
        turn_count=0,
        route_cost=-1.0,
        expanded_states=len(visited) if expanded_states is None else expanded_states,
        message=message,
    )
