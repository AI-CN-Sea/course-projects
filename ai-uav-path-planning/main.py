from __future__ import annotations

import random
import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, ttk
from typing import Callable

from benchmark_config import (
    BENCHMARK_SCENARIOS,
    COLS,
    GOAL,
    ROWS,
    START,
    clone_grid,
    corridor_points,
    make_benchmark_grid,
)
from export_utils import export_result_package
from path_algorithms import Grid, Point, SearchResult, astar, dijkstra, improved_astar

CELL_SIZE = 24

COLORS = {
    "free": "#f8fafc",
    "grid": "#cbd5e1",
    "obstacle": "#334155",
    "start": "#16a34a",
    "goal": "#dc2626",
    "visited": "#93c5fd",
    "path": "#f59e0b",
    "drone": "#7c3aed",
}

ALGORITHM_EXPLAIN = {
    "Dijkstra": "Dijkstra：只使用已走代价 g(n)，不使用目标启发信息，能作为最短路径搜索基线，但通常搜索范围更大。",
    "A*": "原始 A*：使用 f(n)=g(n)+h(n)，用曼哈顿距离引导搜索，通常比 Dijkstra 扩展更少节点。",
    "改进A*": "改进 A*：把“进入方向”加入状态，并在搜索代价中加入转弯惩罚，使路径更关注飞行平稳性。",
}

ALGORITHMS: tuple[tuple[str, Callable[[Grid, Point, Point], SearchResult]], ...] = (
    ("Dijkstra", dijkstra),
    ("A*", astar),
    ("改进A*", improved_astar),
)


class PlannerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("无人机动态避障路径规划模拟系统")
        self.resizable(False, False)

        self.grid_data: Grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.start: Point = START
        self.goal: Point = GOAL
        self.mode = tk.StringVar(value="obstacle")
        self.status = tk.StringVar(value="系统就绪：可编辑地图，或直接运行三算法路径规划对比。")
        self.stats = tk.StringVar(value="暂无实验结果。单算法指标会在最终路径动画结束后显示；算法对比会显示三算法统计表。")
        self.is_animating = False
        self.run_counter = 0

        self.last_result: SearchResult | None = None
        self.current_results: list[SearchResult] = []
        self.current_records: list[dict[str, object]] = []
        self.current_grid_snapshot: Grid | None = None
        self.comparison_window: tk.Toplevel | None = None
        self.table_window: tk.Toplevel | None = None

        self._build_layout()
        self.random_map(clear_path=True, ratio=0.16, seed=20260617, update_status=False)
        self.draw_grid()

    # ------------------------- 界面 -------------------------
    def _build_layout(self) -> None:
        main = ttk.Frame(self, padding=10)
        main.grid(row=0, column=0)

        title = ttk.Label(
            main,
            text="无人机路径规划模拟系统：Dijkstra、A*、改进A* 三算法对比与动态避障",
            font=("Microsoft YaHei UI", 14, "bold"),
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.canvas = tk.Canvas(
            main,
            width=COLS * CELL_SIZE,
            height=ROWS * CELL_SIZE,
            bg=COLORS["free"],
            highlightthickness=1,
            highlightbackground="#94a3b8",
        )
        self.canvas.grid(row=1, column=0, rowspan=2)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)

        panel = ttk.Frame(main, padding=(12, 0, 0, 0), width=370)
        panel.grid(row=1, column=1, sticky="n")
        panel.grid_columnconfigure(0, weight=1)

        row = 0
        ttk.Label(panel, text="地图编辑模式", font=("Microsoft YaHei UI", 10, "bold")).grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1
        for text, value in (("添加障碍物", "obstacle"), ("设置起点", "start"), ("设置终点", "goal"), ("擦除网格", "erase")):
            ttk.Radiobutton(panel, text=text, variable=self.mode, value=value).grid(row=row, column=0, sticky="w")
            row += 1

        ttk.Separator(panel).grid(row=row, column=0, sticky="ew", pady=10)
        row += 1
        ttk.Label(panel, text="实验说明", font=("Microsoft YaHei UI", 10, "bold")).grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1
        explain = (
            "算法对比实验固定同时运行 Dijkstra、A*、改进A* 三种算法。\n"
            "当前导出：导出当前这一次运行/对比的数据。\n"
            "内置批量导出：固定生成普通障碍、密集障碍、高密障碍三组静态场景，用于报告中的统计表和条形图。"
        )
        ttk.Label(panel, text=explain, wraplength=350).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, sticky="ew", pady=10)
        row += 1
        buttons = [
            ("运行 Dijkstra 算法", self.run_dijkstra),
            ("运行 A* 算法", self.run_astar),
            ("运行 改进A* 算法", self.run_improved_astar),
            ("算法对比实验（三算法/当前地图）", self.compare_algorithms),
            ("动态避障重规划演示", self.dynamic_demo),
            ("查看当前统计表", self.show_current_table),
            ("查看内置批量场景说明", self.show_benchmark_info),
            ("导出当前CSV+PNG分析图", self.export_current_data),
            ("导出内置批量CSV+PNG分析图", self.export_batch_experiments),
            ("生成随机巡检地图", lambda: self.random_map(clear_path=True)),
            ("清除障碍物", self.clear_obstacles),
            ("清除路径显示", self.clear_path_only),
        ]
        for text, command in buttons:
            ttk.Button(panel, text=text, command=command, width=34).grid(row=row, column=0, sticky="ew", pady=2)
            row += 1

        ttk.Separator(panel).grid(row=row, column=0, sticky="ew", pady=10)
        row += 1
        ttk.Label(panel, text="运行状态", font=("Microsoft YaHei UI", 10, "bold")).grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Label(panel, textvariable=self.status, wraplength=350).grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Label(panel, text="实验指标", font=("Microsoft YaHei UI", 10, "bold")).grid(row=row, column=0, sticky="w", pady=(12, 0))
        row += 1
        ttk.Label(panel, textvariable=self.stats, wraplength=350).grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Separator(panel).grid(row=row, column=0, sticky="ew", pady=10)
        row += 1
        ttk.Label(panel, text="图例", font=("Microsoft YaHei UI", 10, "bold")).grid(row=row, column=0, sticky="w")
        row += 1
        legend = "绿色：起点  红色：终点\n深灰：障碍物  蓝色：搜索区域\n橙色：最终路径  紫色：无人机位置"
        ttk.Label(panel, text=legend, wraplength=350).grid(row=row, column=0, sticky="w")

    # ------------------------- 地图交互 -------------------------
    def on_canvas_click(self, event: tk.Event) -> None:
        self.apply_cell_edit(event)

    def on_canvas_drag(self, event: tk.Event) -> None:
        if self.mode.get() in {"obstacle", "erase"}:
            self.apply_cell_edit(event)

    def apply_cell_edit(self, event: tk.Event) -> None:
        if self.is_animating:
            return
        point = self.event_to_point(event)
        if point is None:
            return
        row, col = point
        mode = self.mode.get()

        if mode == "start" and point != self.goal:
            self.grid_data[self.start[0]][self.start[1]] = 0
            self.start = point
            self.grid_data[row][col] = 0
        elif mode == "goal" and point != self.start:
            self.grid_data[self.goal[0]][self.goal[1]] = 0
            self.goal = point
            self.grid_data[row][col] = 0
        elif mode == "obstacle" and point not in (self.start, self.goal):
            self.grid_data[row][col] = 1
        elif mode == "erase":
            self.grid_data[row][col] = 0

        self.clear_path_only(redraw=False)
        self.draw_grid()

    def event_to_point(self, event: tk.Event) -> Point | None:
        col = int(event.x // CELL_SIZE)
        row = int(event.y // CELL_SIZE)
        if 0 <= row < ROWS and 0 <= col < COLS:
            return row, col
        return None

    def draw_grid(self, visited: set[Point] | None = None, path: set[Point] | None = None) -> None:
        visited = visited or set()
        path = path or set()
        self.canvas.delete("cell")
        self.canvas.delete("drone")

        for row in range(ROWS):
            for col in range(COLS):
                point = (row, col)
                color = COLORS["free"]
                if self.grid_data[row][col] == 1:
                    color = COLORS["obstacle"]
                if point in visited:
                    color = COLORS["visited"]
                if point in path:
                    color = COLORS["path"]
                if point == self.start:
                    color = COLORS["start"]
                if point == self.goal:
                    color = COLORS["goal"]

                x1 = col * CELL_SIZE
                y1 = row * CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x1 + CELL_SIZE, y1 + CELL_SIZE, fill=color, outline=COLORS["grid"], tags="cell")

    def draw_drone(self, point: Point) -> None:
        self.canvas.delete("drone")
        row, col = point
        margin = 5
        x1 = col * CELL_SIZE + margin
        y1 = row * CELL_SIZE + margin
        x2 = (col + 1) * CELL_SIZE - margin
        y2 = (row + 1) * CELL_SIZE - margin
        self.canvas.create_oval(x1, y1, x2, y2, fill=COLORS["drone"], outline="#ffffff", tags="drone")

    # ------------------------- 单算法与三算法对比 -------------------------
    def run_dijkstra(self) -> None:
        self.run_single_algorithm("Dijkstra", dijkstra)

    def run_astar(self) -> None:
        self.run_single_algorithm("A*", astar)

    def run_improved_astar(self) -> None:
        self.run_single_algorithm("改进A*", improved_astar)

    def run_single_algorithm(self, algorithm_name: str, func: Callable[[Grid, Point, Point], SearchResult]) -> None:
        if self.is_animating:
            return
        self.run_counter += 1
        self.current_grid_snapshot = clone_grid(self.grid_data)
        result = run_with_average_time(func, self.grid_data, self.start, self.goal, repeats=10)
        self.current_results = [result]
        self.current_records = []
        self.last_result = result
        self.status.set(f"正在演示 {algorithm_name} 的搜索过程；最终路径确定后再显示统计指标。")
        self.stats.set("搜索动画运行中……\n最终路径确定后显示路径长度、转折次数、综合代价、搜索格数、扩展状态数和平均运行时间。")
        self.animate_result(result, after_done=lambda: self.finish_single_run(result))

    def finish_single_run(self, result: SearchResult) -> None:
        self.current_records = [self.make_record(result, source="单算法演示", scenario="当前地图", grid=self.current_grid_snapshot)]
        self.status.set(f"{result.algorithm} 演示完成：最终路径已确定，可以查看统计表或导出当前结果。")
        self.stats.set(self.format_result(result))

    def compare_algorithms(self) -> None:
        if self.is_animating:
            return
        self.run_counter += 1
        self.current_grid_snapshot = clone_grid(self.grid_data)
        results = self.run_all_algorithms(self.grid_data, self.start, self.goal, repeats=30)
        preferred = self.pick_default_display_result(results)
        self.current_results = results
        self.current_records = []
        self.last_result = preferred
        self.status.set(f"三算法对比实验已计算完成，正在演示默认显示算法：{preferred.algorithm}。")
        self.stats.set("对比指标将在默认路径动画完成后显示；弹出的统计表可切换查看三种算法路径。")
        self.animate_result(preferred, after_done=lambda: self.finish_comparison(results, preferred))

    def finish_comparison(self, results: list[SearchResult], preferred: SearchResult) -> None:
        self.current_records = [self.make_record(result, source="算法对比实验", scenario="当前地图", grid=self.current_grid_snapshot) for result in results]
        self.status.set(f"算法对比实验完成：当前主界面显示 {preferred.algorithm} 路径。")
        self.stats.set(self.format_comparison(results))
        self.show_comparison_window(results)

    def run_all_algorithms(self, grid: Grid, start: Point, goal: Point, repeats: int = 30) -> list[SearchResult]:
        return [run_with_average_time(func, grid, start, goal, repeats=repeats) for _name, func in ALGORITHMS]

    def pick_default_display_result(self, results: list[SearchResult]) -> SearchResult:
        for result in results:
            if result.algorithm == "改进A*" and result.success:
                return result
        for result in results:
            if result.success:
                return result
        return results[0]

    def show_comparison_window(self, results: list[SearchResult]) -> None:
        if self.comparison_window is not None and self.comparison_window.winfo_exists():
            self.comparison_window.destroy()

        win = tk.Toplevel(self)
        self.comparison_window = win
        win.title("三算法对比实验结果（当前地图）")
        win.resizable(False, False)
        container = ttk.Frame(win, padding=10)
        container.grid(row=0, column=0, sticky="nsew")

        intro = (
            "本表是当前地图、当前起点和终点下的一次三算法对比实验。"
            "三种算法固定同时运行，指标口径一致；运行时间为重复搜索平均值，不包含动画播放时间。"
        )
        ttk.Label(container, text=intro, wraplength=880).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 8))
        self._insert_result_tree(container, results, row=1, height=max(3, len(results)))

        ttk.Label(container, text="算法说明", font=("Microsoft YaHei UI", 10, "bold")).grid(row=2, column=0, sticky="w", pady=(10, 2))
        explain = "\n".join(ALGORITHM_EXPLAIN.get(result.algorithm, result.algorithm) for result in results)
        ttk.Label(container, text=explain, wraplength=880).grid(row=3, column=0, columnspan=5, sticky="w")

        button_frame = ttk.Frame(container)
        button_frame.grid(row=4, column=0, columnspan=5, sticky="w", pady=(10, 0))
        for index, result in enumerate(results):
            ttk.Button(button_frame, text=f"显示 {result.algorithm} 路径", command=lambda item=result: self.display_comparison_result(item)).grid(row=0, column=index, padx=(0, 8))
        ttk.Button(button_frame, text="导出当前CSV+PNG分析图", command=self.export_current_data).grid(row=0, column=len(results), padx=(10, 0))

    def display_comparison_result(self, result: SearchResult) -> None:
        if self.is_animating:
            self.status.set("动画仍在播放，播放结束后再切换对比算法路径。")
            return
        self.last_result = result
        self.status.set(f"正在切换显示三算法对比实验中的 {result.algorithm} 路径。")
        self.stats.set("路径切换动画运行中，结束后显示该算法指标。")
        self.animate_result(result, after_done=lambda: self.finish_display_comparison_result(result))

    def finish_display_comparison_result(self, result: SearchResult) -> None:
        self.status.set(f"当前显示：三算法对比实验中的 {result.algorithm} 路径。")
        self.stats.set(self.format_result(result))

    # ------------------------- 动态避障 -------------------------
    def dynamic_demo(self) -> None:
        if self.is_animating:
            return
        self.run_counter += 1
        self.current_grid_snapshot = clone_grid(self.grid_data)
        initial_result = run_with_average_time(improved_astar, self.grid_data, self.start, self.goal, repeats=10)
        if not initial_result.success:
            self.current_results = [initial_result]
            self.current_records = [self.make_record(initial_result, source="动态避障初始规划", scenario="当前地图")]
            self.last_result = initial_result
            self.status.set("动态避障演示失败：初始地图不可达。")
            self.stats.set(self.format_result(initial_result))
            messagebox.showwarning("路径不可达", "改进A* 未找到初始路径，请调整障碍物分布。")
            return

        self.current_results = [initial_result]
        self.current_records = [self.make_record(initial_result, source="动态避障初始规划", scenario="当前地图")]
        self.last_result = initial_result
        self.is_animating = True
        self.status.set("动态避障演示运行中：飞行过程中会加入临时障碍并触发改进A*重新规划。")
        self.stats.set("动态飞行中……最终到达终点后再显示总路径、重规划次数和统计结果。")

        state = {"path": initial_result.path, "index": 0, "replans": 0, "travelled": [initial_result.path[0]]}

        def move() -> None:
            path = state["path"]
            index = state["index"]
            if index >= len(path):
                self.is_animating = False
                self.status.set(f"动态避障演示完成：无人机到达终点，累计重新规划 {state['replans']} 次。")
                self.stats.set(self.format_dynamic_summary(state["travelled"], int(state["replans"])))
                return

            position = path[index]
            if not state["travelled"] or state["travelled"][-1] != position:
                state["travelled"].append(position)
            self.draw_grid(path=set(path) | set(state["travelled"]))
            self.draw_drone(position)

            if index > 0 and index % 7 == 0 and state["replans"] < 3:
                blocked = self.pick_future_path_cell(path, index)
                if blocked:
                    self.grid_data[blocked[0]][blocked[1]] = 1
                    state["replans"] += 1
                    new_result = run_with_average_time(improved_astar, self.grid_data, position, self.goal, repeats=10)
                    self.current_results.append(new_result)
                    self.current_records.append(self.make_record(new_result, source=f"动态避障第{state['replans']}次重规划", scenario="当前地图"))
                    self.status.set(f"检测到动态障碍 {blocked}，已触发第 {state['replans']} 次重新规划。")
                    if new_result.success:
                        self.last_result = new_result
                        state["path"] = new_result.path
                        state["index"] = 0
                    else:
                        self.is_animating = False
                        self.last_result = new_result
                        self.status.set("动态障碍阻断所有可行路径，任务终止。")
                        self.stats.set(self.format_result(new_result))
                        return
                else:
                    state["index"] += 1
            else:
                state["index"] += 1
            self.after(120, move)

        move()

    def pick_future_path_cell(self, path: list[Point], current_index: int) -> Point | None:
        candidates = [point for point in path[current_index + 3 : current_index + 8] if point not in (self.start, self.goal)]
        return candidates[len(candidates) // 2] if candidates else None

    # ------------------------- 动画 -------------------------
    def animate_result(self, result: SearchResult, after_done: Callable[[], None] | None = None) -> None:
        self.is_animating = True
        visited_seen: set[Point] = set()
        path_set = set(result.path)

        def search_step(index: int) -> None:
            if index < len(result.visited_order):
                visited_seen.add(result.visited_order[index])
                self.draw_grid(visited_seen, set())
                self.after(5, lambda: search_step(index + 1))
            else:
                self.draw_grid(visited_seen, path_set)
                if result.success and result.path:
                    self.after(120, lambda: drone_step(0))
                else:
                    finish()

        def drone_step(index: int) -> None:
            if index < len(result.path):
                self.draw_grid(visited_seen, path_set)
                self.draw_drone(result.path[index])
                self.after(40, lambda: drone_step(index + 1))
            else:
                finish()

        def finish() -> None:
            self.draw_grid(visited_seen, path_set)
            if result.success and result.path:
                self.draw_drone(result.path[-1])
            self.is_animating = False
            if after_done is not None:
                after_done()

        search_step(0)

    # ------------------------- 地图控制 -------------------------
    def random_map(self, clear_path: bool = True, ratio: float = 0.18, seed: int | None = None, update_status: bool = True) -> None:
        if self.is_animating:
            return
        rng = random.Random(seed)
        for row in range(ROWS):
            for col in range(COLS):
                point = (row, col)
                if point in (self.start, self.goal):
                    self.grid_data[row][col] = 0
                else:
                    self.grid_data[row][col] = 1 if rng.random() < ratio else 0
        if clear_path:
            self.keep_basic_corridor()
        self.clear_path_only(redraw=False)
        if update_status:
            self.status.set("已生成随机巡检地图。")
        self.draw_grid()

    def keep_basic_corridor(self) -> None:
        for row, col in corridor_points(self.start, self.goal):
            self.grid_data[row][col] = 0
            for nr, nc in ((row + 1, col), (row, col + 1), (row - 1, col), (row, col - 1)):
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    self.grid_data[nr][nc] = 0

    def clear_obstacles(self) -> None:
        if self.is_animating:
            return
        self.grid_data = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.clear_path_only(redraw=False)
        self.status.set("已清除全部障碍物。")
        self.draw_grid()

    def clear_path_only(self, redraw: bool = True) -> None:
        self.canvas.delete("drone")
        self.last_result = None
        self.current_results = []
        self.current_records = []
        self.current_grid_snapshot = None
        self.stats.set("暂无实验结果。单算法指标会在最终路径动画结束后显示；算法对比会显示三算法统计表。")
        if redraw:
            self.draw_grid()

    # ------------------------- 统计表与导出 -------------------------
    def make_record(
        self,
        result: SearchResult,
        source: str,
        scenario: str,
        grid: Grid | None = None,
        obstacle_ratio_setting: str = "当前地图",
        start: Point | None = None,
        goal: Point | None = None,
        scenario_description: str = "用户当前编辑地图",
        seed: int | str = "无",
    ) -> dict[str, object]:
        grid_for_record = grid if grid is not None else self.grid_data
        start = start if start is not None else self.start
        goal = goal if goal is not None else self.goal
        obstacles = count_obstacles(grid_for_record)
        total_cells = len(grid_for_record) * len(grid_for_record[0]) if grid_for_record and grid_for_record[0] else 1
        return {
            "实验编号": f"T{self.run_counter:04d}",
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

    def show_current_table(self) -> None:
        if not self.current_results and not self.current_records:
            messagebox.showinfo("暂无数据", "请先运行单算法演示、三算法对比实验、动态避障演示或内置批量实验。")
            return
        if self.table_window is not None and self.table_window.winfo_exists():
            self.table_window.destroy()
        win = tk.Toplevel(self)
        self.table_window = win
        win.title("当前实验统计表")
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=10)
        frame.grid(row=0, column=0)
        ttk.Label(frame, text="当前实验统计表", font=("Microsoft YaHei UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))
        if self.current_records:
            self._insert_record_tree(frame, self.current_records, row=1, height=max(3, min(12, len(self.current_records))))
        else:
            self._insert_result_tree(frame, self.current_results, row=1, height=max(3, len(self.current_results)))

    def _insert_record_tree(self, container: ttk.Frame, records: list[dict[str, object]], row: int, height: int) -> None:
        columns = ("scenario", "source", "algorithm", "success", "length", "turns", "cost", "cells", "states", "time")
        tree = ttk.Treeview(container, columns=columns, show="headings", height=height)
        headings = {
            "scenario": "实验场景",
            "source": "数据来源",
            "algorithm": "算法",
            "success": "成功",
            "length": "路径长度",
            "turns": "转折次数",
            "cost": "综合代价",
            "cells": "搜索格数",
            "states": "扩展状态数",
            "time": "平均运行时间/ms",
        }
        widths = {
            "scenario": 92,
            "source": 150,
            "algorithm": 90,
            "success": 54,
            "length": 78,
            "turns": 78,
            "cost": 78,
            "cells": 78,
            "states": 90,
            "time": 120,
        }
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(col, width=widths[col], anchor="center")
        for record in records:
            tree.insert(
                "",
                "end",
                values=(
                    record.get("实验场景", ""),
                    record.get("数据来源", ""),
                    record.get("算法", ""),
                    record.get("是否成功", ""),
                    record.get("路径长度", ""),
                    record.get("转折次数", ""),
                    record.get("综合代价", ""),
                    record.get("搜索格数", ""),
                    record.get("扩展状态数", ""),
                    record.get("平均运行时间/ms", ""),
                ),
            )
        tree.grid(row=row, column=0, sticky="ew")

    def _insert_result_tree(self, container: ttk.Frame, results: list[SearchResult], row: int, height: int) -> None:
        columns = ("algorithm", "success", "length", "turns", "cost", "cells", "states", "time")
        tree = ttk.Treeview(container, columns=columns, show="headings", height=height)
        headings = {
            "algorithm": "算法",
            "success": "成功",
            "length": "路径长度",
            "turns": "转折次数",
            "cost": "综合代价",
            "cells": "搜索格数",
            "states": "扩展状态数",
            "time": "平均运行时间/ms",
        }
        widths = {"algorithm": 96, "success": 60, "length": 82, "turns": 82, "cost": 82, "cells": 82, "states": 94, "time": 122}
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(col, width=widths[col], anchor="center")
        for result in results:
            tree.insert(
                "",
                "end",
                values=(
                    result.algorithm,
                    "是" if result.success else "否",
                    result.distance,
                    result.turn_count,
                    f"{result.route_cost:.2f}",
                    result.searched_nodes,
                    result.expanded_states,
                    f"{result.runtime_ms:.3f}",
                ),
            )
        tree.grid(row=row, column=0, sticky="ew")

    def show_benchmark_info(self) -> None:
        win = tk.Toplevel(self)
        win.title("内置批量实验场景说明")
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=10)
        frame.grid(row=0, column=0)
        ttk.Label(
            frame,
            text="内置批量实验固定使用下面三组静态场景。它们由程序自动生成，不需要在界面中额外手动选择障碍密度。",
            wraplength=760,
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        columns = ("scenario", "ratio", "seed", "desc")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=len(BENCHMARK_SCENARIOS))
        headings = {"scenario": "实验场景", "ratio": "设定障碍比例", "seed": "随机种子", "desc": "实验目的"}
        widths = {"scenario": 100, "ratio": 110, "seed": 120, "desc": 440}
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(col, width=widths[col], anchor="center" if col != "desc" else "w")
        for scene in BENCHMARK_SCENARIOS:
            tree.insert("", "end", values=(scene.name, f"{scene.obstacle_ratio:.2f}", scene.seed, scene.description))
        tree.grid(row=1, column=0, sticky="ew")

    def export_current_data(self) -> None:
        if self.is_animating:
            messagebox.showinfo("动画未结束", "请等最终路径显示完成后再导出结果。")
            return
        if not self.current_records:
            messagebox.showinfo("暂无数据", "请先运行单算法演示、三算法对比实验或动态避障演示。")
            return
        outputs = export_result_package(self.current_records, prefix="current_experiment", export_png_charts=True)
        self._show_export_status(outputs, "当前实验")

    def export_batch_experiments(self) -> None:
        if self.is_animating:
            messagebox.showinfo("动画未结束", "请等当前动画结束后再导出批量实验。")
            return
        records: list[dict[str, object]] = []
        original_grid = clone_grid(self.grid_data)
        original_start, original_goal = self.start, self.goal

        try:
            for scene_index, scenario in enumerate(BENCHMARK_SCENARIOS, start=1):
                grid = make_benchmark_grid(ROWS, COLS, original_start, original_goal, scenario.obstacle_ratio, scenario.seed)
                self.run_counter += 1
                for result in self.run_all_algorithms(grid, original_start, original_goal, repeats=30):
                    records.append(
                        self.make_record(
                            result,
                            source="内置三场景批量对比实验",
                            scenario=scenario.name,
                            grid=grid,
                            obstacle_ratio_setting=f"{scenario.obstacle_ratio:.2f}",
                            start=original_start,
                            goal=original_goal,
                            scenario_description=scenario.description,
                            seed=scenario.seed,
                        )
                    )
        finally:
            self.grid_data = original_grid
            self.start, self.goal = original_start, original_goal
            self.draw_grid()

        outputs = export_result_package(records, prefix="batch_experiment", export_png_charts=True)
        self.current_results = []
        self.current_records = records
        self._show_export_status(outputs, "内置批量实验")
        self.stats.set(
            "批量实验已完成：3 个内置静态场景 × 3 种算法，共 9 条记录。\n"
            "导出的 CSV 是报告中的实验结果表；PNG 是路径长度、转折次数、搜索格数、扩展状态数、综合代价和平均运行时间条形图。"
        )

    def _show_export_status(self, outputs: list[object], label: str) -> None:
        png_count = sum(1 for path in outputs if str(path).lower().endswith(".png"))
        csv_count = sum(1 for path in outputs if str(path).lower().endswith(".csv"))
        if png_count:
            message = f"{label}导出完成：{csv_count} 个 CSV，{png_count} 张 PNG 分析图。"
        else:
            message = f"{label}导出完成：{csv_count} 个 CSV。本机未成功生成 PNG/JPG 分析图，因此已按要求跳过图片导出。"
        paths_text = "；".join(str(path) for path in outputs)
        self.status.set(message + "\n" + paths_text)
        if not png_count:
            messagebox.showinfo("导出完成", message + "\n如需图片，请安装 matplotlib：pip install matplotlib")

    # ------------------------- 格式化展示 -------------------------
    def format_result(self, result: SearchResult) -> str:
        if not result.success:
            return (
                f"算法：{result.algorithm}\n"
                "状态：未找到可行路径\n"
                f"失败原因：{result.message}\n"
                f"搜索格数：{result.searched_nodes}\n"
                f"扩展状态数：{result.expanded_states}\n"
                f"平均运行时间：{result.runtime_ms:.3f} ms\n"
                f"说明：{ALGORITHM_EXPLAIN.get(result.algorithm, '')}"
            )
        return (
            f"算法：{result.algorithm}\n"
            f"路径长度：{result.distance}\n"
            f"转折次数：{result.turn_count}\n"
            f"综合代价：{result.route_cost:.2f}\n"
            f"搜索格数：{result.searched_nodes}\n"
            f"扩展状态数：{result.expanded_states}\n"
            f"平均运行时间：{result.runtime_ms:.3f} ms\n"
            "时间说明：只统计算法计算时间，不包含搜索动画和无人机飞行动画。\n"
            f"说明：{ALGORITHM_EXPLAIN.get(result.algorithm, '')}"
        )

    def format_comparison(self, results: list[SearchResult]) -> str:
        lines = ["三算法对比实验：同一张地图、同一起点终点，固定对比 Dijkstra、A*、改进A*。"]
        lines.append("运行时间为重复搜索平均值，不包含动画；路径长度、转折次数等来自最终规划路径。")
        for result in results:
            lines.append(
                f"{result.algorithm}：成功={result.success}，长度={result.distance}，"
                f"转折={result.turn_count}，代价={result.route_cost:.2f}，"
                f"搜索格={result.searched_nodes}，状态={result.expanded_states}，平均耗时={result.runtime_ms:.3f} ms"
            )
        return "\n".join(lines)

    def format_dynamic_summary(self, travelled: list[Point], replans: int) -> str:
        return (
            "动态避障演示结果\n"
            f"累计重规划次数：{replans}\n"
            f"累计飞行步数：{max(0, len(travelled) - 1)}\n"
            f"最终位置：{point_to_text(travelled[-1]) if travelled else '未知'}\n"
            f"记录条数：{len(self.current_records)}\n"
            "可点击“查看当前统计表”查看初始规划和每次重规划的指标，也可导出当前 CSV+PNG 分析图。"
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
    """用重复搜索的均值修正运行时间。

    SearchResult 中除 runtime_ms 外的指标来自一次确定性搜索结果；runtime_ms 为 repeats 次
    独立搜索的平均耗时。该时间不包含 GUI 动画、绘图或表格弹窗时间。
    """
    repeats = max(1, repeats)
    representative = func(clone_grid(grid), start, goal)
    total_time = 0.0
    for _ in range(repeats):
        timed = func(clone_grid(grid), start, goal)
        total_time += timed.runtime_ms
    return replace(representative, runtime_ms=total_time / repeats)


if __name__ == "__main__":
    app = PlannerApp()
    app.mainloop()
