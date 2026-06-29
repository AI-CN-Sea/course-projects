from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

# 统一 CSV 字段。导出表格只记录实验统计数据，不再导出路径地图截图。
CSV_FIELDS = [
    "实验编号",
    "数据来源",
    "实验场景",
    "场景说明",
    "随机种子",
    "算法",
    "是否成功",
    "起点",
    "终点",
    "设定障碍比例",
    "实际障碍数量",
    "实际障碍比例",
    "路径长度",
    "转折次数",
    "综合代价",
    "搜索格数",
    "扩展状态数",
    "平均运行时间/ms",
    "路径节点序列",
]

CHART_METRICS = [
    ("路径长度", "path_length", "路径长度对比"),
    ("转折次数", "turn_count", "路径转折次数对比"),
    ("搜索格数", "searched_cells", "算法搜索格数对比"),
    ("扩展状态数", "expanded_states", "算法扩展状态数对比"),
    ("综合代价", "route_cost", "综合路径代价对比"),
    ("平均运行时间/ms", "runtime_ms", "算法平均运行时间对比"),
]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_output_dir(output_dir: str | Path = "results") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_records_csv(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in CSV_FIELDS})
    return path


def export_result_package(
    records: list[dict[str, Any]],
    prefix: str,
    output_dir: str | Path = "results",
    export_png_charts: bool = True,
) -> list[Path]:
    """导出实验结果。

    输出内容：
    1. CSV 统计表：始终导出；
    2. PNG 分析条形图：仅在 matplotlib 可用时导出；如果本机无法生成 PNG，则自动跳过图片。

    本函数不再导出路径地图图片。路径地图建议直接使用 GUI 运行截图，避免“当前地图截图”
    和“实验统计图”混在一起造成结果来源不清。
    """
    out_dir = ensure_output_dir(output_dir)
    stamp = timestamp()
    outputs: list[Path] = []
    csv_path = out_dir / f"{prefix}_{stamp}.csv"
    outputs.append(write_records_csv(records, csv_path))
    if export_png_charts and records:
        outputs.extend(write_analysis_png_charts(records, prefix=prefix, stamp=stamp, output_dir=out_dir))
    return outputs


def write_analysis_png_charts(
    records: list[dict[str, Any]],
    prefix: str,
    stamp: str | None = None,
    output_dir: str | Path = "results",
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception:
        # 用户要求：如果不能导出 png/jpg，则不导出图片内容。
        return []

    _configure_chinese_font(plt, font_manager)
    out_dir = ensure_output_dir(output_dir)
    stamp = stamp or timestamp()
    outputs: list[Path] = []
    for metric_key, suffix, title in CHART_METRICS:
        path = out_dir / f"{prefix}_{suffix}_{stamp}.png"
        try:
            write_metric_bar_png(records, metric_key, path, title, plt)
        except Exception:
            # 单张图失败时跳过，不影响 CSV 和其它图。
            continue
        outputs.append(path)
    return outputs


def write_metric_bar_png(records: list[dict[str, Any]], metric_key: str, output_path: str | Path, title: str, plt: Any) -> Path:
    scenarios = _unique([str(record.get("实验场景", "当前地图")) for record in records])
    algorithms = _unique([str(record.get("算法", "")) for record in records])

    # 批量实验：按“场景 × 算法”做分组柱状图。
    if len(scenarios) > 1 and len(algorithms) > 1:
        fig_width = max(8.5, len(scenarios) * 2.2)
        fig, ax = plt.subplots(figsize=(fig_width, 5.2))
        x_positions = list(range(len(scenarios)))
        bar_width = min(0.24, 0.75 / max(1, len(algorithms)))
        offsets = [(i - (len(algorithms) - 1) / 2) * bar_width for i in range(len(algorithms))]

        for alg_index, algorithm in enumerate(algorithms):
            values: list[float] = []
            for scenario in scenarios:
                record = _find_record(records, scenario, algorithm)
                values.append(_to_float(record.get(metric_key, 0)) if record else 0.0)
            positions = [x + offsets[alg_index] for x in x_positions]
            ax.bar(positions, values, width=bar_width, label=algorithm)
            for x, value in zip(positions, values):
                ax.text(x, value, _format_metric(value), ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(scenarios)
        ax.set_xlabel("内置静态实验场景")
        ax.set_ylabel(metric_key)
        ax.legend(loc="best")
    else:
        # 当前地图：按算法直接对比。
        labels = [str(record.get("算法", record.get("实验场景", ""))) for record in records]
        values = [_to_float(record.get(metric_key, 0)) for record in records]
        fig_width = max(7.5, len(labels) * 1.6)
        fig, ax = plt.subplots(figsize=(fig_width, 5.0))
        bars = ax.bar(labels, values)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), _format_metric(value), ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("算法")
        ax.set_ylabel(metric_key)

    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.text(
        0.01,
        0.01,
        "说明：运行时间为多次重复搜索的平均值，不包含界面动画播放时间；其它指标来自最终规划路径。",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _configure_chinese_font(plt: Any, font_manager: Any) -> None:
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _find_record(records: list[dict[str, Any]], scenario: str, algorithm: str) -> dict[str, Any] | None:
    for record in records:
        if str(record.get("实验场景", "")) == scenario and str(record.get("算法", "")) == algorithm:
            return record
    return None


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_metric(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}"
