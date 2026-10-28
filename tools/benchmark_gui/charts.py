from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .results import as_float
from .theme import COLORS, plot_font_supports_cjk


class MetricCharts(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 12)
        layout.setSpacing(6)

        self.hint = QLabel("运行 Benchmark 或刷新结果后，这里会显示性能图表。")
        self.hint.setObjectName("SectionHint")
        layout.addWidget(self.hint)

        self.figure = Figure(figsize=(9, 7), dpi=100, layout="constrained")
        self.figure.set_facecolor(COLORS["surface"])
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)

    def show_results(self, rows, summary):
        self.figure.clear()
        use_chinese = plot_font_supports_cjk()

        if use_chinese:
            per_request_metrics = [
                ("output_tokens_per_s", "单请求输出吞吐量", "tokens/s", "#2563EB"),
                ("ttft_s", "首字延迟 TTFT", "秒", "#0E9F8E"),
                ("tpot_s", "单 Token 延迟 TPOT", "秒/token", "#D97706"),
            ]
            batch_title = "每轮 Batch 吞吐量"
            batch_ylabel = "tokens/s"
            run_xlabel = "Run"
            req_xlabel = "请求序号"
        else:
            per_request_metrics = [
                ("output_tokens_per_s", "Per-request throughput", "tokens/s", "#2563EB"),
                ("ttft_s", "Time to first token", "seconds", "#0E9F8E"),
                ("tpot_s", "Time per output token", "seconds/token", "#D97706"),
            ]
            batch_title = "Batch throughput per run"
            batch_ylabel = "tokens/s"
            run_xlabel = "Run"
            req_xlabel = "Request"

        usable = [
            r for r in rows
            if any(as_float(r.get(k)) is not None for k, _, _, _ in per_request_metrics)
        ]

        # 提取 batch 每轮吞吐量
        batch_points = []
        if isinstance(summary, dict):
            runs = summary.get("runs")
            if isinstance(runs, list):
                for i, run in enumerate(runs, start=1):
                    if not isinstance(run, dict):
                        continue
                    val = as_float(run.get("aggregate_output_tokens_per_s"))
                    if val is not None:
                        batch_points.append((i, val))

        if batch_points:
            self.hint.setText(
                f"已加载 {len(usable)} 条请求记录 · {len(batch_points)} 轮 Batch 吞吐量"
                if use_chinese
                else f"{len(usable)} request records · {len(batch_points)} batch runs"
            )
        else:
            self.hint.setText(
                f"已加载 {len(usable)} 条请求记录"
                if use_chinese
                else f"{len(usable)} request records"
            )

        # 第一张图：Batch 吞吐量按 run
        ax_batch = self.figure.add_subplot(4, 1, 1)
        if batch_points:
            bx = [p[0] for p in batch_points]
            by = [p[1] for p in batch_points]
            ax_batch.plot(
                bx, by,
                color="#7C3AED",
                linewidth=2.0,
                marker="s",
                markersize=6,
                markerfacecolor="#FFFFFF",
                markeredgewidth=1.8,
                label=batch_title,
            )
            for x in bx:
                ax_batch.axvline(x, color="#E5EAF1", linewidth=0.8, zorder=0)
            ax_batch.legend(loc="upper right", frameon=False, fontsize=9)
        else:
            ax_batch.text(
                0.5, 0.5,
                "暂无 Batch 数据" if use_chinese else "No batch data",
                transform=ax_batch.transAxes,
                ha="center", va="center",
                color=COLORS["muted"], fontsize=11,
            )
        self._style_axis(ax_batch, batch_title, batch_ylabel, run_xlabel, COLORS["muted"])

        # 后三张图：per-request 趋势
        axes = [self.figure.add_subplot(4, 1, i) for i in range(2, 5)]
        x = list(range(1, len(usable) + 1))
        for ax, (key, title, unit, color) in zip(axes, per_request_metrics):
            y = [as_float(row.get(key)) for row in usable]
            pairs = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
            if pairs:
                px = [p[0] for p in pairs]
                py = [p[1] for p in pairs]
                ax.plot(
                    px, py,
                    color=color,
                    linewidth=2.0,
                    marker="o",
                    markersize=5,
                    markerfacecolor="#FFFFFF",
                    markeredgewidth=1.8,
                    label=title,
                )
                if len(py) > 1:
                    ax.fill_between(px, py, color=color, alpha=0.08)
                ax.legend(loc="upper right", frameon=False, fontsize=9)
            else:
                ax.text(
                    0.5, 0.5,
                    "暂无数据" if use_chinese else "No data",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    color=COLORS["muted"], fontsize=11,
                )
            self._style_axis(ax, title, unit, req_xlabel, color)

        self.canvas.draw()

    @staticmethod
    def _style_axis(ax, title, ylabel, xlabel, color):
        ax.set_title(title, loc="left", color=COLORS["text"], fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, color=COLORS["muted"], fontsize=9)
        ax.set_xlabel(xlabel, color=COLORS["muted"], fontsize=9)
        ax.grid(axis="y", color="#E5EAF1", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.tick_params(colors=COLORS["muted"], labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#C9D3DF")
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.margins(x=0.025, y=0.16)
        ax.set_facecolor(COLORS["surface"])