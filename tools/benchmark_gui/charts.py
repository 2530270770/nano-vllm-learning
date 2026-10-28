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
        self.figure = Figure(figsize=(9, 6), dpi=100, layout="constrained")
        self.figure.set_facecolor(COLORS["surface"])
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)

    def show_rows(self, rows):
        self.figure.clear()
        use_chinese = plot_font_supports_cjk()
        if use_chinese:
            metrics = [
                ("output_tokens_per_s", "单请求输出吞吐量", "tokens/s", "#2563EB"),
                ("ttft_s", "首字延迟 TTFT", "秒", "#0E9F8E"),
                ("tpot_s", "单 Token 延迟 TPOT", "秒/token", "#D97706"),
            ]
        else:
            # A missing CJK font must not corrupt Matplotlib or produce glyph
            # warnings. The surrounding Qt UI can still use the system fallback.
            metrics = [
                ("output_tokens_per_s", "Per-request throughput", "tokens/s", "#2563EB"),
                ("ttft_s", "Time to first token", "seconds", "#0E9F8E"),
                ("tpot_s", "Time per output token", "seconds/token", "#D97706"),
            ]
        usable = [
            r for r in rows
            if any(as_float(r.get(k)) is not None for k, _, _, _ in metrics)
        ]

        self.hint.setText(
            f"已加载 {len(usable)} 条请求记录 · 悬停可查看趋势"
            if usable
            else "CSV 中没有可绘制的指标，请检查 request_metrics.csv 是否生成。"
        )
        axes = self.figure.subplots(3, 1)
        x = list(range(1, len(usable) + 1))
        for ax, (key, title, unit, color) in zip(axes, metrics):
            y = [as_float(row.get(key)) for row in usable]
            pairs = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
            if pairs:
                px = [p[0] for p in pairs]
                py = [p[1] for p in pairs]
                ax.plot(
                    px,
                    py,
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
                    0.5,
                    0.5,
                    "暂无数据" if use_chinese else "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color=COLORS["muted"],
                    fontsize=11,
                )
            ax.set_title(title, loc="left", color=COLORS["text"], fontsize=11,
                         fontweight="bold", pad=8)
            ax.set_ylabel(unit, color=COLORS["muted"], fontsize=9)
            ax.set_xlabel(
                "请求序号" if use_chinese else "Request",
                color=COLORS["muted"],
                fontsize=9,
            )
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
            if pairs:
                ax.margins(x=0.025, y=0.16)
            ax.set_facecolor(COLORS["surface"])
        self.canvas.draw()
