import os
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

# 支持 `python tools/benchmark_gui/main.py` 从仓库根目录启动
GUI_DIR = Path(__file__).resolve().parent
REPO_ROOT = GUI_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.benchmark_gui.worker import BenchmarkWorker
from tools.benchmark_gui.results import load_results, find_nested_dict
from tools.benchmark_gui.charts import MetricCharts
from tools.benchmark_gui.theme import apply_theme, choose_ui_font_family


class MetricCard(QFrame):
    """Small KPI card used by the result summary."""

    def __init__(self, title, detail, parent=None):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setMinimumWidth(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 11)
        layout.setSpacing(3)

        title_label = QLabel(title)
        title_label.setObjectName("MetricTitle")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("MetricValue")
        self.value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        detail_label = QLabel(detail)
        detail_label.setObjectName("MetricDetail")

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(detail_label)

    def set_value(self, value):
        self.value_label.setText(str(value))


def format_number(value, digits=2, suffix=""):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{number:,.{digits}f}{suffix}"


def set_widget_state(widget, state):
    widget.setProperty("state", state)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def make_app_icon():
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor("#2563EB"))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(4, 4, 56, 56, 14, 14)
    painter.setPen(QColor("#FFFFFF"))
    font = QFont(choose_ui_font_family())
    font.setPixelSize(27)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "nV")
    painter.end()
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("nano-vLLM Benchmark Console")
        self.resize(1380, 900)
        self.setMinimumSize(1080, 720)
        self.setWindowIcon(make_app_icon())

        self.worker = None
        self.output_dir = REPO_ROOT / "outputs" / "benchmark"
        self.current_run_dir = None       # 本次运行生成的 run_* 目录
        self.current_results_dir = None   # 当前界面上展示的结果目录

        central = QWidget()
        central.setObjectName("AppRoot")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 18)
        root.setSpacing(14)

        # ---- Header ----
        header_card = QFrame()
        header_card.setObjectName("HeaderCard")
        header = QHBoxLayout(header_card)
        header.setContentsMargins(22, 16, 20, 16)
        header.setSpacing(14)

        title_stack = QVBoxLayout()
        title_stack.setSpacing(2)
        title = QLabel("nano-vLLM Benchmark")
        title.setObjectName("AppTitle")
        subtitle = QLabel("推理性能测试 · 实时日志 · 指标可视化")
        subtitle.setObjectName("AppSubtitle")
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header.addLayout(title_stack)
        header.addStretch()

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("StatusPill")
        self.status_label.setProperty("state", "idle")
        header.addWidget(self.status_label)
        root.addWidget(header_card)

        # ---- Splitter ----
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        root.addWidget(splitter, 1)

        # ---- Left panel ----
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(370)
        left_scroll.setMaximumWidth(460)
        left = QWidget()
        left_scroll.setWidget(left)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(12)

        # 实验配置
        config_group = QGroupBox("实验配置")
        config_group.setObjectName("Card")
        form = QGridLayout(config_group)
        form.setContentsMargins(16, 19, 16, 13)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(9)
        form.setColumnStretch(1, 1)

        self.model_edit = QLineEdit("")
        self.model_edit.setPlaceholderText("模型目录或 Hugging Face 模型名（必填）")
        form.addWidget(self.field_label("模型路径"), 0, 0, 1, 2)
        form.addWidget(self.model_edit, 1, 0, 1, 2)

        self.num_requests = self.spin(4, 1, 512)
        self.prompt_tokens = self.spin(128, 1, 4096)
        self.max_tokens = self.spin(32, 1, 4096)
        self.max_num_seqs = self.spin(4, 1, 512)
        self.max_batched_tokens = self.spin(16384, 1, 1048576)
        self.warmup_runs = self.spin(1, 0, 100)
        self.repeats = self.spin(5, 1, 100)

        form.addWidget(self.field_label("请求数量"), 2, 0)
        form.addWidget(self.field_label("Prompt Tokens"), 2, 1)
        form.addWidget(self.num_requests, 3, 0)
        form.addWidget(self.prompt_tokens, 3, 1)
        form.addWidget(self.field_label("输出 Tokens"), 4, 0)
        form.addWidget(self.field_label("并发序列数"), 4, 1)
        form.addWidget(self.max_tokens, 5, 0)
        form.addWidget(self.max_num_seqs, 5, 1)

        self.temperature = QLineEdit("0.6")
        self.temperature.setPlaceholderText("例如 0.6")
        form.addWidget(self.field_label("批处理 Token 上限"), 6, 0, 1, 2)
        form.addWidget(self.max_batched_tokens, 7, 0, 1, 2)
        form.addWidget(self.field_label("预热轮数"), 8, 0)
        form.addWidget(self.field_label("重复轮数"), 8, 1)
        form.addWidget(self.warmup_runs, 9, 0)
        form.addWidget(self.repeats, 9, 1)

        form.addWidget(self.field_label("Temperature"), 10, 0, 1, 2)
        form.addWidget(self.temperature, 11, 0, 1, 2)

        self.output_edit = QLineEdit(str(self.output_dir))
        self.output_edit.setPlaceholderText("Benchmark 结果父目录")
        output_row = QHBoxLayout()
        output_row.setSpacing(7)
        output_row.addWidget(self.output_edit, 1)
        browse_button = QPushButton("浏览")
        browse_button.clicked.connect(self.choose_output_dir)
        output_row.addWidget(browse_button)
        form.addWidget(self.field_label("结果目录"), 12, 0, 1, 2)
        form.addLayout(output_row, 13, 0, 1, 2)

        left_layout.addWidget(config_group)

        # 运行控制
        action_group = QGroupBox("运行控制")
        action_group.setObjectName("Card")
        actions = QVBoxLayout(action_group)
        actions.setContentsMargins(14, 19, 14, 12)
        actions.setSpacing(8)
        buttons = QGridLayout()
        buttons.setHorizontalSpacing(8)
        buttons.setVerticalSpacing(8)

        self.start_button = QPushButton("开始 Benchmark")
        self.start_button.setProperty("role", "primary")
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self.start_benchmark)

        self.stop_button = QPushButton("停止")
        self.stop_button.setProperty("role", "danger")
        self.stop_button.clicked.connect(self.stop_benchmark)
        self.stop_button.setEnabled(False)

        self.load_button = QPushButton("刷新最新结果")
        self.load_button.clicked.connect(self.on_refresh_clicked)

        self.open_dir_button = QPushButton("打开当前结果目录")
        self.open_dir_button.clicked.connect(self.open_output_dir)

        buttons.addWidget(self.start_button, 0, 0, 1, 2)
        buttons.addWidget(self.stop_button, 1, 0)
        buttons.addWidget(self.load_button, 1, 1)
        buttons.addWidget(self.open_dir_button, 2, 0, 1, 2)
        actions.addLayout(buttons)
        left_layout.addWidget(action_group)

        # 结果详情
        self.metrics_label = QLabel("尚未加载结果")
        self.metrics_label.setObjectName("ResultText")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        summary_frame = QFrame()
        summary_frame.setObjectName("ResultSummary")
        summary_layout = QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(3, 3, 3, 3)
        summary_layout.addWidget(self.metrics_label)

        summary_group = QGroupBox("结果详情")
        summary_group.setObjectName("Card")
        summary_group_layout = QVBoxLayout(summary_group)
        summary_group_layout.setContentsMargins(12, 18, 12, 12)
        summary_group_layout.addWidget(summary_frame)
        left_layout.addWidget(summary_group)
        left_layout.addStretch()

        # ---- Right panel ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        metric_row = QHBoxLayout()
        metric_row.setSpacing(10)
        self.requests_card = MetricCard("请求记录", "已加载结果")
        self.batch_card = MetricCard("Batch 吞吐量", "mean · tokens/s")
        self.request_card = MetricCard("单请求吞吐量", "mean · tokens/s")
        self.memory_card = MetricCard("显存 Reserved", "峰值 Reserved / 总显存")
        for card in (
            self.requests_card,
            self.batch_card,
            self.request_card,
            self.memory_card,
        ):
            metric_row.addWidget(card)
        right_layout.addLayout(metric_row)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        right_layout.addWidget(tabs, 1)

        self.tabs_charts = MetricCharts()
        tabs.addTab(self.tabs_charts, "性能图表")

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText("运行 Benchmark 后，子进程日志会实时显示在这里。")
        log_wrapper = QWidget()
        log_layout = QVBoxLayout(log_wrapper)
        log_layout.setContentsMargins(12, 12, 12, 12)
        log_layout.addWidget(self.log_view)
        tabs.addTab(log_wrapper, "运行日志")

        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 940])

        self.refresh_results()

    # ---- helpers ----
    @staticmethod
    def field_label(text):
        label = QLabel(text)
        label.setObjectName("MutedLabel")
        return label

    @staticmethod
    def spin(value, minimum, maximum):
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return widget

    @staticmethod
    def _has_results(path):
        return path.is_dir() and (
            (path / "request_metrics.csv").is_file()
            or (path / "summary.json").is_file()
        )

    def resolve_results_dir(self, preferred=None):
        """
        决定本次要展示的结果目录：
        1) 显式传入的目录（本次 run_dir）；
        2) 父目录自身有 CSV/JSON；
        3) 父目录下最新的 run_* 子目录；
        4) 兜底返回父目录。
        """
        if preferred:
            p = Path(preferred).expanduser()
            if p.is_dir():
                return p

        base = Path(self.output_edit.text()).expanduser()
        if self._has_results(base):
            return base

        if base.is_dir():
            run_dirs = sorted(
                [
                    d for d in base.iterdir()
                    if d.is_dir() and d.name.startswith("run_")
                ],
                key=lambda d: d.name,
                reverse=True,
            )
            if run_dirs:
                return run_dirs[0]
        return base

    def set_status(self, text, state="idle"):
        self.status_label.setText(text)
        set_widget_state(self.status_label, state)

    def choose_output_dir(self):
        selected = QFileDialog.getExistingDirectory(
            self, "选择结果目录", self.output_edit.text() or str(self.output_dir)
        )
        if selected:
            self.output_edit.setText(selected)

    # ---- command ----
    def command(self):
        script = REPO_ROOT / "benchmarks" / "benchmark.py"
        if not script.is_file():
            raise FileNotFoundError(f"没有找到 Benchmark 脚本：{script}")

        model = self.model_edit.text().strip()
        if not model:
            raise ValueError("模型路径不能为空。")

        try:
            temperature = float(self.temperature.text())
        except ValueError:
            raise ValueError("Temperature 必须是数字，例如 0.6")
        if temperature <= 0:
            raise ValueError("Temperature 必须大于 0。")

        prompt_tokens = self.prompt_tokens.value()
        max_tokens = self.max_tokens.value()
        if prompt_tokens + max_tokens > 4096:
            raise ValueError("Prompt Tokens + 输出 Tokens 不能超过 4096。")

        output_dir = Path(self.output_edit.text()).expanduser()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            probe = output_dir / ".gui_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            raise ValueError(f"结果目录不可写：{output_dir}\n{exc}") from exc

        # 每次都落到带时间戳的子目录，避免覆盖历史结果
        run_dir = output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable, "-u", str(script),
            "--model", model,
            "--num-requests", str(self.num_requests.value()),
            "--prompt-tokens", str(prompt_tokens),
            "--max-tokens", str(max_tokens),
            "--max-num-seqs", str(self.max_num_seqs.value()),
            "--max-num-batched-tokens", str(self.max_batched_tokens.value()),
            "--warmup-runs", str(self.warmup_runs.value()),
            "--repeats", str(self.repeats.value()),
            "--temperature", str(temperature),
            "--output-dir", str(run_dir),
        ]
        return command, run_dir

    # ---- actions ----
    def start_benchmark(self):
        if self.worker is not None and self.worker.isRunning():
            return
        try:
            command, run_dir = self.command()
        except Exception as exc:
            QMessageBox.critical(self, "无法启动", str(exc))
            return

        self.current_run_dir = run_dir
        self.log_view.appendPlainText("\n" + "=" * 70)
        self.log_view.appendPlainText("开始新的 Benchmark")
        self.log_view.appendPlainText(f"[GUI] 本次结果目录：{run_dir}")
        self.set_status("● 运行中", "running")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.load_button.setEnabled(False)

        self.worker = BenchmarkWorker(command, str(REPO_ROOT), self)
        self.worker.log_line.connect(self.log_view.appendPlainText)
        self.worker.finished_with_code.connect(self.on_benchmark_finished)
        self.worker.start()

    def stop_benchmark(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.set_status("正在停止…", "running")
            self.log_view.appendPlainText("[GUI] 已请求停止 Benchmark 子进程。")

    def on_benchmark_finished(self, code):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.load_button.setEnabled(True)
        self.set_status(
            "● 测试完成" if code == 0 else f"● 异常退出 ({code})",
            "success" if code == 0 else "error",
        )
        self.log_view.appendPlainText(f"[GUI] Benchmark 子进程退出，code={code}")
        # 关键修复：刷新本次运行目录，而不是父目录
        self.refresh_results(self.current_run_dir)

    def on_refresh_clicked(self):
        # 手动刷新：优先本次运行目录，其次父目录下最新的 run_*
        self.refresh_results(self.current_run_dir)

    # ---- results ----
    def refresh_results(self, target_dir=None):
        results_dir = self.resolve_results_dir(target_dir)
        self.current_results_dir = results_dir

        try:
            rows, summary, csv_path, json_path = load_results(str(results_dir))
        except Exception as exc:
            # 损坏 JSON / 旧版本结果 / 读写失败都在这里兜底
            self.requests_card.set_value("--")
            self.batch_card.set_value("--")
            self.request_card.set_value("--")
            self.memory_card.set_value("--")
            self.metrics_label.setText(
                f"加载结果失败：\n{exc}\n\n结果目录：{results_dir}"
            )
            return

        self.tabs_charts.show_results(rows, summary)

        if not rows and not summary:
            self.requests_card.set_value("0")
            self.batch_card.set_value("--")
            self.request_card.set_value("--")
            self.memory_card.set_value("--")
            self.metrics_label.setText(
                "尚未发现可用的测试结果。\n\n"
                f"结果目录 {results_dir}\n"
                f"CSV {csv_path}\n"
                f"JSON {json_path}"
            )
            return

        batch = find_nested_dict(summary, "aggregate_output_tokens_per_s")
        request = find_nested_dict(summary, "output_tokens_per_s")
        memory = summary.get("memory", {}) if isinstance(summary, dict) else {}

        self.requests_card.set_value(f"{len(rows):,}")
        self.batch_card.set_value(
            format_number(batch.get("mean") if batch else None, 2)
        )
        self.request_card.set_value(
            format_number(request.get("mean") if request else None, 2)
        )

        used_gib = None
        total_gib = None
        if isinstance(memory, dict):
            used_gib = memory.get("peak_torch_reserved_gib")
            baseline = memory.get("baseline_after_warmup")
            if isinstance(baseline, dict):
                total_gib = baseline.get("gpu_total_gib")

        if used_gib is None and total_gib is None:
            self.memory_card.set_value("--")
        elif used_gib is None:
            self.memory_card.set_value(f"-- / {float(total_gib):.1f} GiB")
        elif total_gib is None:
            self.memory_card.set_value(f"{float(used_gib):.1f} / -- GiB")
        else:
            self.memory_card.set_value(
                f"{float(used_gib):.1f} / {float(total_gib):.1f} GiB"
            )

        lines = [
            f"结果目录 {results_dir}",
            f"请求记录 {len(rows)}",
            f"CSV {csv_path}",
            f"JSON {json_path}",
        ]
        if batch:
            lines.append(
                "\nBatch 吞吐量 "
                f"mean={batch.get('mean', 'N/A')} | "
                f"p50={batch.get('p50', 'N/A')} | "
                f"p95={batch.get('p95', 'N/A')} tokens/s"
            )
        if request:
            lines.append(
                "单请求吞吐量 "
                f"mean={request.get('mean', 'N/A')} | "
                f"p50={request.get('p50', 'N/A')} | "
                f"p95={request.get('p95', 'N/A')} tokens/s"
            )
        if isinstance(memory, dict):
            if memory.get("peak_torch_allocated_gib") is not None:
                lines.append(
                    "\n峰值 Torch allocated "
                    f"{memory.get('peak_torch_allocated_gib')} GiB"
                )
            if memory.get("peak_torch_reserved_gib") is not None:
                lines.append(
                    "峰值 Torch reserved "
                    f"{memory.get('peak_torch_reserved_gib')} GiB"
                )
        self.metrics_label.setText("\n".join(lines))

    def open_output_dir(self):
        path = self.current_results_dir or Path(self.output_edit.text()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", str(exc))

    # ---- lifecycle ----
    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("nano-vLLM Benchmark Console")
    apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()