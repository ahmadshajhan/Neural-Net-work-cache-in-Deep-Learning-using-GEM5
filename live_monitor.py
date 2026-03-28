#!/usr/bin/env python3
"""
Stylized PyQt5 dashboard for the gem5 cache-analysis pipeline.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from model_utils import CLASSES, FEATURES, load_model_metrics, predict_image, workload_macs
from stats_utils import DEFAULT_RESULTS, ensure_matplotlib_env, load_results, result_complete

ensure_matplotlib_env()

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


STATUS_PATH = Path("results/live_status.json")
RESULT_DIRS = {
    "Direct-Mapped\n(1-way)": Path("results/direct"),
    "4-way\nSet-Assoc": Path("results/set4way"),
    "Fully\nAssociative": Path("results/fullassoc"),
}
COLORS = ["#DA6A57", "#2A7F9E", "#5C9E6F"]
CARD_STYLES = ["#E9F5DB", "#FFF4D6", "#DCEBFF", "#FDE2E4"]


class DropPreviewLabel(QLabel):
    def __init__(self, parent: "LiveWindow") -> None:
        super().__init__("Drop image here or click Browse Image")
        self.parent_window = parent
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        urls = event.mimeData().urls()
        if not urls:
            return
        local_path = urls[0].toLocalFile()
        if local_path:
            self.parent_window._predict_and_render(Path(local_path))
            event.acceptProposedAction()


class MetricCard(QFrame):
    def __init__(self, title: str, accent: str) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        self.setStyleSheet(
            f"""
            QFrame#metricCard {{
                background: {accent};
                border-radius: 18px;
                border: 1px solid rgba(32, 46, 58, 0.08);
            }}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)

        self.title = QLabel(title)
        self.title.setStyleSheet("color: #4A5568; font-size: 12px; font-weight: 700; letter-spacing: 0.6px;")
        self.value = QLabel("--")
        self.value.setStyleSheet("color: #152238; font-size: 28px; font-weight: 800;")
        self.detail = QLabel("Waiting for data")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color: #455468; font-size: 12px;")

        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.detail)

    def update_text(self, value: str, detail: str) -> None:
        self.value.setText(value)
        self.detail.setText(detail)


class LiveWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("gem5 Food Mapping Dashboard")
        self.resize(1220, 860)
        self.setMinimumSize(960, 680)
        self._apply_style()
        self.last_prediction: dict[str, object] | None = None
        self.current_preview_pixmap: QPixmap | None = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        self.setCentralWidget(scroll)

        root = QWidget()
        scroll.setWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(20, 18, 20, 18)
        main.setSpacing(16)

        hero = QFrame()
        hero.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #F8EDE3, stop:0.5 #F6FFF8, stop:1 #E8F1FF);
                border-radius: 24px;
                border: 1px solid rgba(32, 46, 58, 0.08);
            }
            """
        )
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(24, 20, 24, 20)
        hero_layout.setSpacing(8)

        title = QLabel("gem5 Neural Cache Monitor")
        title.setStyleSheet("font-size: 30px; font-weight: 900; color: #102A43;")
        subtitle = QLabel("Watch all cache mappings, model accuracy, and live gem5 logs in one place.")
        subtitle.setStyleSheet("font-size: 14px; color: #486581;")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-size: 14px; color: #243B53; font-weight: 600;")
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        hero_layout.addWidget(self.summary)
        main.addWidget(hero)

        card_grid = QGridLayout()
        card_grid.setHorizontalSpacing(14)
        card_grid.setVerticalSpacing(14)
        self.train_card = MetricCard("TRAIN ACCURACY", CARD_STYLES[0])
        self.test_card = MetricCard("TEST ACCURACY", CARD_STYLES[1])
        self.batch_card = MetricCard("GEM5 BATCH", CARD_STYLES[2])
        self.best_card = MetricCard("BEST MAPPING", CARD_STYLES[3])
        cards = [self.train_card, self.test_card, self.batch_card, self.best_card]
        for index, card in enumerate(cards):
            card_grid.addWidget(card, index // 2, index % 2)
        main.addLayout(card_grid)

        middle = QSplitter(Qt.Horizontal)
        middle.setChildrenCollapsible(False)
        middle.setHandleWidth(10)
        main.addWidget(middle, stretch=1)

        chart_frame = QFrame()
        chart_frame.setStyleSheet("QFrame { background: #FFFFFF; border-radius: 24px; border: 1px solid rgba(32, 46, 58, 0.08); }")
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(18, 18, 18, 18)
        chart_title = QLabel("Performance Overview")
        chart_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #102A43;")
        self.figure = Figure(figsize=(10, 7))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chart_layout.addWidget(chart_title)
        chart_layout.addWidget(self.canvas)
        middle.addWidget(chart_frame)

        side_container = QSplitter(Qt.Vertical)
        side_container.setChildrenCollapsible(False)
        side_container.setHandleWidth(10)
        middle.addWidget(side_container)
        middle.setStretchFactor(0, 3)
        middle.setStretchFactor(1, 2)

        predictor_frame = QFrame()
        predictor_frame.setStyleSheet("QFrame { background: #FFFFFF; border-radius: 24px; border: 1px solid rgba(32, 46, 58, 0.08); }")
        predictor_layout = QVBoxLayout(predictor_frame)
        predictor_layout.setContentsMargins(18, 18, 18, 18)
        predictor_title = QLabel("Image Prediction")
        predictor_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #102A43;")
        predictor_subtitle = QLabel("Upload any food image to run the current model and see timing, confidence, and raw probabilities.")
        predictor_subtitle.setWordWrap(True)
        predictor_subtitle.setStyleSheet("font-size: 12px; color: #52667A;")
        self.image_path_label = QLabel("Image: sample test image")
        self.image_path_label.setWordWrap(True)
        self.image_path_label.setStyleSheet("font-size: 12px; color: #334E68; font-weight: 700;")
        self.preview = DropPreviewLabel(self)
        self.preview.setMinimumHeight(180)
        self.preview.setStyleSheet("background: #F8FAFC; border-radius: 18px; color: #7B8794; font-weight: 700; border: 2px dashed #B8C4D4;")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.prediction_headline = QLabel("Prediction: --")
        self.prediction_headline.setStyleSheet("font-size: 18px; font-weight: 900; color: #102A43;")
        self.prediction_meta = QLabel("Upload an image to begin")
        self.prediction_meta.setWordWrap(True)
        self.prediction_meta.setStyleSheet("font-size: 12px; color: #52667A;")
        self.prediction_truth = QLabel("Ground truth: unknown")
        self.prediction_truth.setWordWrap(True)
        self.prediction_truth.setStyleSheet("font-size: 12px; color: #7C2D12; font-weight: 700;")
        self.prediction_scores = QTextEdit()
        self.prediction_scores.setReadOnly(True)
        self.prediction_scores.setMaximumHeight(120)
        self.prediction_scores.setStyleSheet("background: #F8FAFC; border-radius: 14px; padding: 8px;")
        self.mapping_projection = QTextEdit()
        self.mapping_projection.setReadOnly(True)
        self.mapping_projection.setMaximumHeight(110)
        self.mapping_projection.setStyleSheet("background: #F8FAFC; border-radius: 14px; padding: 8px;")
        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.upload_button = QPushButton("Browse Image")
        self.upload_button.clicked.connect(self.open_image)
        self.sample_button = QPushButton("Use Sample")
        self.sample_button.clicked.connect(self._load_default_example)
        button_row.addWidget(self.upload_button)
        button_row.addWidget(self.sample_button)
        predictor_layout.addWidget(predictor_title)
        predictor_layout.addWidget(predictor_subtitle)
        predictor_layout.addWidget(self.image_path_label)
        predictor_layout.addWidget(self.preview)
        predictor_layout.addWidget(self.prediction_headline)
        predictor_layout.addWidget(self.prediction_meta)
        predictor_layout.addWidget(self.prediction_truth)
        predictor_layout.addWidget(self.prediction_scores)
        predictor_layout.addWidget(self.mapping_projection)
        predictor_layout.addLayout(button_row)
        side_container.addWidget(predictor_frame)

        table_frame = QFrame()
        table_frame.setStyleSheet("QFrame { background: #FFFFFF; border-radius: 24px; border: 1px solid rgba(32, 46, 58, 0.08); }")
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(18, 18, 18, 18)
        table_title = QLabel("Mapping Results")
        table_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #102A43;")
        self.table = QTableWidget(3, 5)
        self.table.setHorizontalHeaderLabels(["Mapping", "Miss %", "Ticks (M)", "Speedup", "Throughput"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setStyleSheet("font-weight: 700;")
        table_layout.addWidget(table_title)
        table_layout.addWidget(self.table)
        side_container.addWidget(table_frame)

        logs_frame = QFrame()
        logs_frame.setStyleSheet("QFrame { background: #FFFFFF; border-radius: 24px; border: 1px solid rgba(32, 46, 58, 0.08); }")
        logs_layout = QGridLayout(logs_frame)
        logs_layout.setContentsMargins(18, 18, 18, 18)
        logs_layout.setSpacing(10)
        self.logs: dict[str, QTextEdit] = {}
        for index, (label, result_dir) in enumerate(RESULT_DIRS.items()):
            title_label = QLabel(label.replace("\n", " "))
            title_label.setStyleSheet("font-weight: 800; color: #102A43;")
            log_box = QTextEdit()
            log_box.setReadOnly(True)
            log_box.setLineWrapMode(QTextEdit.NoWrap)
            log_box.setStyleSheet("background: #F8FAFC; border-radius: 14px; padding: 8px;")
            logs_layout.addWidget(title_label, 0, index)
            logs_layout.addWidget(log_box, 1, index)
            self.logs[str(result_dir)] = log_box
        side_container.addWidget(logs_frame)
        side_container.setStretchFactor(0, 3)
        side_container.setStretchFactor(1, 2)
        side_container.setStretchFactor(2, 3)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)
        self._load_default_example()
        self.refresh()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #F7F4EA;
                color: #102A43;
                font-family: DejaVu Sans;
            }
            QLabel {
                background: transparent;
            }
            QTableWidget {
                background: #F8FAFC;
                border-radius: 16px;
                gridline-color: rgba(50, 65, 75, 0.08);
                alternate-background-color: #EEF6FF;
            }
            QPushButton {
                background: #0F766E;
                color: white;
                border: none;
                border-radius: 14px;
                padding: 10px 16px;
                font-weight: 800;
            }
            QPushButton:hover {
                background: #115E59;
            }
            QTextEdit {
                font-family: DejaVu Sans Mono;
                font-size: 11px;
            }
            """
        )

    def _load_status(self) -> dict[str, object]:
        if not STATUS_PATH.exists():
            return {
                "state": "starting",
                "step": "Preparing pipeline",
                "started_at": time.time(),
                "updated_at": time.time(),
            }
        try:
            return json.loads(STATUS_PATH.read_text())
        except json.JSONDecodeError:
            return {
                "state": "updating",
                "step": "Refreshing status",
                "started_at": time.time(),
                "updated_at": time.time(),
            }

    def _tail_log(self, result_dir: Path) -> str:
        log_path = result_dir / "gem5.log"
        if not log_path.exists():
            return "Waiting for log output..."
        lines = log_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-18:]) if lines else "Waiting for log output..."

    def _load_default_example(self) -> None:
        sample_candidates = sorted(Path("data/pizza_steak_sushi/test").glob("*/*.jpg"))
        if sample_candidates:
            self._predict_and_render(sample_candidates[0])

    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an image",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if file_path:
            self._predict_and_render(Path(file_path))

    def _predict_and_render(self, image_path: Path) -> None:
        try:
            self.last_prediction = predict_image(image_path)
        except Exception as exc:
            self.prediction_headline.setText("Prediction: failed")
            self.prediction_meta.setText(str(exc))
            self.prediction_scores.setPlainText("Could not run inference for this file.")
            return

        pixmap = QPixmap(str(image_path))
        if not pixmap.isNull():
            self.current_preview_pixmap = pixmap
            self._refresh_preview_pixmap()
        else:
            self.current_preview_pixmap = None
            self.preview.setText("Preview unavailable")
        self.image_path_label.setText(f"Image: {image_path}")

        confidence = float(self.last_prediction["confidence"]) * 100.0
        elapsed_ms = float(self.last_prediction["elapsed_ms"])
        self.prediction_headline.setText(f"Prediction: {self.last_prediction['predicted_class']}")
        self.prediction_meta.setText(
            f"Confidence {confidence:.2f}% | Inference time {elapsed_ms:.3f} ms\n"
            f"Image: {image_path.name}"
        )
        if "actual_class" in self.last_prediction:
            verdict = "correct" if bool(self.last_prediction.get("is_correct")) else "incorrect"
            verdict_color = "#166534" if bool(self.last_prediction.get("is_correct")) else "#B91C1C"
            self.prediction_truth.setStyleSheet(f"font-size: 12px; color: {verdict_color}; font-weight: 800;")
            self.prediction_truth.setText(
                f"Ground truth: {self.last_prediction['actual_class']} | Prediction is {verdict}"
            )
        else:
            self.prediction_truth.setStyleSheet("font-size: 12px; color: #7C2D12; font-weight: 700;")
            self.prediction_truth.setText("Ground truth: unknown for this uploaded file")
        probabilities = self.last_prediction["probabilities"]
        logits = self.last_prediction["logits"]
        lines = [
            f"{class_name:<8} prob={prob * 100:>6.2f}%  logit={logit:>7.4f}"
            for class_name, prob, logit in zip(CLASSES, probabilities, logits)
        ]
        self.prediction_scores.setPlainText("\n".join(lines))
        self.mapping_projection.setPlainText(self._prediction_mapping_lines())

    def _refresh_preview_pixmap(self) -> None:
        if self.current_preview_pixmap is None:
            return
        scaled = self.current_preview_pixmap.scaled(
            max(self.preview.width() - 20, 120),
            max(self.preview.height() - 20, 120),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview.setPixmap(scaled)

    def refresh(self) -> None:
        status = self._load_status()
        model_metrics = load_model_metrics()
        results = load_results(DEFAULT_RESULTS, use_demo_fallback=False)

        started_at = float(status.get("started_at", time.time()))
        elapsed = max(0.0, time.time() - started_at)
        state = str(status.get("state", "running"))
        step = str(status.get("step", "Running"))
        current = str(status.get("current_config", ""))
        details = str(status.get("details", "")).strip()
        summary = f"State: {state}  |  Step: {step}  |  Elapsed: {elapsed:.1f}s"
        if current:
            summary += f"  |  Current mapping: {current}"
        if details:
            summary += f"\n{details}"
        self.summary.setText(summary)

        self._update_cards(model_metrics, results, current or "Waiting")
        self._update_table(results)
        self._draw_charts(results, model_metrics)

        for result_dir_str, widget in self.logs.items():
            widget.setPlainText(self._tail_log(Path(result_dir_str)))

    def _update_cards(
        self,
        model_metrics: dict[str, object] | None,
        results: dict[str, dict[str, float | int | None]],
        current_mapping: str,
    ) -> None:
        if model_metrics:
            self.train_card.update_text(
                f"{float(model_metrics['train_accuracy']) * 100:.2f}%",
                f"{int(model_metrics['train_samples'])} training images",
            )
            self.test_card.update_text(
                f"{float(model_metrics['test_accuracy']) * 100:.2f}%",
                f"{int(model_metrics['test_samples'])} test images",
            )
            self.batch_card.update_text(
                f"{float(model_metrics['gem5_subset_accuracy']) * 100:.2f}%",
                f"{int(model_metrics['gem5_subset_samples'])} images simulated inside gem5",
            )
        else:
            self.train_card.update_text("--", "Model metrics not ready yet")
            self.test_card.update_text("--", "Model metrics not ready yet")
            self.batch_card.update_text("--", "Model metrics not ready yet")

        complete = {
            label: metrics
            for label, metrics in results.items()
            if result_complete(metrics)
        }
        if complete:
            baseline_label = list(complete.keys())[0]
            baseline_ticks = int(complete[baseline_label]["sim_ticks"])
            best_label = min(complete, key=lambda label: int(complete[label]["sim_ticks"]))
            best_ticks = int(complete[best_label]["sim_ticks"])
            speedup = baseline_ticks / best_ticks if best_ticks else 0.0
            throughput = self._format_throughput(model_metrics, best_ticks)
            self.best_card.update_text(
                best_label.replace("\n", " "),
                f"{speedup:.3f}x speedup | {throughput} | active: {current_mapping}",
            )
        else:
            self.best_card.update_text(current_mapping, "Waiting for gem5 stats")

    def _update_table(self, results: dict[str, dict[str, float | int | None]]) -> None:
        labels = list(results.keys())
        complete_ticks = [
            int(results[label]["sim_ticks"])
            for label in labels
            if result_complete(results[label])
        ]
        baseline = complete_ticks[0] if complete_ticks else None
        sample_count = 24
        if (metrics_data := load_model_metrics()) is not None:
            sample_count = int(metrics_data.get("gem5_subset_samples", sample_count))
        total_macs = workload_macs(sample_count)

        for row, label in enumerate(labels):
            metrics = results[label]
            mapping_name = label.replace("\n", " ")
            miss = "--"
            ticks = "--"
            speedup = "--"
            throughput = "--"
            if result_complete(metrics):
                miss = f"{float(metrics['miss_rate']):.2f}"
                tick_value = int(metrics["sim_ticks"]) / 1e6
                ticks = f"{tick_value:.2f}"
                if baseline:
                    speedup = f"{baseline / int(metrics['sim_ticks']):.3f}x"
                sim_seconds = int(metrics["sim_ticks"]) / 1e12
                if sim_seconds > 0:
                    throughput = f"{(total_macs / sim_seconds) / 1e6:.2f} M/s"

            values = [mapping_name, miss, ticks, speedup, throughput]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()

    def _format_throughput(self, model_metrics: dict[str, object] | None, ticks: int) -> str:
        sample_count = 24
        if model_metrics:
            sample_count = int(model_metrics.get("gem5_subset_samples", sample_count))
        sim_seconds = ticks / 1e12
        if sim_seconds <= 0:
            return "--"
        return f"{(workload_macs(sample_count) / sim_seconds) / 1e6:.2f} M MAC/s"

    def _prediction_mapping_lines(self) -> str:
        model_metrics = load_model_metrics()
        sample_count = 24
        if model_metrics:
            sample_count = int(model_metrics.get("gem5_subset_samples", sample_count))
        results = load_results(DEFAULT_RESULTS, use_demo_fallback=False)
        lines = ["Estimated gem5 time per uploaded image:"]
        for label, metrics in results.items():
            if not result_complete(metrics):
                lines.append(f"{label.replace(chr(10), ' ')}: waiting for stats")
                continue
            ticks = int(metrics["sim_ticks"])
            per_image_ms = (ticks / 1e12) / sample_count * 1000.0
            miss_rate = float(metrics["miss_rate"])
            lines.append(
                f"{label.replace(chr(10), ' ')}: {per_image_ms:.3f} ms/image | miss {miss_rate:.2f}%"
            )
        lines.append(f"Matrix workload: {sample_count} images | {FEATURES} features | {workload_macs(sample_count):,} MACs")
        return "\n".join(lines)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_preview_pixmap()

    def _draw_charts(
        self,
        results: dict[str, dict[str, float | int | None]],
        model_metrics: dict[str, object] | None,
    ) -> None:
        self.figure.clear()
        axes = self.figure.subplots(2, 2)
        axes = axes.flatten()
        labels = list(results.keys())

        miss_rates = [float(results[label]["miss_rate"]) if result_complete(results[label]) else 0.0 for label in labels]
        ticks_m = [(int(results[label]["sim_ticks"]) / 1e6) if result_complete(results[label]) else 0.0 for label in labels]
        baseline = next((int(results[label]["sim_ticks"]) for label in labels if result_complete(results[label])), None)
        speedups = [
            (baseline / int(results[label]["sim_ticks"])) if baseline and result_complete(results[label]) else 0.0
            for label in labels
        ]
        acc_labels = ["Train", "Test", "GEM5 Batch"]
        acc_values = [0.0, 0.0, 0.0]
        if model_metrics:
            acc_values = [
                float(model_metrics.get("train_accuracy", 0.0)) * 100.0,
                float(model_metrics.get("test_accuracy", 0.0)) * 100.0,
                float(model_metrics.get("gem5_subset_accuracy", 0.0)) * 100.0,
            ]

        chart_specs = [
            ("L1 Miss Rate (%)", miss_rates, "%", COLORS),
            ("Simulation Ticks (M)", ticks_m, "M", COLORS),
            ("Speedup vs Direct", speedups, "x", COLORS),
            ("Classifier Accuracy (%)", acc_values, "%", ["#D77A61", "#3D7EA6", "#7FB069"]),
        ]

        for ax, (title, values, suffix, colors) in zip(axes, chart_specs):
            x_labels = labels if title != "Classifier Accuracy (%)" else acc_labels
            bars = ax.bar(x_labels, values, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
            ax.set_title(title, fontweight="bold", color="#102A43")
            ax.grid(axis="y", alpha=0.25, linestyle="--")
            ax.tick_params(axis="x", labelsize=9)
            ymax = max(values) if any(values) else 1.0
            ax.set_ylim(0, ymax * 1.25)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(ymax * 0.03, 0.02),
                    f"{value:.2f}{suffix}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#102A43",
                )

        self.figure.tight_layout()
        self.canvas.draw_idle()


def main() -> int:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("No display detected; live monitor requires a GUI session.")
        return 1

    app = QApplication(sys.argv)
    window = LiveWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
