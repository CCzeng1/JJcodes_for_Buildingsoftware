# src/gui/main_window.py
import sys
import os
import multiprocessing
from datetime import datetime
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QStatusBar, QMenuBar, QMenu,
    QMessageBox, QFileDialog, QApplication, QProgressBar,
    QLabel, QPushButton, QComboBox, QGroupBox,
    QLineEdit, QSpinBox, QCheckBox, QFrame
)
from PyQt6.QtGui import QAction, QKeySequence, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from ..core import JosephsonJunctionSolver, PathManager, CalculationType
from ..core.plotter import JJPlotter

from .panels.parameter_panel import ParameterPanel
from .panels.disorder_panel import DisorderPanel
from .panels.plot_panel import PlotPanel

# src/gui/main_window.py（进度和实时参数显示修复）

class CalculationThread(QThread):
    progress_updated = pyqtSignal(int, int, str, object)  # 添加 current_params
    calculation_finished = pyqtSignal(bool, str, object)
    
    def __init__(self, solver, calc_type):
        super().__init__()
        self.solver = solver
        self.calc_type = calc_type
        self._is_running = True
        
    def run(self):
        try:
            # 设置进度回调
            self.solver.progress_callback = self._on_progress
            
            filepath, metadata, data = self.solver.run_calculation(self.calc_type)
            
            if self._is_running:
                self.calculation_finished.emit(True, f"Completed: {filepath}", (metadata, data))
            else:
                self.calculation_finished.emit(False, "Stopped by user", None)
        except Exception as e:
            self.calculation_finished.emit(False, f"Error: {str(e)}", None)
    
    def _on_progress(self, current, total, message, current_params):
        """接收 solver 的进度报告"""
        if self._is_running:
            self.progress_updated.emit(current, total, message, current_params)
    
    def stop(self):
        self._is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("JJ Simulator")
        self.setGeometry(100, 100, 1600, 900)
        
        self.current_solver: Optional[JosephsonJunctionSolver] = None
        self.calculation_thread: Optional[CalculationThread] = None
        self.is_parameters_locked = False
        
        self._create_menu_bar()
        self._create_status_bar()
        self._create_central_widget()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_status)
        self.timer.start(1000)
        
        self._update_title()
        
    def _create_menu_bar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Task", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._on_new_task)
        file_menu.addAction(new_action)
        
        load_action = QAction("&Load Parameters...", self)
        load_action.triggered.connect(self._on_load_parameters)
        file_menu.addAction(load_action)
        
        save_action = QAction("&Save Parameters...", self)
        save_action.triggered.connect(self._on_save_parameters)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu("&View")
        
        preview_action = QAction("Preview &Disorder", self)
        preview_action.triggered.connect(self._on_preview_disorder)
        view_menu.addAction(preview_action)
        
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
        
    def _create_central_widget(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # ===== 左侧：参数面板（固定宽度400）=====
        left_widget = QWidget()
        left_widget.setFixedWidth(400)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        
        # 模式选择
        mode_group = QGroupBox("Calculation Mode")
        mode_layout = QHBoxLayout(mode_group)
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["DC_IV", "DC_IV_Bsweep", "CPR", "CPR_Bsweep", "ABS", "SPECTRA"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        left_layout.addWidget(mode_group)
        
        # 参数面板
        self.param_panel = ParameterPanel()
        left_layout.addWidget(self.param_panel, stretch=4)
        
        # Disorder 面板
        self.disorder_panel = DisorderPanel()
        self.disorder_panel.preview_requested.connect(self._on_preview_disorder)
        left_layout.addWidget(self.disorder_panel, stretch=2)
        
        # Run Control（底部：Lock, Run, Stop）
        control_group = QGroupBox("Run Control")
        control_layout = QHBoxLayout(control_group)
        
        self.lock_btn = QPushButton("🔓 Lock")
        self.lock_btn.setCheckable(True)
        self.lock_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        self.lock_btn.clicked.connect(self._on_toggle_lock)
        control_layout.addWidget(self.lock_btn)
        
        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self._on_run_calculation)
        control_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_calculation)
        control_layout.addWidget(self.stop_btn)
        
        left_layout.addWidget(control_group)
        
        # ===== 右侧：上方 Remote/Parallel + 下方绘图（含底部进度）=====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        
        # 上方：Remote + Parallel
        remote_parallel_widget = self._create_remote_parallel_widget()
        right_layout.addWidget(remote_parallel_widget, stretch=0)
        
        # 中间：绘图面板
        self.plot_panel = PlotPanel()
        right_layout.addWidget(self.plot_panel, stretch=1)
        
        # 底部：进度显示（放在绘图区下方）
        progress_widget = self._create_progress_widget()
        right_layout.addWidget(progress_widget, stretch=0)
        
        main_layout.addWidget(left_widget, stretch=0)
        main_layout.addWidget(right_widget, stretch=1)
        
    def _create_remote_parallel_widget(self):
        group = QGroupBox("Remote & Parallel Settings")
        layout = QHBoxLayout(group)
        layout.setSpacing(15)
        
        # Remote
        remote_box = QWidget()
        remote_layout = QHBoxLayout(remote_box)
        remote_layout.setContentsMargins(0, 0, 0, 0)
        
        self.remote_check = QCheckBox("Use Remote")
        self.remote_check.stateChanged.connect(self._on_remote_toggled)
        remote_layout.addWidget(self.remote_check)
        
        remote_layout.addWidget(QLabel("Host:"))
        self.remote_host = QLineEdit()
        self.remote_host.setPlaceholderText("user@server.edu")
        self.remote_host.setMaximumWidth(150)
        self.remote_host.setEnabled(False)
        remote_layout.addWidget(self.remote_host)
        
        remote_layout.addWidget(QLabel("Port:"))
        self.remote_port = QSpinBox()
        self.remote_port.setRange(1, 65535)
        self.remote_port.setValue(22)
        self.remote_port.setMaximumWidth(60)
        self.remote_port.setEnabled(False)
        remote_layout.addWidget(self.remote_port)
        
        remote_layout.addWidget(QLabel("Key:"))
        self.key_file = QLineEdit()
        self.key_file.setPlaceholderText("~/.ssh/id_rsa")
        self.key_file.setMaximumWidth(120)
        self.key_file.setEnabled(False)
        remote_layout.addWidget(self.key_file)
        
        self.sync_btn = QPushButton("Sync")
        self.sync_btn.setEnabled(False)
        self.sync_btn.clicked.connect(self._on_sync_code)
        remote_layout.addWidget(self.sync_btn)
        
        self.test_btn = QPushButton("Test")
        self.test_btn.setEnabled(False)
        self.test_btn.clicked.connect(self._on_test_connection)
        remote_layout.addWidget(self.test_btn)
        
        layout.addWidget(remote_box)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Parallel
        parallel_box = QWidget()
        parallel_layout = QHBoxLayout(parallel_box)
        parallel_layout.setContentsMargins(0, 0, 0, 0)
        
        cpu_count = multiprocessing.cpu_count()
        
        parallel_layout.addWidget(QLabel("Parallel:"))
        
        parallel_layout.addWidget(QLabel("Outer:"))
        self.outer_parallel = QSpinBox()
        self.outer_parallel.setRange(1, cpu_count)
        self.outer_parallel.setValue(min(4, cpu_count))
        self.outer_parallel.setMaximumWidth(50)
        parallel_layout.addWidget(self.outer_parallel)
        
        parallel_layout.addWidget(QLabel("Inner:"))
        self.inner_parallel = QSpinBox()
        self.inner_parallel.setRange(1, cpu_count)
        self.inner_parallel.setValue(min(4, cpu_count))
        self.inner_parallel.setMaximumWidth(50)
        parallel_layout.addWidget(self.inner_parallel)
        
        parallel_layout.addWidget(QLabel(f"(max: {cpu_count})"))
        
        layout.addWidget(parallel_box)
        layout.addStretch()
        
        return group
        
    def _create_progress_widget(self):
        """创建底部进度显示"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.progress_label = QLabel("Ready")
        layout.addWidget(self.progress_label)
        
        layout.addStretch()
        
        self.main_progress = QProgressBar()
        self.main_progress.setRange(0, 100)
        self.main_progress.setValue(0)
        self.main_progress.setMaximumWidth(300)
        layout.addWidget(self.main_progress)
        
        self.elapsed_label = QLabel("00:00:00")
        layout.addWidget(self.elapsed_label)
        
        return widget
        
    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_msg = QLabel("Ready")
        self.status_bar.addWidget(self.status_msg)
        
    def _update_title(self):
        mode = self.mode_combo.currentText()
        title = f"JJ Simulator - {mode}"
        if self.is_parameters_locked:
            title += " [LOCKED]"
        self.setWindowTitle(title)
        
    def _on_mode_changed(self, mode: str):
        self._update_title()
        
    def _on_remote_toggled(self, state):
        enabled = state == Qt.CheckState.Checked.value
        self.remote_host.setEnabled(enabled)
        self.remote_port.setEnabled(enabled)
        self.key_file.setEnabled(enabled)
        self.sync_btn.setEnabled(enabled)
        self.test_btn.setEnabled(enabled)
        
    def _on_sync_code(self):
        QMessageBox.information(self, "Sync", "Code sync: Upload local code to remote server")
        
    def _on_test_connection(self):
        QMessageBox.information(self, "Test", "Testing SSH connection...")
        
    def _on_run_calculation(self):
        if not self.is_parameters_locked:
            QMessageBox.warning(self, "Not Locked", "Please lock parameters first!")
            return
            
        # 强制清零绘图面板
        self.plot_panel.clear()
        self.plot_panel.info_label.setText("Starting calculation...")
        
        params = self._gather_all_parameters()
        calc_type = self.mode_combo.currentText()
        
        path_manager = PathManager(
            base_dir=params.get('output_dir', 'results'),
            task_name=params.get('task_name', None)
        )
        self.current_solver = JosephsonJunctionSolver(params, path_manager)
        
        self.calculation_thread = CalculationThread(self.current_solver, calc_type)
        self.calculation_thread.progress_updated.connect(self._on_progress_updated)
        self.calculation_thread.calculation_finished.connect(self._on_calculation_finished)
        self.calculation_thread.start()
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_label.setText(f"Running {calc_type}...")
        self._update_title()
        
    def _on_stop_calculation(self):
        """强力停止计算"""
        # 1. 停止线程
        if self.calculation_thread:
            self.calculation_thread.stop()
            self.calculation_thread.terminate()  # 强制终止
            self.calculation_thread.wait(1000)  # 等待1秒
            self.calculation_thread = None
            
        # 2. 清零绘图面板
        self.plot_panel.clear()
        self.plot_panel.info_label.setText("Stopped by user")
        
        # 3. 重置 UI
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_label.setText("Stopped")
        self.main_progress.setValue(0)
        self.status_msg.setText("Ready")
        
    def _on_toggle_lock(self):
        self.is_parameters_locked = not self.is_parameters_locked
        self.param_panel.set_locked(self.is_parameters_locked)
        self.disorder_panel.set_locked(self.is_parameters_locked)
        
        if self.is_parameters_locked:
            self.lock_btn.setText("🔒 Locked")
            self.lock_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        else:
            self.lock_btn.setText("🔓 Lock")
            self.lock_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        self._update_title()
        
    def _on_reset_parameters(self):
        reply = QMessageBox.question(self, "Reset", "Reset all parameters to default?")
        if reply == QMessageBox.StandardButton.Yes:
            self.param_panel.reset_to_default()
            self.disorder_panel.reset_to_default()
            self.is_parameters_locked = False
            self.lock_btn.setChecked(False)
            self.lock_btn.setText("🔓 Lock")
            self.lock_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
            self.plot_panel.clear()
            self._update_title()
            self.progress_label.setText("Ready")
            self.main_progress.setValue(0)
            
    def _gather_all_parameters(self) -> Dict[str, Any]:
        params = {}
        params.update(self.param_panel.get_parameters())
        params.update(self.disorder_panel.get_parameters())
        params['job_parallel'] = [self.outer_parallel.value(), self.inner_parallel.value()]
        params['use_remote'] = self.remote_check.isChecked()
        params['remote_host'] = self.remote_host.text()
        params['remote_port'] = self.remote_port.value()
        params['key_file'] = self.key_file.text()
        return params
               
    # 在 MainWindow 中修改 _on_progress_updated：
    def _on_progress_updated(self, current: int, total: int, message: str, current_params: dict):
        """更新进度显示 - 显示实时参数"""
        percent = int(100 * current / total) if total > 0 else 0
        self.main_progress.setValue(percent)
        
        # 构建显示消息
        display_msg = f"{message}"
        if current_params:
            # 显示关键参数
            if 'phi' in current_params:
                display_msg += f" | φ={current_params['phi']:.3f}"
            if 'B' in current_params:
                display_msg += f" | B={current_params['B']:.3f}"
            if 'Vbias' in current_params:
                display_msg += f" | V={current_params['Vbias']:.3f}"
        
        # 显示进度分数
        progress_text = f"[{current}/{total}]"
        
        self.progress_label.setText(f"{progress_text} {display_msg}")
        self.status_msg.setText(f"{progress_text} {message}")        
        
    def _on_calculation_finished(self, success: bool, message: str, result):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self.progress_label.setText("Completed")
            self.main_progress.setValue(100)
            metadata, data = result
            
            try:
                plot_path = JJPlotter.plot_single_result(
                    metadata, data,
                    self.current_solver.path_manager.get_plots_dir()
                )
                self.plot_panel.load_image(plot_path)
            except Exception as e:
                import traceback
                error_msg = f"Plot Error: {str(e)}\n{traceback.format_exc()}"
                QMessageBox.warning(self, "Plot Error", error_msg)
        else:
            # 显示完整错误信息
            import traceback
            full_error = f"{message}\n\n{traceback.format_exc()}"
            self.progress_label.setText(f"Failed: {message[:50]}")
            self.plot_panel.clear()
            QMessageBox.critical(self, "Calculation Error", full_error)
            
    def _on_new_task(self):
        reply = QMessageBox.question(self, "New Task", "Clear current task?")
        if reply == QMessageBox.StandardButton.Yes:
            self._on_reset_parameters()
            
    def _on_load_parameters(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Parameters", "", "JSON (*.json)")
        if filename:
            import json
            with open(filename, 'r') as f:
                params = json.load(f)
            self.param_panel.set_parameters(params)
            self.disorder_panel.set_parameters(params)
            
    def _on_save_parameters(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Parameters", "", "JSON (*.json)")
        if filename:
            import json
            params = self._gather_all_parameters()
            with open(filename, 'w') as f:
                json.dump(params, f, indent=4)
                
    def _on_preview_disorder(self):
        from ..core.hamiltonian import HamiltonianBuilder
        params = self._gather_all_parameters()
        builder = HamiltonianBuilder(params)
        fig = JJPlotter.plot_disorder_preview(
            builder.disorder_distribution,
            params['N_SC'], params['N_junction'],
            params.get('disorder_type', 'none'),
            params.get('disorder_region', 'all_leads')
        )
        self.plot_panel.set_figure(fig)
        
    def _on_about(self):
        QMessageBox.about(self, "About", "JJ Simulator v1.0")
        
    def _update_status(self):
        pass
        
    def closeEvent(self, event):
        if self.calculation_thread and self.calculation_thread.isRunning():
            reply = QMessageBox.question(self, "Exit", "Calculation running. Exit?")
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont("Monaco" if sys.platform == "darwin" else "Consolas", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
