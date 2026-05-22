"""
PTZ Intelligent Tracking System
Monolithic script for Pastor tracking via YOLO + VISCA + RTSP
NEOiD PTZ NDI 20X 2 | UDP 1259 | CUDA required
"""

import cv2
import threading
import socket
import time
import atexit
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import numpy as np


# ============================================================================
# THREAD 1: RTSP Stream Capture (Non-blocking)
# ============================================================================

class RTSPStream:
    """Captures frames from RTSP stream in a background thread."""
    
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # CRITICAL: prevents frame lag
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()
    
    def _reader(self):
        """Background thread that continuously reads frames."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)  # Brief pause if no frame available
    
    def read(self):
        """Thread-safe frame read."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        """Gracefully stop the stream."""
        self.running = False
        self.reader_thread.join(timeout=2)
        self.cap.release()


# ============================================================================
# THREAD 3: VISCA Command Sender (Throttled)
# ============================================================================

class VISCASender:
    """Sends throttled VISCA commands via UDP to camera."""
    
    def __init__(self, ip="192.168.0.182", port=1259):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_sent = 0
        self.min_interval = 0.08  # ~12 commands/sec max
        self.lock = threading.Lock()
    
    def send(self, pan_speed: int, pan_dir: int):
        """
        Send pan command. pan_speed 0-24, pan_dir 0x01 (right) or 0x02 (left).
        """
        now = time.time()
        with self.lock:
            if now - self.last_sent < self.min_interval:
                return
            
            if pan_speed == 0:
                # Full stop command
                cmd = bytes([0xFF, 0x01, 0x00, 0x06, 0x01, 0x00, 0x00, 0x03, 0x03])
            else:
                # Pan move command
                cmd = bytes([0xFF, 0x01, 0x00, 0x06, 0x01, pan_speed, 0x01, pan_dir, 0x03])
            
            try:
                self.sock.sendto(cmd, (self.ip, self.port))
            except Exception as e:
                print(f"[VISCA] Send error: {e}")
            
            self.last_sent = now


# ============================================================================
# Speed Calculation (Proportional / LERP)
# ============================================================================

def calc_pan_speed(offset_x: float, frame_w: int, deadzone_ratio: float, sensitivity: float) -> int:
    """
    Maps horizontal pixel offset to VISCA pan speed (1–24).
    Returns 0 if inside deadzone.
    """
    half_dz = (frame_w * deadzone_ratio) / 2
    
    if abs(offset_x) < half_dz:
        return 0
    
    excess = abs(offset_x) - half_dz
    available = (frame_w / 2) - half_dz
    
    if available <= 0:
        return 0
    
    ratio = min(excess / available, 1.0)
    speed = int(ratio * 24 * sensitivity)
    
    return max(1, min(speed, 24))


# ============================================================================
# Target Selection
# ============================================================================

def select_target(detections, frame_w, frame_h):
    """Pick the most centered person bounding box."""
    cx = frame_w / 2
    best = None
    best_score = float('inf')
    
    for det in detections:
        x1, y1, x2, y2 = det.xyxy[0]
        bx = (x1 + x2) / 2
        
        dist = abs(bx - cx)  # Prefer center-most person
        if dist < best_score:
            best_score = dist
            best = det
    
    return best


# ============================================================================
# Main Tracking Loop
# ============================================================================

def tracking_step(frame, model, visca, deadzone_w_ratio, sensitivity, status_callback):
    """Execute one tracking step."""
    if frame is None:
        status_callback("Aguardando frame...", 0.0)
        visca.send(0, 0x03)  # Stop if no frame
        return 0.0
    
    h, w = frame.shape[:2]
    
    # Run YOLO inference on GPU
    results = model(frame, classes=[0], device=0, verbose=False)
    
    if not results[0].boxes:
        status_callback("Nenhuma pessoa detectada", 0.0)
        visca.send(0, 0x03)  # No detection → stop
        return 0.0
    
    target = select_target(results[0].boxes, w, h)
    if target is None:
        status_callback("Alvo não selecionado", 0.0)
        visca.send(0, 0x03)
        return 0.0
    
    x1, y1, x2, y2 = target.xyxy[0]
    target_cx = (x1 + x2) / 2
    offset_x = target_cx - (w / 2)
    
    pan_speed = calc_pan_speed(offset_x, w, deadzone_w_ratio, sensitivity)
    pan_dir = 0x01 if offset_x > 0 else 0x02  # Right=0x01, Left=0x02
    
    if pan_speed == 0:
        status_callback("Dentro da zona segura", 0.0)
        visca.send(0, 0x03)
    else:
        status_callback(f"Rastreando | Velocidade: {pan_speed}", 0.0)
        visca.send(pan_speed, pan_dir)
    
    return 1.0 / (time.time() - time.time() + 0.001)  # Rough FPS estimate


# ============================================================================
# GUI (Tkinter)
# ============================================================================

class TrackingGUI:
    """Simple control interface for deadzone and sensitivity."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PTZ Tracker — NEOiD 20X")
        self.root.geometry("450x250")
        self.root.resizable(False, False)
        
        self.deadzone_var = tk.DoubleVar(value=0.30)
        self.sensitivity_var = tk.DoubleVar(value=1.0)
        self.status_var = tk.StringVar(value="Iniciando...")
        self.fps_var = tk.StringVar(value="FPS: --")
        
        self._build_ui()
        
        # Tracking state
        self.tracking_active = False
        self.stream = None
        self.model = None
        self.visca = None
        self.inference_thread = None
        self.stop_tracking = False
    
    def _build_ui(self):
        """Build the UI layout."""
        pad = {"padx": 16, "pady": 6}
        
        # Title
        title = ttk.Label(self.root, text="Sistema de Rastreamento PTZ", font=("Arial", 12, "bold"))
        title.grid(row=0, columnspan=3, sticky="w", padx=16, pady=10)
        
        # Deadzone control
        ttk.Label(self.root, text="Zona Segura (%)", font=("Arial", 10)).grid(row=1, column=0, sticky="w", **pad)
        ttk.Scale(self.root, from_=5, to=60, variable=self.deadzone_var,
                  orient="horizontal", length=220).grid(row=1, column=1, **pad)
        self.dz_label = ttk.Label(self.root, text="30%", font=("Arial", 10))
        self.dz_label.grid(row=1, column=2, sticky="w", **pad)
        self.deadzone_var.trace_add("write", self._update_dz_label)
        
        # Sensitivity control
        ttk.Label(self.root, text="Sensibilidade", font=("Arial", 10)).grid(row=2, column=0, sticky="w", **pad)
        ttk.Scale(self.root, from_=0.1, to=2.0, variable=self.sensitivity_var,
                  orient="horizontal", length=220).grid(row=2, column=1, **pad)
        self.sens_label = ttk.Label(self.root, text="1.0x", font=("Arial", 10))
        self.sens_label.grid(row=2, column=2, sticky="w", **pad)
        self.sensitivity_var.trace_add("write", self._update_sens_label)
        
        # Separator
        ttk.Separator(self.root, orient="horizontal").grid(row=3, columnspan=3, sticky="ew", pady=8)
        
        # Status display
        ttk.Label(self.root, textvariable=self.status_var, foreground="green", 
                  font=("Arial", 10)).grid(row=4, columnspan=3, sticky="w", **pad)
        ttk.Label(self.root, textvariable=self.fps_var, font=("Arial", 10)).grid(row=5, columnspan=3, sticky="w", **pad)
        
        # Separator
        ttk.Separator(self.root, orient="horizontal").grid(row=6, columnspan=3, sticky="ew", pady=8)
        
        # Control buttons
        self.start_btn = ttk.Button(self.root, text="Iniciar Rastreamento", command=self.start_tracking)
        self.start_btn.grid(row=7, column=0, sticky="ew", padx=8, pady=10)
        
        self.stop_btn = ttk.Button(self.root, text="Parar", command=self.stop_tracking_btn, state="disabled")
        self.stop_btn.grid(row=7, column=1, sticky="ew", padx=8, pady=10)
        
        self.exit_btn = ttk.Button(self.root, text="Sair", command=self.root.quit)
        self.exit_btn.grid(row=7, column=2, sticky="ew", padx=8, pady=10)
    
    def _update_dz_label(self, *args):
        val = int(self.deadzone_var.get() * 100)
        self.dz_label.config(text=f"{val}%")
    
    def _update_sens_label(self, *args):
        val = self.sensitivity_var.get()
        self.sens_label.config(text=f"{val:.1f}x")
    
    def start_tracking(self):
        """Initialize RTSP, YOLO, VISCA and start tracking thread."""
        self.status_var.set("Carregando modelo YOLO...")
        self.root.update()
        
        try:
            # Initialize RTSP stream
            rtsp_url = "rtsp://192.168.0.182:554/live/ch0"
            self.stream = RTSPStream(rtsp_url)
            print(f"[RTSP] Conectado a {rtsp_url}")
            
            # Load YOLO model
            self.model = YOLO("yolov8n.pt")
            self.model.to("cuda:0")
            print("[YOLO] Modelo carregado na GPU (CUDA:0)")
            
            # Initialize VISCA sender
            self.visca = VISCASender()
            print("[VISCA] Inicializado para 192.168.0.182:1259")
            
            self.tracking_active = True
            self.stop_tracking = False
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_var.set("Rastreamento ativo")
            
            # Start inference thread
            self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inference_thread.start()
            
        except Exception as e:
            self.status_var.set(f"Erro ao iniciar: {str(e)[:40]}")
            print(f"[ERROR] {e}")
    
    def stop_tracking_btn(self):
        """Stop tracking and cleanup."""
        self.stop_tracking = True
        self.tracking_active = False
        
        if self.stream:
            self.stream.stop()
        if self.visca:
            self.visca.send(0, 0x03)  # Final stop command
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Rastreamento interrompido")
        print("[TRACKER] Parado")
    
    def _inference_loop(self):
        """Background inference loop."""
        frame_count = 0
        fps_time = time.time()
        fps_counter = 0.0
        
        while not self.stop_tracking and self.tracking_active:
            try:
                frame = self.stream.read()
                if frame is None:
                    time.sleep(0.05)
                    continue
                
                # Run tracking step
                fps_counter = tracking_step(
                    frame, self.model, self.visca,
                    self.deadzone_var.get(), self.sensitivity_var.get(),
                    self._update_status
                )
                
                frame_count += 1
                
                # Update FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_time
                    estimated_fps = 30 / elapsed if elapsed > 0 else 0
                    self.fps_var.set(f"FPS: {estimated_fps:.1f}")
                    fps_time = time.time()
                
            except Exception as e:
                print(f"[INFERENCE] Erro: {e}")
                time.sleep(0.1)
    
    def _update_status(self, msg, fps=None):
        """Thread-safe status update."""
        self.status_var.set(msg)
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()


# ============================================================================
# Shutdown Handler
# ============================================================================

_global_visca = None

def on_exit():
    """Send final STOP command on script exit."""
    if _global_visca:
        try:
            stop_cmd = bytes([0xFF, 0x01, 0x00, 0x06, 0x01, 0x00, 0x00, 0x03, 0x03])
            _global_visca.sock.sendto(stop_cmd, ("192.168.0.182", 1259))
            print("[EXIT] Comando STOP enviado para a câmera")
        except:
            pass

atexit.register(on_exit)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PTZ Intelligent Tracking System")
    print("NEOiD PTZ NDI 20X 2 | CUDA Required")
    print("=" * 60)
    
    gui = TrackingGUI()
    _global_visca = gui.visca  # Store for atexit handler
    
    print("[STARTUP] Interface iniciada")
    gui.run()
