#!/usr/bin/env python3
"""
Eridian — Map the world in 3D. One frame at a time.
Just run:  python3 run.py
"""

import sys, os, subprocess, platform
from pathlib import Path

VENV_DIR = Path(__file__).parent / ".venv_eridian"

# ──────────────────────────────────────────────────────────────
#  STEP 1 — If not inside our venv, create it and re-launch
# ──────────────────────────────────────────────────────────────

def in_our_venv():
    return str(VENV_DIR.resolve()) in sys.executable

def relaunch_in_venv():
    print(f"[Setup] Creating virtual environment at {VENV_DIR} ...")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    venv_python = VENV_DIR / ("Scripts/python.exe" if platform.system() == "Windows"
                               else "bin/python")
    print("[Setup] Relaunching inside venv...\n")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

if not in_our_venv():
    relaunch_in_venv()

# ──────────────────────────────────────────────────────────────
#  STEP 2 — Install deps (open3d replaced with rerun-sdk,
#           which supports Python 3.12+ and 3.13)
# ──────────────────────────────────────────────────────────────

REQUIRED = {
    "torch":       "torch",
    "torchvision": "torchvision",
    "cv2":         "opencv-python",
    "rerun":       "rerun-sdk",
    "timm":        "timm",
    "numpy":       "numpy",
}

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *pkgs])

def ensure_deps():
    missing = []
    for mod, pkg in REQUIRED.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return

    print(f"[Setup] Installing: {', '.join(missing)}")

    torch_pkgs = [p for p in missing if p in ("torch", "torchvision")]
    other_pkgs = [p for p in missing if p not in ("torch", "torchvision")]

    if torch_pkgs:
        system = platform.system()
        if system == "Darwin":
            pip(*torch_pkgs)
        elif system == "Linux" and _has_nvidia():
            pip(*torch_pkgs, "--index-url", "https://download.pytorch.org/whl/cu118")
        else:
            pip(*torch_pkgs, "--index-url", "https://download.pytorch.org/whl/cpu")
    if other_pkgs:
        pip(*other_pkgs)

    print("[Setup] All dependencies installed ✓\n")

def _has_nvidia():
    return subprocess.call(["which", "nvidia-smi"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

ensure_deps()

# ──────────────────────────────────────────────────────────────
#  STEP 3 — Real imports
# ──────────────────────────────────────────────────────────────

import cv2, numpy as np, torch, rerun as rr
import time, threading, signal

# ──────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────

SPLAT_DIR        = Path("./splat")
SAVE_INTERVAL    = 30        # frames between auto-saves
BACKUP_INTERVAL  = 150       # frames between timestamped backups
MAX_POINTS       = 800_000
PROCESS_W        = 640
PROCESS_H        = 480
POINT_STEP       = 4         # sample every Nth pixel; press +/- to adjust
MIN_D            = 0.05      # discard disparity below this (too far)
MAX_D            = 0.92      # discard disparity above this (too close / noise)
MAX_DEPTH_M      = 8.0       # metric scale ceiling in metres
FOV_DEG          = 77.0      # horizontal FOV — tune for your camera
DEPTH_MODEL      = "MiDaS_small"   # or "DPT_Hybrid" for better quality

# ──────────────────────────────────────────────────────────────
#  DEPTH ESTIMATOR  (MiDaS, fully local)
# ──────────────────────────────────────────────────────────────

class DepthEstimator:
    def __init__(self):
        print(f"[Depth] Loading {DEPTH_MODEL} ...")
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("mps")  if torch.backends.mps.is_available() else
            torch.device("cpu")
        )
        print(f"[Depth] Device: {self.device}")
        self.model = torch.hub.load("intel-isl/MiDaS", DEPTH_MODEL, trust_repo=True)
        self.model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = (transforms.small_transform
                          if "small" in DEPTH_MODEL.lower()
                          else transforms.dpt_transform)
        print("[Depth] Ready ✓")

    @torch.inference_mode()
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        inp  = self.transform(rgb).to(self.device)
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=rgb.shape[:2],
            mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()
        lo, hi = pred.min(), pred.max()
        return ((pred - lo) / (hi - lo + 1e-8)).astype(np.float32)

# ──────────────────────────────────────────────────────────────
#  POSE TRACKER  (Lucas-Kanade + Essential Matrix)
# ──────────────────────────────────────────────────────────────

class PoseTracker:
    def __init__(self, K):
        self.K   = K
        self.lk  = dict(winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.gft = dict(maxCorners=500, qualityLevel=0.03, minDistance=7, blockSize=7)
        self.prev_gray = None
        self.prev_pts  = None
        self.R     = np.eye(3, dtype=np.float64)
        self.t     = np.zeros((3, 1), dtype=np.float64)
        self.scale = 1.0
        self.n     = 0

    def update(self, gray, depth=None):
        self.n += 1
        if self.prev_gray is None:
            self._detect(gray)
            self.prev_gray = gray
            return self._T()

        if self.prev_pts is None or len(self.prev_pts) < 10:
            self._detect(gray)
            self.prev_gray = gray
            return self._T()

        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray,
                                               self.prev_pts, None, **self.lk)
        if nxt is None or st is None:
            self._detect(gray)
            self.prev_gray = gray
            return self._T()

        ok = st.ravel() == 1
        if ok.sum() < 8:
            self._detect(gray)
            self.prev_gray = gray
            return self._T()

        p1, p2 = self.prev_pts[ok], nxt[ok]
        try:
            E, mask = cv2.findEssentialMat(p1, p2, self.K,
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None:
                raise ValueError("No essential matrix")
            _, R_rel, t_rel, inlier_mask = cv2.recoverPose(E, p1, p2, self.K)
        except Exception:
            self._detect(gray)
            self.prev_gray = gray
            return self._T()

        # Scale translation from depth at inlier feature points
        if depth is not None and inlier_mask is not None:
            h, w = depth.shape
            inliers = p1[inlier_mask.ravel() > 0].astype(int)
            depths = [
                (1.0 - depth[
                    min(max(pt[1], 0), h - 1),
                    min(max(pt[0], 0), w - 1)
                ]) * MAX_DEPTH_M + 0.1
                for pt in inliers[:60]
                if MIN_D < depth[
                    min(max(pt[1], 0), h - 1),
                    min(max(pt[0], 0), w - 1)
                ] < MAX_D
            ]
            if len(depths) > 3:
                self.scale = 0.85 * self.scale + 0.15 * (float(np.median(depths)) * 0.08)

        self.R = self.R @ R_rel.T
        self.t = self.t - self.scale * self.R @ t_rel

        # Refresh feature points every 20 frames or when too few tracked
        keep = nxt[ok].reshape(-1, 1, 2)
        if ok.sum() >= 50 and self.n % 20 != 0:
            self.prev_pts = keep
        else:
            self.prev_pts = None
            self._detect(gray)

        self.prev_gray = gray
        return self._T()

    def _detect(self, gray):
        self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.gft)

    def _T(self):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3,  3] = self.t.ravel()
        return T

    def reset(self):
        self.R         = np.eye(3, dtype=np.float64)
        self.t         = np.zeros((3, 1), dtype=np.float64)
        self.prev_gray = None
        self.prev_pts  = None
        self.scale     = 1.0
        self.n         = 0

# ──────────────────────────────────────────────────────────────
#  SPLAT BUILDER
# ──────────────────────────────────────────────────────────────

class SplatBuilder:
    def __init__(self):
        self._lock = threading.Lock()
        self._pos  = []; self._col = []
        self._sc   = []; self._op  = []; self._rot = []
        self.total  = 0
        self.frames = 0

    def add(self, rgb, depth, pose, K, step=POINT_STEP):
        h, w = depth.shape
        us = np.arange(0, w, step)
        vs = np.arange(0, h, step)
        uu, vv = np.meshgrid(us, vs)
        uu, vv = uu.ravel(), vv.ravel()

        d   = depth[vv, uu]
        ok  = (d > MIN_D) & (d < MAX_D)
        if ok.sum() < 5:
            return

        d, uu, vv = d[ok], uu[ok], vv[ok]
        dm = 0.1 + (1.0 - d) * (MAX_DEPTH_M - 0.1)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts = np.stack([(uu - cx) * dm / fx,
                        (vv - cy) * dm / fy,
                        dm,
                        np.ones_like(dm)], axis=1)

        world_pts = (pose @ pts.T).T[:, :3].astype(np.float32)
        colors    = rgb[vv, uu, :].astype(np.float32) / 255.0
        sv        = np.clip(dm / fx * step * 0.5, 0.002, 0.15).astype(np.float32)
        scales    = np.stack([sv, sv, sv * 0.25], axis=1)
        opacity   = np.clip(0.9 - d * 0.3, 0.2, 0.95).astype(np.float32)
        rot       = np.zeros((len(world_pts), 4), np.float32)
        rot[:, 0] = 1.0   # identity quaternion (w=1)

        with self._lock:
            self._pos.append(world_pts)
            self._col.append(colors)
            self._sc.append(scales)
            self._op.append(opacity)
            self._rot.append(rot)
            self.total  += len(world_pts)
            self.frames += 1

        if self.total > MAX_POINTS and self.frames % 8 == 0:
            self._downsample()

    def _downsample(self):
        with self._lock:
            pos = np.concatenate(self._pos); col = np.concatenate(self._col)
            sc  = np.concatenate(self._sc);  op  = np.concatenate(self._op)
            rot = np.concatenate(self._rot)
            target = int(MAX_POINTS * 0.8)
            if len(pos) > target:
                idx = np.random.choice(len(pos), target, replace=False)
                pos, col, sc, op, rot = pos[idx], col[idx], sc[idx], op[idx], rot[idx]
            self._pos = [pos]; self._col = [col]
            self._sc  = [sc];  self._op  = [op]; self._rot = [rot]
            self.total = len(pos)
        print(f"[Splat] Downsampled -> {self.total:,}")

    def arrays(self):
        """Return (positions, colors) numpy arrays, or (None, None) if empty."""
        with self._lock:
            if not self._pos:
                return None, None
            return np.concatenate(self._pos), np.concatenate(self._col)

    def save(self, path: str):
        """Write a standard 3DGS .ply file (atomic write)."""
        with self._lock:
            if not self._pos:
                print("[Save] Nothing to save yet.")
                return
            pos = np.concatenate(self._pos); col = np.concatenate(self._col)
            sc  = np.concatenate(self._sc);  op  = np.concatenate(self._op)
            rot = np.concatenate(self._rot)

        n   = len(pos)
        C0  = 0.28209479177387814          # SH zero-order coefficient
        sh  = (col - 0.5) / C0
        sl  = np.log(np.clip(sc, 1e-7, None))
        ol  = np.log(np.clip(op, 1e-6, 1 - 1e-6) / (1 - np.clip(op, 1e-6, 1 - 1e-6)))

        header = (
            f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
            "property float opacity\n"
            "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
            "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
            "end_header\n"
        )
        data = np.zeros((n, 17), np.float32)
        data[:, 0:3]   = pos
        data[:, 6:9]   = sh
        data[:, 9]     = ol
        data[:, 10:13] = sl
        data[:, 13:17] = rot

        tmp = path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(header.encode())
                f.write(data.tobytes())
            os.replace(tmp, path)
            print(f"[Save] ✓  {n:,} gaussians  →  {path}")
        except Exception as e:
            print(f"[Save] ✗  {e}")

    def reset(self):
        with self._lock:
            self._pos = []; self._col = []
            self._sc  = []; self._op  = []; self._rot = []
            self.total = 0; self.frames = 0

# ──────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────

def best_camera():
    """Pick the highest-resolution camera available."""
    print("[Camera] Scanning ...")
    best_cap, best_res, best_idx = None, 0, 0
    for i in range(8):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"  [{i}] {int(w)}×{int(h)}")
        if w * h >= best_res:
            if best_cap:
                best_cap.release()
            best_cap, best_res, best_idx = cap, w * h, i
        else:
            cap.release()
    return best_cap, best_idx

def make_K(w, h, fov_deg=FOV_DEG):
    """Build a simple pinhole intrinsics matrix."""
    fx = w / (2 * np.tan(np.radians(fov_deg) / 2))
    return np.array([[fx, 0, w / 2],
                     [0, fx, h / 2],
                     [0,  0,     1]], dtype=np.float64)

# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    SPLAT_DIR.mkdir(parents=True, exist_ok=True)

    banner = "ERIDIAN  —  Map the world in 3D. One frame at a time."
    print(f"\n{'═' * len(banner)}\n  {banner}\n  Output → {SPLAT_DIR.resolve()}\n{'═' * len(banner)}\n")

    # ── Rerun viewer (opens automatically in your browser or native app) ──
    rr.init("eridian", spawn=True)
    rr.log("world/origin",
           rr.Arrows3D(
               vectors=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.4]],
               colors=[[255, 60, 60], [60, 255, 60], [60, 60, 255]],
           ),
           static=True)

    depth_est = DepthEstimator()

    cap, cam_idx = best_camera()
    if cap is None:
        print("[Error] No camera found."); sys.exit(1)
    ret, frame0 = cap.read()
    if not ret:
        print("[Error] Cannot read from camera."); sys.exit(1)

    oh, ow = frame0.shape[:2]
    scale  = min(PROCESS_W / ow, PROCESS_H / oh, 1.0)
    pw, ph = int(ow * scale), int(oh * scale)
    print(f"[Camera] {cam_idx}: {ow}×{oh}  →  processing at {pw}×{ph}\n")

    K     = make_K(pw, ph)
    track = PoseTracker(K)
    splat = SplatBuilder()

    running    = True
    show_depth = False
    step       = POINT_STEP
    saving     = False
    traj_pts: list = []

    # ── Background save helper ──────────────────────────────────
    def do_save(tag="splat"):
        nonlocal saving
        if saving:
            return
        saving = True
        path = str(SPLAT_DIR / f"{tag}.ply")
        def _worker():
            nonlocal saving
            splat.save(path)
            saving = False
        threading.Thread(target=_worker, daemon=True).start()

    def shutdown(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    fc  = 0
    fpc = 0
    fps = 0.0
    t0  = time.time()

    print("Controls  (camera window focused)")
    print("  Q / Esc  —  quit and save      S  —  force save now")
    print("  R        —  reset map + pose   D  —  toggle depth overlay")
    print("  +  /  -  —  denser / sparser point cloud\n")

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[Camera] Disconnected — saving final splat ...")
            break

        frm = cv2.resize(frame, (pw, ph)) if scale < 1.0 else frame
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        gry = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        depth = depth_est.estimate(rgb)
        pose  = track.update(gry, depth)
        splat.add(rgb, depth, pose, K, step=step)

        # Track camera path
        cam_pos = pose[:3, 3].tolist()
        traj_pts.append(cam_pos)

        # ── Log to Rerun every frame ────────────────────────────
        rr.set_time_sequence("frame", fc)

        # Camera feeds
        rr.log("camera/rgb",   rr.Image(rgb))
        rr.log("camera/depth", rr.DepthImage(depth, meter=1.0 / MAX_DEPTH_M))

        # Camera trajectory
        if len(traj_pts) > 1:
            rr.log("world/trajectory",
                   rr.LineStrips3D([traj_pts],
                                   colors=[[80, 255, 130]],
                                   radii=0.005))

        # Full point cloud — update every SAVE_INTERVAL frames (expensive)
        if fc % SAVE_INTERVAL == 0:
            pos, col = splat.arrays()
            if pos is not None:
                # radii proportional to density step
                radii = np.full(len(pos), 0.005 * step, dtype=np.float32)
                rr.log("world/splat",
                       rr.Points3D(pos,
                                   colors=(col * 255).astype(np.uint8),
                                   radii=radii))

        # ── Auto-save PLY ───────────────────────────────────────
        if fc % SAVE_INTERVAL    == 0: do_save("splat")
        if fc % BACKUP_INTERVAL  == 0 and fc > 0: do_save(f"splat_{fc:06d}")

        # ── FPS readout ─────────────────────────────────────────
        fpc += 1
        if fpc >= 20:
            fps = fpc / (time.time() - t0)
            t0  = time.time()
            fpc = 0
            tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]
            print(f"  FPS:{fps:5.1f}  Pts:{splat.total:>8,}  "
                  f"Frame:{fc:>5}  Pos:({tx:+.2f}, {ty:+.2f}, {tz:+.2f})")

        # ── OpenCV preview window ───────────────────────────────
        if show_depth:
            depth_vis = cv2.applyColorMap((depth * 255).astype(np.uint8),
                                          cv2.COLORMAP_MAGMA)
            disp = np.hstack([frm, depth_vis])
        else:
            disp = frm.copy()

        cv2.putText(disp, f"FPS:{fps:.0f}  Pts:{splat.total:,}  F:{fc}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(disp, "Q=quit  S=save  R=reset  D=depth  +/-=density",
                    (8, disp.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.imshow("Eridian — Capture", disp)

        k = cv2.waitKey(1) & 0xFF
        if   k in (ord('q'), 27):              running = False
        elif k == ord('s'):                    do_save("splat")
        elif k == ord('r'):
            track.reset(); splat.reset()
            traj_pts.clear()
            print("[Reset] Map and pose cleared.")
        elif k == ord('d'):                    show_depth = not show_depth
        elif k in (ord('+'), ord('=')):
            step = max(1, step - 1)
            print(f"[Density] step={step}")
        elif k == ord('-'):
            step = min(12, step + 1)
            print(f"[Density] step={step}")

        fc += 1

    # ── Clean shutdown ──────────────────────────────────────────
    print("\n[Shutdown] Writing final splat files ...")
    splat.save(str(SPLAT_DIR / "splat.ply"))
    splat.save(str(SPLAT_DIR / "splat_final.ply"))
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'═' * 50}\n"
          f"  Done!  {fc} frames  |  {splat.total:,} gaussians\n"
          f"  View your splat →  https://supersplat.playcanvas.com\n"
          f"{'═' * 50}\n")

if __name__ == "__main__":
    main()
