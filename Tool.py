#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import shutil
import subprocess
import json
import math
import threading
import queue
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, Listbox, messagebox
from functools import lru_cache

try:
    from PIL import Image, ImageTk
    import cv2
    PREVIEW_ENABLED = True
except ImportError:
    PREVIEW_ENABLED = False

try:
    from matplotlib.font_manager import findSystemFonts, fontManager
    FONT_MANAGER_ENABLED = True
except ImportError:
    FONT_MANAGER_ENABLED = False

# ==============================================================================
# PHẦN 1: CORE LOGIC
# ==============================================================================

# ----- CONFIG -----
VIDEO_CODEC    = "libx264"
CRF            = "23"
PRESET         = "veryfast"
PIX_FMT        = "yuv420p"
AUDIO_CODEC    = "aac"
AUDIO_BITRATE  = "192k"
LOOP_BGM       = True
SETTINGS_FILE  = "last_settings_joinvideo.json"

# ----- THƯ MỤC -----
BASE = Path.cwd()
DIR_VIDEO_INPUT      = BASE / "Video_Input"
DIR_BGM              = BASE / "Sound_Background"
DIR_FONT             = BASE / "Font"
DIR_TEMP_OUTPUT      = BASE / "Temp_Output"
DIR_DONE_INPUT       = BASE / "DONE" / "Video_Input"
DIR_VIDEO_OUTPUT     = BASE / "Video_Output"
DIR_VIDEO_ERRORS     = BASE / "Video_Errors"

for d in [DIR_VIDEO_INPUT, DIR_BGM, DIR_FONT, DIR_TEMP_OUTPUT, DIR_DONE_INPUT, DIR_VIDEO_OUTPUT, DIR_VIDEO_ERRORS]:
    d.mkdir(parents=True, exist_ok=True)

# ----- HELPERS -----
log_queue = queue.Queue()

def log_message(msg: str):
    log_queue.put(msg)

def run(cmd: List[str]):
    cmd_str_list = [str(c) for c in cmd]
    log_message("RUNNING: " + " ".join(cmd_str_list))
    process = subprocess.Popen(cmd_str_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    for line in iter(process.stdout.readline, ''):
        log_message(line.strip())
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, "FFmpeg command failed.")

@lru_cache(maxsize=32)
def get_video_thumbnail(video_path: str, size=(240, 135)):
    if not PREVIEW_ENABLED or not os.path.exists(video_path):
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        frame_pos = min(int(cap.get(cv2.CAP_PROP_FPS) * 1) if cap.get(cv2.CAP_PROP_FPS) else 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))
        frame_pos = max(1, frame_pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail(size)
            return ImageTk.PhotoImage(img)
    except Exception:
        return None
    return None

def ffprobe_duration(path: Path) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path)
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        out, err = process.communicate(timeout=10)
        if process.returncode == 0 and out:
             duration = float(out.strip())
             return duration if duration > 0 else 0.0
        else:
             log_message(f"FFprobe error getting duration for {path.name}: {err.strip()}")
             return 0.0
    except Exception as e:
        log_message(f"Exception getting duration for {path.name}: {e}")
        return 0.0

def pick_random_bgm(selected_paths: Optional[List[str]] = None) -> Optional[Path]:
    if selected_paths:
        valid_selected = [p for p in selected_paths if Path(p).exists()]
        return Path(random.choice(valid_selected)) if valid_selected else None
    cands = [p for p in DIR_BGM.glob("*") if p.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".flac"}]
    return random.choice(cands) if cands else None

def group_videos_by_prefix(video_source: Union[List[Path], Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    pat = re.compile(r"^(.+?_)\d+\.mp4$", re.IGNORECASE)
    files_to_scan = []
    if isinstance(video_source, Path):
        files_to_scan = sort_videos_numerically(list(video_source.glob("*.mp4")))
    elif isinstance(video_source, list):
        files_to_scan = [p for p in video_source if isinstance(p, Path)]
    for f in files_to_scan:
        m = pat.match(f.name)
        if m:
            key = m.group(1)
            groups.setdefault(key, []).append(f)
    return groups

def sort_videos_numerically(video_paths: List[Path]) -> List[Path]:
    pat = re.compile(r'_(\d+)\.mp4$', re.IGNORECASE)
    def get_number(path: Path):
        match = pat.search(path.name)
        return int(match.group(1)) if match else float('inf')
    return sorted(video_paths, key=lambda p: (p.name.split('_')[0] if '_' in p.name else p.name, get_number(p)))

# ----- CORE FFmpeg & PROCESSING LOGIC -----
def normalize_video(in_path: Path, out_path: Path, target_w: int, target_h: int, orig_gain: float, clip_duration: Optional[float] = None) -> bool:
    duration_info = f"{clip_duration:.2f}s (forced)" if clip_duration else "original duration"
    log_message(f"  - Đang chuẩn hóa file: {in_path.name} -> {duration_info}")
    try:
        v_filters = [f"fps=30", f"format={PIX_FMT}"]
        a_filters = [f"volume={orig_gain:.2f}", "aresample=44100"]
        if clip_duration is not None and clip_duration > 0:
            v_filters.extend([f"tpad=stop_mode=clone:stop_duration={clip_duration}", f"trim=duration={clip_duration}"])
            a_filters.append(f"atrim=duration={clip_duration}")
        v_filters.extend([f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease", f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black", "setsar=1"])
        filter_complex = f"[0:v]{','.join(v_filters)}[v];[0:a]{','.join(a_filters)}[a]"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(in_path), "-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]", "-c:v", VIDEO_CODEC, "-preset", PRESET, "-crf", CRF, "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, str(out_path)]
        run(cmd)
        return True
    except Exception as e:
        log_message(f"  - ❌ LỖI CHUẨN HÓA file {in_path.name}: {e}")
        return False

def build_pair_concat_cmd(segment1: Path, segment2: Path, out_path: Path, transition: str, transition_duration: float) -> bool:
    log_message(f"  - Đang ghép cặp: {segment1.name} + {segment2.name}")
    try:
        duration1 = ffprobe_duration(segment1)
        duration2 = ffprobe_duration(segment2)
        effective_transition = transition
        if duration1 <= 0 or duration2 <= 0:
             log_message(f"  - Cảnh báo: Không thể lấy thời lượng của 1 trong 2 video. Sử dụng ghép nối trực tiếp.")
             effective_transition = 'none'
        if not math.isclose(duration1, duration2, rel_tol=0.1):
            log_message(f"  - Cảnh báo: Thời lượng 2 video tạm không khớp ({duration1:.3f}s vs {duration2:.3f}s). Dùng ghép nối trực tiếp để đảm bảo an toàn.")
            effective_transition = 'none'
        if effective_transition != 'none' and transition_duration > 0 and duration1 > transition_duration:
            offset = duration1 - transition_duration
            filter_complex = (f"[0:v][1:v]xfade=transition={effective_transition}:duration={transition_duration}:offset={offset},format={PIX_FMT}[v];"
                              f"[0:a][1:a]acrossfade=d={transition_duration}[a]")
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(segment1), "-i", str(segment2), "-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]", "-c:v", VIDEO_CODEC, "-preset", PRESET, "-crf", CRF, "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, str(out_path)]
        else:
            if effective_transition != 'none' and transition != 'none':
                log_message(f"  - Cảnh báo: Thời lượng video 1 ({duration1:.2f}s) không đủ dài cho hiệu ứng ({transition_duration:.2f}s). Sử dụng ghép nối trực tiếp.")
            tmp_list = out_path.with_suffix(".txt")
            with open(tmp_list, "w", encoding="utf-8") as f:
                f.write(f"file '{segment1.resolve()}'\n")
                f.write(f"file '{segment2.resolve()}'\n")
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp_list), "-c", "copy", str(out_path)]
        run(cmd)
        if 'tmp_list' in locals() and tmp_list.exists(): tmp_list.unlink()
        return True
    except Exception as e:
        log_message(f"  - ❌ LỖI khi ghép cặp: {e}")
        return False

# ==============================================================================
# SỬA LỖI CHÍNH: Thay thế hàm concat_sequential_videos
# ==============================================================================
def concat_sequential_videos(video_list: List[Path], out_path: Path) -> bool:
    """
    Sửa lại: Dùng concat demuxer (giống file Join.py gốc) để ghép nối tiếp.
    Đây là phương pháp -c copy rất nhanh, thay vì concat filter (rất chậm).
    """
    log_message(f"  - Đang ghép nối tiếp {len(video_list)} video bằng concat demuxer (nhanh)...")
    list_file = out_path.with_suffix(".txt")
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for video in video_list:
                # Phải dùng .resolve() để FFmpeg tìm thấy đường dẫn tuyệt đối
                f.write(f"file '{video.resolve()}'\n")
        
        # Lệnh gốc từ Join.py, dùng -c copy để ghép nhanh
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_path)]
        run(cmd)
        return True
    except Exception as e:
        log_message(f"  - ❌ LỖI khi ghép nối tiếp (demuxer): {e}")
        return False
    finally:
        # Quan trọng: Xóa file .txt tạm
        if list_file.exists(): 
            list_file.unlink()

def add_watermark_cmd(in_path: Path, out_path: Path, watermark_text: str, font_path: str):
    if not font_path or not Path(font_path).exists():
        log_message("⚠️ Lỗi Watermark: Đường dẫn font không hợp lệ hoặc file font không tồn tại. Bỏ qua watermark.")
        shutil.copy2(str(in_path), str(out_path))
        return
    font_path_escaped = font_path.replace("\\", "/").replace(":", "\\:")
    # Hiệu ứng này của Tool.py, khác với Join.py. Giữ nguyên.
    watermark_filter = (f"drawtext=text='{watermark_text}':fontfile='{font_path_escaped}':fontsize=40:fontcolor=white@0.3:x='(w-tw)/2+(w-tw)/2.5*sin(t/8)':y='(h-th)/3+(h-th)/4*cos(t/10)'")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(in_path), "-vf", watermark_filter, "-c:v", VIDEO_CODEC, "-preset", PRESET, "-crf", CRF, "-c:a", "copy", str(out_path)]
    run(cmd)

# ==============================================================================
# SỬA LỖI TIỀM ẨN: Thêm cờ '-t' (duration) vào mix_bgm_cmd
# ==============================================================================
def mix_bgm_cmd(video_path: Path, final_path: Path, bgm_path: Path, bgm_gain: float):
    video_duration = ffprobe_duration(video_path)
    if video_duration <= 0: raise RuntimeError(f"Could not get final video duration for {video_path.name}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path)]
    if LOOP_BGM: cmd += ["-stream_loop", "-1", "-i", str(bgm_path)]
    else: cmd += ["-i", str(bgm_path)]
    filter_complex = (f"[0:a]aresample=async=1:first_pts=0[a0];[1:a]aresample=async=1:first_pts=0,volume={bgm_gain:.2f}[bg];[a0][bg]amix=inputs=2:duration=longest:normalize=0[aout]")
    
    # SỬA LỖI: Thêm '-t' (giống hệt Join.py) để đảm bảo file dừng đúng thời lượng
    cmd += [
        "-filter_complex", filter_complex, 
        "-map", "0:v", "-map", "[aout]", 
        "-c:v", "copy", "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, 
        "-t", f"{video_duration:.3f}", # <-- ĐÃ THÊM THAM SỐ NÀY
        str(final_path)
    ]
    run(cmd)

# --- RUN MODES ---
def run_mode_random(settings: dict, video_source: Union[List[Path], Path]) -> Tuple[set, list, set]:
    log_message("\n🚀 Bắt đầu xử lý hàng loạt (Chế độ Ngẫu nhiên)...")
    video_groups = group_videos_by_prefix(video_source)
    group_keys = list(video_groups.keys())
    num_to_combine = settings.get('num_to_combine', 2)
    if len(group_keys) < num_to_combine:
        log_message(f"⚠️ Không đủ nhóm video. Cần {num_to_combine} nhóm, chỉ có {len(group_keys)}.")
        return set(), [], set()
    random.shuffle(group_keys)
    batch_index = 1
    processed_files_in_run = set()
    failed_files_in_run = set()
    created_files = []
    while len(group_keys) >= num_to_combine:
        log_message(f"\n--- 🔄 Đang xử lý lô #{batch_index} ---")
        selected_keys = [group_keys.pop(0) for _ in range(num_to_combine)]
        temp_files_to_delete: List[Path] = []
        videos_in_batch_orig = [random.choice(video_groups[key]) for key in selected_keys]
        try:
            arc = settings.get('aspect_ratio_choice')
            target_w, target_h = (1080, 1920) if arc == 'd' else (1920, 1080)
            orig_gain = settings.get('orig_gain', 1.0)
            for p in videos_in_batch_orig: processed_files_in_run.add(p)
            log_message("Bước 1: Chuẩn hóa các video đầu vào...")
            clip_duration = ffprobe_duration(videos_in_batch_orig[0])
            if clip_duration <= 0:
                log_message(f"  - ❌ LỖI: Không đo được thời lượng file chuẩn {videos_in_batch_orig[0].name}. Bỏ qua lô này.")
                for video_path in videos_in_batch_orig:
                    failed_files_in_run.add(video_path)
                    if video_path in processed_files_in_run: processed_files_in_run.remove(video_path)
                continue
            log_message(f"  - Thời lượng chuẩn cho lô này là: {clip_duration:.2f} giây.")
            normalized_videos, failed_normalize = [], []
            for video_path in videos_in_batch_orig:
                norm_path = DIR_TEMP_OUTPUT / f"NORM_{batch_index}_{video_path.name}"
                temp_files_to_delete.append(norm_path)
                if normalize_video(video_path, norm_path, target_w, target_h, orig_gain, clip_duration):
                    normalized_videos.append(norm_path)
                else:
                    failed_normalize.append(video_path)
            for video_path in failed_normalize:
                failed_files_in_run.add(video_path)
                if video_path in processed_files_in_run: processed_files_in_run.remove(video_path)
            if len(normalized_videos) < 2:
                log_message(f"  - ⚠️ Không đủ video hợp lệ để ghép. Bỏ qua lô này.")
                continue
            current_level_videos = normalized_videos
            level = 1
            while len(current_level_videos) > 1:
                log_message(f"Bước 2.{level}: Ghép cặp vòng {level}...")
                next_level_videos = []
                for i in range(0, len(current_level_videos), 2):
                    if i + 1 < len(current_level_videos):
                        clip1, clip2 = current_level_videos[i], current_level_videos[i+1]
                        pair_out_path = DIR_TEMP_OUTPUT / f"TEMP_{batch_index}_level{level}_pair{i//2}.mp4"
                        temp_files_to_delete.append(pair_out_path)
                        if not build_pair_concat_cmd(clip1, clip2, pair_out_path, settings.get('transition', 'none'), settings.get('transition_duration', 1.0)):
                            raise Exception("Ghép cặp thất bại.")
                        next_level_videos.append(pair_out_path)
                    else:
                        next_level_videos.append(current_level_videos[i])
                current_level_videos = next_level_videos
                level += 1
            final_concatenated_video = current_level_videos[0]
            current_processed_file = final_concatenated_video
            if settings.get('watermark_text'):
                log_message("Bước 3: Thêm watermark...")
                watermarked_file = DIR_TEMP_OUTPUT / f"TEMP_WM_{batch_index}.mp4"
                temp_files_to_delete.append(watermarked_file)
                add_watermark_cmd(current_processed_file, watermarked_file, settings.get('watermark_text', ""), settings.get('font_path'))
                current_processed_file = watermarked_file
            output_name_parts = [p.stem.split('_')[0] for p in videos_in_batch_orig if p not in failed_files_in_run]
            output_filename = "_".join(output_name_parts) + ".mp4"
            final_output_path = DIR_VIDEO_OUTPUT / output_filename
            if settings.get('use_bgm'):
                log_message("Bước 4: Trộn nhạc nền...")
                bgm_path = pick_random_bgm(settings.get('bgm_paths'))
                if not bgm_path:
                    log_message("⚠️ Không tìm thấy BGM. Video sẽ không có nhạc nền.")
                    shutil.move(str(current_processed_file), str(final_output_path))
                else:
                    log_message(f"🔊 Đang dùng BGM: {Path(bgm_path).name}")
                    mix_bgm_cmd(current_processed_file, final_output_path, bgm_path, settings.get('bgm_gain', 0.1))
            else:
                shutil.move(str(current_processed_file), str(final_output_path))
            created_files.append(final_output_path)
            log_message(f"✅ Hoàn thành lô #{batch_index}: {final_output_path.name}")
        except Exception as e:
            log_message(f"❌ LỖI khi xử lý lô #{batch_index}: {e}")
            for video_path in videos_in_batch_orig:
                failed_files_in_run.add(video_path)
                if video_path in processed_files_in_run:
                    processed_files_in_run.remove(video_path)
        finally:
            log_message("  - Dọn dẹp file tạm...")
            for temp_file in temp_files_to_delete:
                temp_file.unlink(missing_ok=True)
        batch_index += 1
    if group_keys:
        log_message(f"\nℹ️ Còn dư {len(group_keys)} nhóm video, không đủ để tạo lô tiếp theo.")
    return processed_files_in_run, created_files, failed_files_in_run

def run_mode_sequential(settings: dict, video_source: Union[List[Path], Path]) -> Tuple[set, list, set]:
    log_message("\n🚀 Bắt đầu xử lý hàng loạt (Chế độ Nối tiếp)...")
    video_groups = group_videos_by_prefix(video_source)
    if not video_groups:
        log_message("⚠️ Không tìm thấy nhóm video nào để xử lý.")
        return set(), [], set()
    batch_index = 1
    processed_files_in_run = set()
    failed_files_in_run = set()
    created_files = []
    sorted_prefixes = sorted(video_groups.keys())
    for prefix in sorted_prefixes:
        video_list = video_groups[prefix]
        if not video_list:
            continue
        log_message(f"\n--- 🔄 Đang xử lý nhóm '{prefix.strip('_')}' (lô #{batch_index}) ---")
        temp_files_to_delete: List[Path] = []
        videos_to_process_orig = sort_videos_numerically(video_list)
        try:
            arc = settings.get('aspect_ratio_choice')
            target_w, target_h = (1080, 1920) if arc == 'd' else (1920, 1080)
            orig_gain = settings.get('orig_gain', 1.0)
            for p in videos_to_process_orig:
                processed_files_in_run.add(p)
            log_message(f"  - Phát hiện {len(videos_to_process_orig)} video. Thứ tự ghép:")
            for i, p in enumerate(videos_to_process_orig):
                log_message(f"    {i+1}. {p.name}")
            log_message("Bước 1: Chuẩn hóa các video đầu vào...")
            normalized_videos, failed_normalize = [], []
            for video_path in videos_to_process_orig:
                norm_path = DIR_TEMP_OUTPUT / f"NORM_{batch_index}_{video_path.name}"
                temp_files_to_delete.append(norm_path)
                if normalize_video(video_path, norm_path, target_w, target_h, orig_gain, clip_duration=None):
                    normalized_videos.append(norm_path)
                else:
                    failed_normalize.append(video_path)
            for video_path in failed_normalize:
                failed_files_in_run.add(video_path)
                if video_path in processed_files_in_run:
                    processed_files_in_run.remove(video_path)
            if not normalized_videos:
                log_message(f"  - ⚠️ Không có video hợp lệ nào trong nhóm. Bỏ qua nhóm này.")
                continue
            if len(normalized_videos) > 1:
                log_message("Bước 2: Ghép nối tiếp...")
                concatenated_file = DIR_TEMP_OUTPUT / f"TEMP_CONCAT_{batch_index}.mp4"
                temp_files_to_delete.append(concatenated_file)
                if not concat_sequential_videos(normalized_videos, concatenated_file):
                    raise Exception("Ghép nối tiếp thất bại.")
                current_processed_file = concatenated_file
            else:
                current_processed_file = normalized_videos[0]
            if settings.get('watermark_text'):
                log_message("Bước 3: Thêm watermark...")
                watermarked_file = DIR_TEMP_OUTPUT / f"TEMP_WM_{batch_index}.mp4"
                temp_files_to_delete.append(watermarked_file)
                add_watermark_cmd(current_processed_file, watermarked_file, settings.get('watermark_text', ""), settings.get('font_path'))
                current_processed_file = watermarked_file
            output_filename = f"{prefix.strip('_')}_final.mp4"
            final_output_path = DIR_VIDEO_OUTPUT / output_filename
            if settings.get('use_bgm'):
                log_message("Bước 4: Trộn nhạc nền...")
                bgm_path = pick_random_bgm(settings.get('bgm_paths'))
                if not bgm_path:
                    log_message("⚠️ Không tìm thấy BGM. Video sẽ không có nhạc nền.")
                    shutil.move(str(current_processed_file), str(final_output_path))
                else:
                    log_message(f"🔊 Đang dùng BGM: {Path(bgm_path).name}")
                    mix_bgm_cmd(current_processed_file, final_output_path, bgm_path, settings.get('bgm_gain', 0.1))
            else:
                shutil.move(str(current_processed_file), str(final_output_path))
            created_files.append(final_output_path)
            log_message(f"✅ Hoàn thành nhóm '{prefix.strip('_')}': {final_output_path.name}")
        except Exception as e:
            log_message(f"❌ LỖI khi xử lý nhóm '{prefix.strip('_')}': {e}")
            for video_path in videos_to_process_orig:
                failed_files_in_run.add(video_path)
                if video_path in processed_files_in_run:
                    processed_files_in_run.remove(video_path)
        finally:
            log_message("  - Dọn dẹp file tạm...")
            for temp_file in temp_files_to_delete:
                temp_file.unlink(missing_ok=True)
        batch_index += 1
    return processed_files_in_run, created_files, failed_files_in_run

# ==============================================================================
# PHẦN 2: GIAO DIỆN NGƯỜI DÙNG (UI)
# (Không thay đổi)
# ==============================================================================

class ListboxPreviewMixin:
    def __init__(self):
        if not PREVIEW_ENABLED: return
        self.preview_window = None
        self.preview_label = None
        self._last_index = -1
        self._after_id = None
        self.bind("<Motion>", self._schedule_preview)
        self.bind("<Leave>", self._hide_preview)

    def _schedule_preview(self, event):
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(100, lambda e=event: self._show_preview(e))

    def _show_preview(self, event):
        self._after_id = None
        current_index = self.index(f"@{event.x},{event.y}")
        if current_index == self._last_index: return
        self._last_index = current_index
        if not hasattr(self, 'paths') or not self.paths or current_index >= len(self.paths):
            self._hide_preview()
            return
        video_path = str(self.paths[current_index])
        thumbnail = get_video_thumbnail(video_path)
        if thumbnail:
            if not self.preview_window or not self.preview_window.winfo_exists():
                self.preview_window = ctk.CTkToplevel(self)
                self.preview_window.overrideredirect(True)
                self.preview_window.attributes("-topmost", True)
                self.preview_label = ctk.CTkLabel(self.preview_window, text="")
                self.preview_label.pack()
            else:
                 self.preview_window.lift()
            self.preview_label.configure(image=thumbnail)
            self.preview_label.image = thumbnail
            listbox_width = self.winfo_width(); screen_width = self.winfo_screenwidth()
            x_root = self.winfo_rootx(); preview_width = thumbnail.width()
            x = x_root + listbox_width + 10
            if x + preview_width > screen_width: x = x_root - preview_width - 10
            x = max(0, x)
            y = event.y_root - thumbnail.height() // 2
            y = max(0, min(y, self.winfo_screenheight() - thumbnail.height()))
            self.preview_window.geometry(f"+{x}+{y}")
            self.preview_window.deiconify()
        else:
            self._hide_preview()

    def _hide_preview(self, event=None):
        self._last_index = -1
        if self._after_id: self.after_cancel(self._after_id); self._after_id = None
        if self.preview_window and self.preview_window.winfo_exists(): self.preview_window.withdraw()

class DraggableListbox(ListboxPreviewMixin, Listbox):
    def __init__(self, master, **kwargs):
        Listbox.__init__(self, master, **kwargs)
        if PREVIEW_ENABLED: ListboxPreviewMixin.__init__(self)
        self.bind("<Button-1>", self.on_drag_start)
        self.bind("<B1-Motion>", self.on_drag_motion)
        self.bind("<ButtonRelease-1>", self.on_drag_end)
        self.drag_start_index = None
        self.paths = []

    def update_list(self, paths: List[Path]):
        self.paths = list(paths)
        self.delete(0, 'end')
        for path in self.paths: self.insert('end', path.name)
        if PREVIEW_ENABLED: self._hide_preview()

    def on_drag_start(self, event):
        self.drag_start_index = self.index(f"@{event.x},{event.y}")

    def on_drag_motion(self, event):
        if self.drag_start_index is None: return
        new_index = self.index(f"@{event.x},{event.y}")
        if new_index != self.drag_start_index and new_index < len(self.paths):
            dragged_path = self.paths.pop(self.drag_start_index)
            self.paths.insert(new_index, dragged_path)
            dragged_item = self.get(self.drag_start_index)
            self.delete(self.drag_start_index)
            self.insert(new_index, dragged_item)
            self.drag_start_index = new_index
            self.selection_clear(0, 'end')
            self.selection_set(new_index)
            self.activate(new_index)

    def on_drag_end(self, event):
        self.drag_start_index = None

class SimpleListboxWithPreview(ListboxPreviewMixin, Listbox):
    def __init__(self, master, **kwargs):
        Listbox.__init__(self, master, **kwargs)
        if PREVIEW_ENABLED: ListboxPreviewMixin.__init__(self)
        self.paths = []

    def update_list(self, paths: List[Path]):
        self.paths = list(paths)
        self.delete(0, 'end')
        for path in self.paths: self.insert('end', path.name)
        if PREVIEW_ENABLED: self._hide_preview()

class ConfigPopup(ctk.CTkToplevel):
    def __init__(self, master_tab: 'JoinVideoTab', **kwargs):
        super().__init__(master_tab, **kwargs)
        self.master_tab = master_tab
        self.title("Cấu hình JoinVideo")
        self.geometry("800x600")
        self.transient(master_tab)
        self.grab_set()
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)
        PAD_Y = 5; PAD_X = 10
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=0, column=0, padx=PAD_X, pady=PAD_Y, sticky="nsew")
        mode_frame = ctk.CTkFrame(left_frame)
        mode_frame.pack(fill="x", padx=PAD_X, pady=PAD_Y)
        ctk.CTkLabel(mode_frame, text="Chế độ & Hiệu ứng", font=ctk.CTkFont(weight="bold")).pack()
        ctk.CTkRadioButton(mode_frame, text="Ghép ngẫu nhiên các nhóm", variable=self.master_tab.mode_var, value="random", command=self._update_popup_ui_state).pack(anchor="w", padx=PAD_X)
        ctk.CTkRadioButton(mode_frame, text="Ghép nối tiếp cùng nhóm", variable=self.master_tab.mode_var, value="sequential", command=self._update_popup_ui_state).pack(anchor="w", padx=PAD_X)
        self.random_mode_frame = ctk.CTkFrame(mode_frame, fg_color="transparent")
        self.random_mode_frame.pack(fill="x", padx=PAD_X)
        ctk.CTkLabel(self.random_mode_frame, text="Số video ghép (ngẫu nhiên):").pack(anchor="w", pady=(5,0))
        self.num_to_combine_entry = ctk.CTkEntry(self.random_mode_frame, textvariable=self.master_tab.num_to_combine_var)
        self.num_to_combine_entry.pack(fill="x", pady=5)
        ctk.CTkLabel(self.random_mode_frame, text="Hiệu ứng chuyển cảnh:").pack(anchor="w", pady=(5,0))
        self.transition_menu = ctk.CTkOptionMenu(self.random_mode_frame, variable=self.master_tab.transition_var, values=self.master_tab.TRANSITIONS, command=lambda _: self.master_tab.update_summary_text())
        self.transition_menu.pack(fill="x", pady=5)
        audio_frame = ctk.CTkFrame(left_frame)
        audio_frame.pack(fill="x", padx=PAD_X, pady=PAD_Y)
        ctk.CTkLabel(audio_frame, text="Cài đặt Âm thanh", font=ctk.CTkFont(weight="bold")).pack()
        ctk.CTkCheckBox(audio_frame, text="Chèn nhạc nền", variable=self.master_tab.use_bgm_var, command=self._update_popup_ui_state).pack(anchor="w", padx=PAD_X, pady=(5,0))
        self.bgm_list_frame = ctk.CTkScrollableFrame(audio_frame, label_text="Chọn nhạc nền (để trống để chọn ngẫu nhiên)")
        self.bgm_list_frame.pack(fill="both", expand=True, padx=PAD_X, pady=5)
        ctk.CTkButton(audio_frame, text="Làm mới danh sách nhạc", command=self._populate_bgm_menu).pack(fill="x", padx=PAD_X, pady=(0,5))
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, padx=PAD_X, pady=PAD_Y, sticky="nsew")
        visuals_frame = ctk.CTkFrame(right_frame)
        visuals_frame.pack(fill="x", padx=PAD_X, pady=PAD_Y)
        ctk.CTkLabel(visuals_frame, text="Hình ảnh & Tùy chỉnh", font=ctk.CTkFont(weight="bold")).pack()
        ctk.CTkLabel(visuals_frame, text="Tỷ lệ khung hình:").pack(anchor="w", padx=PAD_X, pady=(5,0))
        ar_frame = ctk.CTkFrame(visuals_frame, fg_color="transparent")
        ar_frame.pack(fill="x", padx=PAD_X)
        ctk.CTkRadioButton(ar_frame, text="Dọc (9:16)", variable=self.master_tab.aspect_ratio_var, value="d", command=self._update_popup_ui_state).pack(side="left", padx=5)
        ctk.CTkRadioButton(ar_frame, text="Ngang (16:9)", variable=self.master_tab.aspect_ratio_var, value="n", command=self._update_popup_ui_state).pack(side="left", padx=5)
        ctk.CTkCheckBox(visuals_frame, text="Chèn Watermark (Tên kênh)", variable=self.master_tab.watermark_var, command=self._update_popup_ui_state).pack(anchor="w", padx=PAD_X, pady=(5,0))
        self.wm_text_entry = ctk.CTkEntry(visuals_frame, textvariable=self.master_tab.watermark_text_var, placeholder_text="Nhập tên kênh...")
        self.wm_text_entry.pack(fill="x", padx=PAD_X, pady=5)
        ctk.CTkLabel(visuals_frame, text="Chọn Font:").pack(anchor="w", padx=PAD_X)
        self.font_combo = ctk.CTkComboBox(visuals_frame, variable=self.master_tab.font_name_var, command=self._update_popup_ui_state)
        self.font_combo.pack(fill="x", padx=PAD_X, pady=5)
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=1, column=0, columnspan=2, padx=PAD_X, pady=(10, PAD_Y), sticky="sew")
        button_frame.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(button_frame, text="Lưu Cài Đặt", command=self.save_and_close, height=35).grid(row=0, column=0, padx=(0,5), pady=5, sticky="ew")
        ctk.CTkButton(button_frame, text="Hủy", command=self.destroy, height=35, fg_color="gray").grid(row=0, column=1, padx=(5,0), pady=5, sticky="ew")
        self._populate_font_menu()
        self._populate_bgm_menu()
        self._update_popup_ui_state()

    def _populate_font_menu(self):
        font_files = [f for f in DIR_FONT.glob("*") if f.suffix.lower() in {".ttf", ".otf"}]
        if font_files:
            font_names = sorted([f.name for f in font_files])
            self.master_tab.font_map = {f.name: str(f) for f in font_files}
            self.font_combo.configure(values=font_names)
            saved_font = self.master_tab.font_name_var.get()
            if saved_font in font_names: self.master_tab.font_name_var.set(saved_font)
            elif font_names: self.master_tab.font_name_var.set(font_names[0])
            else: self.master_tab.font_name_var.set("Không có font"); self.font_combo.configure(values=["Không có font"], state="disabled")
        else:
            self.font_combo.configure(values=["Không tìm thấy font"], state="disabled")
            self.master_tab.font_name_var.set("Không tìm thấy font")

    def _populate_bgm_menu(self):
        current_selection = {var.get() for var in self.master_tab.bgm_checkboxes if var.get() != "off"}
        for widget in self.bgm_list_frame.winfo_children(): widget.destroy()
        self.master_tab.bgm_checkboxes.clear()
        bgm_files = sorted([f for f in DIR_BGM.glob("*") if f.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".flac"}])
        if not bgm_files:
            ctk.CTkLabel(self.bgm_list_frame, text="Không có file nhạc trong thư mục Sound_Background").pack()
        else:
            for file_path in bgm_files:
                file_name = file_path.name
                var = ctk.StringVar(value="off")
                if file_name in current_selection or file_name in getattr(self.master_tab, 'saved_bgm_selections', []):
                    var.set(file_name)
                cb = ctk.CTkCheckBox(self.bgm_list_frame, text=file_name, variable=var, onvalue=file_name, offvalue="off", command=self.master_tab.update_summary_text)
                cb.pack(anchor="w", padx=10, pady=2)
                self.master_tab.bgm_checkboxes.append(var)
        if hasattr(self.master_tab, 'saved_bgm_selections'): del self.master_tab.saved_bgm_selections

    def _update_popup_ui_state(self, *_):
        is_random_mode = self.master_tab.mode_var.get() == "random"
        new_state = "normal" if is_random_mode else "disabled"
        self.num_to_combine_entry.configure(state=new_state)
        self.transition_menu.configure(state=new_state)
        use_wm = self.master_tab.watermark_var.get()
        self.wm_text_entry.configure(state="normal" if use_wm else "disabled")
        self.font_combo.configure(state="normal" if use_wm and self.master_tab.font_name_var.get() != "Không tìm thấy font" else "disabled")
        use_bgm = self.master_tab.use_bgm_var.get()
        for widget in self.bgm_list_frame.winfo_children():
            if isinstance(widget, ctk.CTkCheckBox): widget.configure(state="normal" if use_bgm else "disabled")
        self.master_tab.update_summary_text()

    def save_and_close(self):
        settings = self.master_tab.get_settings_from_ui()
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f: json.dump(settings, f, indent=4)
            log_message(f"INFO: Cài đặt JoinVideo đã được lưu vào file '{SETTINGS_FILE}'.")
            self.master_tab.update_summary_text()
            self.destroy()
        except IOError as e:
            log_message(f"ERROR: Không thể lưu file cài đặt JoinVideo: {e}")
            messagebox.showerror("Lỗi Lưu", f"Không thể lưu cài đặt:\n{e}", parent=self)

# ==============================================================================
# PHẦN 3: LỚP FRAME CHO TAB JOINVIDEO
# (Không thay đổi)
# ==============================================================================
class JoinVideoTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(0, weight=1)
        self.processing_thread = None; self.selected_files_paths = []
        self.mode_var = ctk.StringVar(value="random"); self.aspect_ratio_var = ctk.StringVar(value="d")
        self.watermark_var = ctk.BooleanVar(value=False); self.watermark_text_var = ctk.StringVar()
        self.font_name_var = ctk.StringVar(value="Arial"); self.use_bgm_var = ctk.BooleanVar(value=True)
        self.bgm_gain_var = ctk.DoubleVar(value=0.1); self.orig_gain_var = ctk.DoubleVar(value=1.1)
        self.num_to_combine_var = ctk.IntVar(value=3); self.transition_var = ctk.StringVar(value="fade")
        self.transition_duration_var = ctk.DoubleVar(value=1.0); self.input_source_var = ctk.StringVar(value="folder")
        self.bgm_percent_var = ctk.StringVar(value="10%"); self.orig_gain_percent_var = ctk.StringVar(value="+10%")
        self.summary_text_var = ctk.StringVar(); self.font_map = {}; self.bgm_checkboxes = []
        self.TRANSITIONS = ['fade', 'dissolve', 'pixelize', 'wipeleft', 'wiperight', 'wipeup', 'wipedown', 'slideleft', 'slideright', 'slideup', 'slidedown', 'circleopen', 'circleclose', 'horzopen', 'vertopen', 'diagtl', 'radial', 'fadewhite', 'fadeblack', 'none']
        self.config_popup_window = None
        self.create_widgets()
        self.load_settings()
        self.update_ui_state()
        if not FONT_MANAGER_ENABLED: log_message("CẢNH BÁO: Thư viện 'matplotlib' chưa được cài đặt.")
        if not PREVIEW_ENABLED: log_message("CẢNH BÁO: Thư viện 'Pillow' hoặc 'opencv-python' chưa được cài đặt.")

    def create_widgets(self):
        self.grid_columnconfigure((0, 1), weight=1); self.grid_rowconfigure(2, weight=1)
        input_source_frame = ctk.CTkFrame(self)
        input_source_frame.grid(row=0, column=0, padx=10, pady=(10,5), sticky="nsew", rowspan=3)
        input_source_frame.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(input_source_frame, text="Nguồn Video (JoinVideo)", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        ctk.CTkRadioButton(input_source_frame, text=f"Dùng thư mục '{DIR_VIDEO_INPUT.name}'", variable=self.input_source_var, value="folder", command=self.update_ui_state).pack(anchor="w", padx=10, pady=(5,0))
        ctk.CTkRadioButton(input_source_frame, text="Chọn tệp tin thủ công", variable=self.input_source_var, value="files", command=self.update_ui_state).pack(anchor="w", padx=10)
        self.select_files_button = ctk.CTkButton(input_source_frame, text="Chọn Tệp Tin...", command=self.select_files)
        self.select_files_button.pack(fill="x", padx=10, pady=5)
        listbox_frame = ctk.CTkFrame(input_source_frame)
        listbox_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.file_status_listbox = DraggableListbox(listbox_frame, background="#2B2B2B", foreground="white", borderwidth=0, highlightthickness=0, selectbackground="#1F6AA5")
        self.file_status_listbox.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        scrollbar = ctk.CTkScrollbar(listbox_frame, command=self.file_status_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.file_status_listbox.configure(yscrollcommand=scrollbar.set)
        self.input_list_context_menu = tk.Menu(self, tearoff=0, background="#2B2B2B", foreground="white", activebackground="#1F6AA5")
        self.input_list_context_menu.add_command(label="Ghép theo thứ tự đã sắp xếp", command=self.start_sequential_join_from_list)
        self.file_status_listbox.bind("<Button-3>", self.show_input_list_context_menu)
        self.file_status_listbox.bind("<Double-Button-1>", self._open_source_video_on_double_click)
        summary_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray15"))
        summary_frame.grid(row=0, column=1, padx=10, pady=10, sticky="new")
        summary_frame.grid_columnconfigure(0, weight=1)
        summary_top_frame = ctk.CTkFrame(summary_frame, fg_color="transparent")
        summary_top_frame.pack(fill="x", padx=10, pady=(5,2))
        summary_top_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(summary_top_frame, text="Tóm Tắt Cài Đặt (JoinVideo)", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(summary_top_frame, text="Cấu hình...", width=100, command=self.open_config_popup).grid(row=0, column=1, sticky="e")
        ctk.CTkLabel(summary_frame, textvariable=self.summary_text_var, justify="left", anchor="w", font=ctk.CTkFont(size=14)).pack(pady=(2,10), padx=10, fill="x", anchor="w")
        audio_quick_frame = ctk.CTkFrame(self)
        audio_quick_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        audio_quick_frame.grid_columnconfigure((0, 1), weight=1)
        bgm_quick_frame = ctk.CTkFrame(audio_quick_frame)
        bgm_quick_frame.grid(row=0, column=0, padx=(0,5), sticky="ew")
        ctk.CTkLabel(bgm_quick_frame, text="Âm lượng nhạc nền").pack()
        bgm_slider_frame = ctk.CTkFrame(bgm_quick_frame, fg_color="transparent")
        bgm_slider_frame.pack(fill="x", padx=10)
        ctk.CTkSlider(bgm_slider_frame, from_=0, to=1, variable=self.bgm_gain_var, command=self.update_audio_labels).pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(bgm_slider_frame, textvariable=self.bgm_percent_var, width=40).pack(side="right")
        orig_gain_quick_frame = ctk.CTkFrame(audio_quick_frame)
        orig_gain_quick_frame.grid(row=0, column=1, padx=(5,0), sticky="ew")
        ctk.CTkLabel(orig_gain_quick_frame, text="Tăng âm gốc").pack()
        orig_slider_frame = ctk.CTkFrame(orig_gain_quick_frame, fg_color="transparent")
        orig_slider_frame.pack(fill="x", padx=10)
        ctk.CTkSlider(orig_slider_frame, from_=1.0, to=3.0, variable=self.orig_gain_var, command=self.update_audio_labels).pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(orig_slider_frame, textvariable=self.orig_gain_percent_var, width=40).pack(side="right")
        completed_frame = ctk.CTkFrame(self)
        completed_frame.grid(row=2, column=1, padx=10, pady=5, sticky="nsew")
        completed_frame.grid_rowconfigure(1, weight=1)
        completed_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(completed_frame, text="Video đã hoàn thành (JoinVideo)", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=5)
        completed_list_frame = ctk.CTkFrame(completed_frame)
        completed_list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        self.completed_listbox = SimpleListboxWithPreview(completed_list_frame, background="#2B2B2B", foreground="white", borderwidth=0, highlightthickness=0, selectbackground="#1F6AA5")
        self.completed_listbox.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        completed_scrollbar = ctk.CTkScrollbar(completed_list_frame, command=self.completed_listbox.yview)
        completed_scrollbar.pack(side="right", fill="y")
        self.completed_listbox.configure(yscrollcommand=completed_scrollbar.set)
        self.context_menu = tk.Menu(self, tearoff=0, background="#2B2B2B", foreground="white", activebackground="#1F6AA5")
        self.context_menu.add_command(label="▶️ Mở Video", command=self.open_selected_video)
        self.context_menu.add_command(label="📂 Mở Thư mục chứa File", command=self.open_folder_location)
        self.completed_listbox.bind("<Button-3>", self.show_context_menu)
        self.completed_listbox.bind("<Double-Button-1>", lambda e: self.open_selected_video())
        ctk.CTkButton(completed_frame, text="Làm mới", command=self.update_completed_list).grid(row=2, column=0, columnspan=2, padx=10, pady=(5,10), sticky="ew")
        self.log_textbox = ctk.CTkTextbox(self, state="disabled", wrap="word", font=("Courier New", 12), height=120)
        self.log_textbox.grid(row=3, column=0, padx=10, pady=5, sticky="sew")
        self.start_button = ctk.CTkButton(self, text="Bắt đầu xử lý (JoinVideo)", command=self.start_processing, height=40, font=ctk.CTkFont(size=16, weight="bold"))
        self.start_button.grid(row=3, column=1, padx=10, pady=10, sticky="sew")

    def open_config_popup(self):
        if self.config_popup_window is None or not self.config_popup_window.winfo_exists():
            self.config_popup_window = ConfigPopup(master_tab=self)
        else:
            self.config_popup_window.focus()

    def show_input_list_context_menu(self, event):
        if self.input_source_var.get() == "files":
            if not self.file_status_listbox.curselection():
                nearest_idx = self.file_status_listbox.index(f"@{event.x},{event.y}")
                if nearest_idx >= 0: self.file_status_listbox.selection_set(nearest_idx)
            if self.file_status_listbox.curselection(): self.input_list_context_menu.post(event.x_root, event.y_root)

    def show_context_menu(self, event):
        if not self.completed_listbox.curselection():
             nearest_idx = self.completed_listbox.index(f"@{event.x},{event.y}")
             if nearest_idx >= 0: self.completed_listbox.selection_set(nearest_idx)
        if self.completed_listbox.curselection(): self.context_menu.post(event.x_root, event.y_root)

    def update_audio_labels(self, *_):
        self.bgm_percent_var.set(f"{int(self.bgm_gain_var.get() * 100)}%")
        boost_val = (self.orig_gain_var.get() - 1.0)
        self.orig_gain_percent_var.set(f"+{int(boost_val * 100)}%" if boost_val > 0.01 else "0%")
        self.update_summary_text()

    def update_file_status_list(self):
        if self.input_source_var.get() == "folder":
            if not DIR_VIDEO_INPUT.exists(): self.file_status_listbox.update_list([]); return
            files = sort_videos_numerically(list(DIR_VIDEO_INPUT.glob("*.mp4")))
            self.file_status_listbox.update_list(files)
            if not files: self.file_status_listbox.insert('end', "Thư mục Video_Input trống.")
        elif self.input_source_var.get() == "files":
            self.file_status_listbox.update_list(self.selected_files_paths)
            if not self.selected_files_paths: self.file_status_listbox.insert('end', "Chưa có tệp nào được chọn.")

    def update_completed_list(self):
        self.completed_listbox.delete(0, 'end');
        if not DIR_VIDEO_OUTPUT.exists(): return
        files = sorted([f for f in DIR_VIDEO_OUTPUT.glob("*.mp4") if f.is_file()], key=os.path.getmtime, reverse=True)
        if not files: self.completed_listbox.insert('end', "Chưa có video nào được tạo.")
        else: self.completed_listbox.update_list([DIR_VIDEO_OUTPUT / f.name for f in files])

    def update_summary_text(self):
        mode = "Ngẫu nhiên" if self.mode_var.get() == "random" else "Nối tiếp"
        aspect = "Dọc 9:16" if self.aspect_ratio_var.get() == "d" else "Ngang 16:9"
        bgm_status = "Tắt"
        if self.use_bgm_var.get():
            count = sum(1 for var in self.bgm_checkboxes if var.get() != "off")
            bgm_status = f"Bật ({count} tệp, {self.bgm_percent_var.get()})" if count > 0 else f"Bật (Ngẫu nhiên, {self.bgm_percent_var.get()})"
        watermark = f"Watermark: '{self.watermark_text_var.get()}'" if self.watermark_var.get() and self.watermark_text_var.get() else "Watermark: Tắt"
        summary = (f"∙ Chế độ: {mode}\n∙ Tỷ lệ: {aspect}\n∙ Nhạc nền: {bgm_status}\n"
                   f"∙ Âm thanh: Tăng âm gốc: {self.orig_gain_percent_var.get()}\n∙ {watermark}")
        self.summary_text_var.set(summary)

    def update_ui_state(self, *_):
        is_folder_mode = self.input_source_var.get() == "folder"
        if hasattr(self, 'select_files_button'): self.select_files_button.configure(state="disabled" if is_folder_mode else "normal")
        if is_folder_mode: self.selected_files_paths.clear()
        if hasattr(self, 'file_status_listbox'): self.update_file_status_list()
        if hasattr(self, 'completed_listbox'): self.update_completed_list()
        self.update_audio_labels()
        self.update_summary_text()

    def select_files(self):
        filepaths = filedialog.askopenfilenames(title="Chọn các tệp video", filetypes=(("Video files", "*.mp4"),))
        if filepaths: self.selected_files_paths = [Path(p) for p in filepaths]; self.update_file_status_list()

    def _get_selected_video_path(self, listbox_widget):
        selected_indices = listbox_widget.curselection()
        if not selected_indices: log_message("⚠️ Vui lòng chọn một video."); return None
        try:
            if hasattr(listbox_widget, 'paths') and listbox_widget.paths and selected_indices[0] < len(listbox_widget.paths):
                filepath = listbox_widget.paths[selected_indices[0]]
                if not filepath.exists():
                    log_message(f"❌ Lỗi: Không tìm thấy file {filepath}")
                    if listbox_widget == self.completed_listbox: self.update_completed_list()
                    elif listbox_widget == self.file_status_listbox: self.update_file_status_list()
                    return None
                return filepath
            else: log_message("⚠️ Lỗi: Không thể lấy đường dẫn file."); return None
        except IndexError: log_message("⚠️ Lỗi: Index không hợp lệ."); return None

    def _open_action(self, path):
        try:
            if sys.platform == "win32": os.startfile(path)
            elif sys.platform == "darwin": subprocess.call(["open", path])
            else: subprocess.call(["xdg-open", path])
        except Exception as e: log_message(f"❌ Lỗi khi mở: {e}")

    def open_selected_video(self):
        if filepath := self._get_selected_video_path(self.completed_listbox):
            log_message(f"Đang mở video: {filepath}"); self._open_action(filepath)

    def open_folder_location(self):
        if filepath := self._get_selected_video_path(self.completed_listbox):
            log_message(f"Đang mở thư mục: {filepath.parent}"); self._open_action(filepath.parent)

    def _open_source_video_on_double_click(self, event):
        if filepath := self._get_selected_video_path(self.file_status_listbox):
            log_message(f"Đang mở video nguồn: {filepath}"); self._open_action(filepath)

    def start_processing(self, force_mode=None, force_source=None):
        if self.processing_thread and self.processing_thread.is_alive():
            log_message("!!! Đang có một tiến trình chạy. Vui lòng đợi hoàn tất."); return
        video_source = force_source
        if not video_source:
            if self.input_source_var.get() == "folder": video_source = DIR_VIDEO_INPUT
            elif self.input_source_var.get() == "files":
                if not self.file_status_listbox.paths: log_message("⚠️ Lỗi: Chưa chọn tệp nào."); return
                video_source = self.file_status_listbox.paths
            else: log_message("⚠️ Lỗi: Chế độ nguồn không xác định."); return
        settings = self.get_settings_from_ui()
        if self.use_bgm_var.get() and not settings['bgm_paths'] and not pick_random_bgm():
             log_message("⚠️ Lỗi: Không có file nhạc nền."); return
        if settings.get('watermark_text') and (not settings.get('font_path') or not Path(settings['font_path']).exists()):
             log_message(f"⚠️ Lỗi: Watermark được bật nhưng font không hợp lệ."); return
        if force_mode: settings['mode'] = force_mode
        self.log_textbox.configure(state="normal"); self.log_textbox.delete("1.0", "end"); self.log_textbox.configure(state="disabled")
        self.start_button.configure(state="disabled", text="Đang xử lý...")
        self.processing_thread = threading.Thread(target=self.run_worker, args=(settings, video_source), daemon=True)
        self.processing_thread.start()

    def start_sequential_join_from_list(self):
        if len(self.file_status_listbox.paths) < 1:
            log_message("⚠️ Cần ít nhất 1 video trong danh sách."); return
        self.start_processing(force_mode="sequential", force_source=self.file_status_listbox.paths)

    def run_worker(self, settings, video_source):
        processed_files, failed_files, created_files = set(), set(), []
        try:
            mode = settings.get('mode', 'random')
            if mode == 'sequential':
                processed_files, created_files, failed_files = run_mode_sequential(settings, video_source)
            else:
                processed_files, created_files, failed_files = run_mode_random(settings, video_source)
            if self.input_source_var.get() == "folder":
                if processed_files:
                    log_message("\nDi chuyển các file nguồn đã xử lý thành công...")
                    for f_path in processed_files:
                        try:
                            if f_path.exists(): shutil.move(str(f_path), str(DIR_DONE_INPUT / f_path.name))
                        except Exception as e: log_message(f"  - ❌ Lỗi khi di chuyển file {f_path.name}: {e}")
                if failed_files:
                    log_message("\nDi chuyển các file nguồn bị lỗi xử lý...")
                    for f_path in failed_files:
                        try:
                            if f_path.exists(): shutil.move(str(f_path), str(DIR_VIDEO_ERRORS / f_path.name))
                        except Exception as e: log_message(f"  - ❌ Lỗi khi di chuyển file lỗi {f_path.name}: {e}")
            log_message("\n🎉 Xử lý hoàn tất!")
        except Exception as e:
            log_message(f"\n❌ LỖI KHÔNG MONG MUỐN XẢY RA: {e}")
            import traceback; log_message(traceback.format_exc())
        finally:
            self.after(0, self.on_processing_finished, created_files)

    def on_processing_finished(self, created_files):
        self.start_button.configure(state="normal", text="Bắt đầu xử lý (JoinVideo)")
        self.update_file_status_list()
        self.update_completed_list()
        if created_files:
            latest_file = created_files[-1]
            try:
                items = [p.name for p in self.completed_listbox.paths]
                if latest_file.name in items:
                    idx = items.index(latest_file.name)
                    self.completed_listbox.selection_clear(0, "end"); self.completed_listbox.selection_set(idx)
                    self.completed_listbox.activate(idx); self.completed_listbox.see(idx)
            except (ValueError, IndexError): pass

    def get_settings_from_ui(self) -> dict:
        selected_bgm_paths = [str(DIR_BGM / var.get()) for var in self.bgm_checkboxes if var.get() != "off"]
        font_path = self.font_map.get(self.font_name_var.get(), "")
        return {
            'mode': self.mode_var.get(), 'aspect_ratio_choice': self.aspect_ratio_var.get(),
            'watermark_text': self.watermark_text_var.get().strip() if self.watermark_var.get() else "",
            'font_path': font_path, 'font_name': self.font_name_var.get(),
            'transition': self.transition_var.get(), 'transition_duration': self.transition_duration_var.get(),
            'use_bgm': self.use_bgm_var.get(), 'bgm_paths': selected_bgm_paths, 'bgm_gain': self.bgm_gain_var.get(),
            'orig_gain': self.orig_gain_var.get(), 'num_to_combine': self.num_to_combine_var.get()
        }

    def load_settings(self):
        settings_path = Path(SETTINGS_FILE)
        if not settings_path.exists(): return
        try:
            with open(settings_path, "r", encoding="utf-8") as f: settings = json.load(f)
            self.mode_var.set(settings.get('mode', 'random'))
            self.aspect_ratio_var.set(settings.get('aspect_ratio_choice', 'd'))
            wm_text = settings.get('watermark_text', "")
            self.watermark_var.set(bool(wm_text)); self.watermark_text_var.set(wm_text)
            self.font_name_var.set(settings.get('font_name', "Arial"))
            self.transition_var.set(settings.get('transition', 'fade'))
            self.transition_duration_var.set(settings.get('transition_duration', 1.0))
            self.use_bgm_var.set(settings.get('use_bgm', True))
            self.saved_bgm_selections = [os.path.basename(p) for p in settings.get('bgm_paths', [])]
            self.bgm_gain_var.set(settings.get('bgm_gain', 0.1))
            self.orig_gain_var.set(settings.get('orig_gain', 1.1))
            self.num_to_combine_var.set(settings.get('num_to_combine', 3))
            log_message("INFO: Đã tải cài đặt JoinVideo từ lần chạy trước.")
        except (json.JSONDecodeError, IOError, KeyError, ValueError) as e:
            log_message(f"ERROR: Không thể tải file cài đặt. Lỗi: {e}")
        finally:
            self.update_ui_state()