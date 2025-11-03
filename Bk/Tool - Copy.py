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
from tkinter import filedialog, Listbox

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
SETTINGS_FILE  = "last_settings.json"

# ----- THƯ MỤC -----
BASE = Path.cwd()
DIR_VIDEO_INPUT      = BASE / "Video_Input"
DIR_BGM              = BASE / "Sound_Background"
DIR_TEMP_OUTPUT      = BASE / "Temp_Output"
DIR_DONE_INPUT       = BASE / "DONE" / "Video_Input"
DIR_VIDEO_OUTPUT     = BASE / "Video_Output"
DIR_VIDEO_ERRORS     = BASE / "Video_Errors"

for d in [DIR_VIDEO_INPUT, DIR_BGM, DIR_TEMP_OUTPUT, DIR_DONE_INPUT, DIR_VIDEO_OUTPUT, DIR_VIDEO_ERRORS]:
    d.mkdir(parents=True, exist_ok=True)

# ----- HELPERS -----
log_queue = queue.Queue()

def log_message(msg: str):
    log_queue.put(msg)

def run(cmd: List[str]):
    cmd_str_list = [str(c) for c in cmd]
    log_message("RUNNING: " + " ".join(cmd_str_list))
    process = subprocess.Popen(cmd_str_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
    for line in iter(process.stdout.readline, ''):
        log_message(line.strip())
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, "FFmpeg command failed.")

def ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path)
        ], text=True, stderr=subprocess.PIPE)
        duration = float(out.strip())
        return duration if duration > 0 else 0.0
    except Exception:
        return 0.0

def pick_random_bgm() -> Optional[Path]:
    cands = [p for p in DIR_BGM.glob("*") if p.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".flac"}]
    return random.choice(cands) if cands else None

def group_videos_by_prefix(video_source: Union[List[Path], Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    pat = re.compile(r"^(.+?_)\d+\.mp4$", re.IGNORECASE)
    
    files_to_scan = []
    if isinstance(video_source, Path):
        files_to_scan = sorted(video_source.glob("*.mp4"))
    elif isinstance(video_source, list):
        files_to_scan = video_source

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
        return int(match.group(1)) if match else 0
    return sorted(video_paths, key=get_number)

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
        if duration1 == 0: raise RuntimeError(f"Không thể lấy thời lượng file tạm.")
        effective_transition = transition
        if effective_transition != 'none':
            offset = duration1 - transition_duration
            filter_complex = (f"[0:v][1:v]xfade=transition={effective_transition}:duration={transition_duration}:offset={offset},format={PIX_FMT}[v];"
                              f"[0:a][1:a]acrossfade=d={transition_duration}[a]")
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(segment1), "-i", str(segment2), "-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]", "-c:v", VIDEO_CODEC, "-preset", PRESET, "-crf", CRF, "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, str(out_path)]
        else:
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

def concat_sequential_videos(video_list: List[Path], out_path: Path) -> bool:
    log_message(f"  - Đang ghép nối tiếp {len(video_list)} video...")
    list_file = out_path.with_suffix(".txt")
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for video in video_list: f.write(f"file '{video.resolve()}'\n")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_path)]
        run(cmd)
        return True
    except Exception as e:
        log_message(f"  - ❌ LỖI khi ghép nối tiếp: {e}")
        return False
    finally:
        if list_file.exists(): list_file.unlink()

def add_watermark_cmd(in_path: Path, out_path: Path, watermark_text: str, font_path: str):
    font_path_escaped = font_path.replace("\\", "/").replace(":", "\\:")
    watermark_filter = (f"drawtext=text='{watermark_text}':fontfile='{font_path_escaped}':fontsize=60:fontcolor=white@0.3:x='(w-tw)/2+(w-tw)/2*sin(t/5)':y='(h-th)/3+(h-th)/4*cos(t/7)'")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(in_path), "-vf", watermark_filter, "-c:v", VIDEO_CODEC, "-preset", PRESET, "-crf", CRF, "-c:a", "copy", str(out_path)]
    run(cmd)

def mix_bgm_cmd(video_path: Path, final_path: Path, bgm_path: Path, bgm_gain: float):
    video_duration = ffprobe_duration(video_path)
    if video_duration == 0: raise RuntimeError(f"Could not get final video duration for {video_path.name}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path)]
    if LOOP_BGM: cmd += ["-stream_loop", "-1", "-i", str(bgm_path)]
    else: cmd += ["-i", str(bgm_path)]
    filter_complex = (f"[0:a]aresample=async=1:first_pts=0[a0];[1:a]aresample=async=1:first_pts=0,volume={bgm_gain:.2f}[bg];[a0][bg]amix=inputs=2:duration=longest:normalize=0[aout]")
    cmd += ["-filter_complex", filter_complex, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, "-t", f"{video_duration:.3f}", str(final_path)]
    run(cmd)

def run_mode_random(settings: dict, video_source: Union[List[Path], Path]):
    log_message("\n🚀 Bắt đầu xử lý hàng loạt (Chế độ Ngẫu nhiên)...")
    video_groups = group_videos_by_prefix(video_source)
    group_keys = list(video_groups.keys())
    num_to_combine = settings.get('num_to_combine', 2)
    if len(group_keys) < num_to_combine:
        log_message(f"⚠️ Không đủ nhóm video. Cần {num_to_combine} nhóm, chỉ có {len(group_keys)}.")
        return set(), []
    random.shuffle(group_keys)
    batch_index = 1
    processed_files_in_run = set()
    created_files = []
    
    while len(group_keys) >= num_to_combine:
        log_message(f"\n--- 🔄 Đang xử lý lô #{batch_index} ---")
        selected_keys = [group_keys.pop(0) for _ in range(num_to_combine)]
        temp_files_to_delete: List[Path] = []
        try:
            arc = settings.get('aspect_ratio_choice')
            target_w, target_h = (1080, 1920) if arc == 'd' else (1920, 1080)
            orig_gain = settings.get('orig_gain', 1.0)
            videos_to_process_orig = [random.choice(video_groups[key]) for key in selected_keys]
            for p in videos_to_process_orig: processed_files_in_run.add(p)

            log_message("Bước 1: Chuẩn hóa các video đầu vào...")
            clip_duration = ffprobe_duration(videos_to_process_orig[0])
            if clip_duration == 0:
                log_message(f"  - ❌ LỖI: Không đo được thời lượng file chuẩn {videos_to_process_orig[0].name}. Bỏ qua lô này.")
                for video_path in videos_to_process_orig: shutil.move(str(video_path), str(DIR_VIDEO_ERRORS / video_path.name))
                continue
            log_message(f"  - Thời lượng chuẩn cho lô này là: {clip_duration:.2f} giây.")
            normalized_videos, failed_videos = [], []
            for video_path in videos_to_process_orig:
                norm_path = DIR_TEMP_OUTPUT / f"NORM_{batch_index}_{video_path.name}"
                temp_files_to_delete.append(norm_path)
                if normalize_video(video_path, norm_path, target_w, target_h, orig_gain, clip_duration):
                    normalized_videos.append(norm_path)
                else:
                    failed_videos.append(video_path)
            for video_path in failed_videos:
                shutil.move(str(video_path), str(DIR_VIDEO_ERRORS / video_path.name))
                log_message(f"  - Đã di chuyển file lỗi '{video_path.name}' vào thư mục '{DIR_VIDEO_ERRORS.name}'.")
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
                    else: next_level_videos.append(current_level_videos[i])
                current_level_videos = next_level_videos
                level += 1
            final_concatenated_video = current_level_videos[0]
            current_processed_file = final_concatenated_video
            if settings.get('watermark_text'):
                log_message("Bước 3: Thêm watermark...")
                watermarked_file = DIR_TEMP_OUTPUT / f"TEMP_WM_{batch_index}.mp4"
                temp_files_to_delete.append(watermarked_file)
                add_watermark_cmd(current_processed_file, watermarked_file, settings.get('watermark_text', ""), settings.get('font_path', "C:/Windows/Fonts/Arial.ttf"))
                current_processed_file = watermarked_file
            output_name_parts = [Path(p).stem.split('_')[0] for p in videos_to_process_orig if p not in failed_videos]
            output_filename = "_".join(output_name_parts) + ".mp4"
            final_output_path = DIR_VIDEO_OUTPUT / output_filename
            
            if settings.get('use_bgm'):
                log_message("Bước 4: Trộn nhạc nền...")
                bgm_path = settings.get('bgm_path') if settings.get('bgm_mode') == 'single' else pick_random_bgm()
                if not bgm_path:
                    log_message("⚠️ Không tìm thấy BGM.")
                    shutil.move(str(current_processed_file), str(final_output_path))
                else:
                    log_message(f"🔊 Đã chọn BGM: {Path(bgm_path).name}")
                    mix_bgm_cmd(current_processed_file, final_output_path, bgm_path, settings.get('bgm_gain', 0.1))
            else: shutil.move(str(current_processed_file), str(final_output_path))
            
            created_files.append(final_output_path)
            log_message(f"✅ Hoàn thành lô #{batch_index}: {final_output_path.name}")
        except Exception as e:
            log_message(f"❌ LỖI khi xử lý lô #{batch_index}: {e}")
            log_message("  - Các file nguồn cho lô này sẽ KHÔNG bị di chuyển.")
        finally:
            log_message("  - Dọn dẹp file tạm...")
            for temp_file in temp_files_to_delete: temp_file.unlink(missing_ok=True)
        batch_index += 1
    if group_keys:
        log_message(f"\nℹ️ Còn dư {len(group_keys)} nhóm video, không đủ để tạo lô tiếp theo.")
    return processed_files_in_run, created_files

def run_mode_sequential(settings: dict, video_source: Union[List[Path], Path]):
    log_message("\n🚀 Bắt đầu xử lý hàng loạt (Chế độ Nối tiếp)...")
    video_groups = group_videos_by_prefix(video_source)
    if not video_groups:
        log_message("⚠️ Không tìm thấy nhóm video nào để xử lý.")
        return set(), []
    batch_index = 1
    processed_files_in_run = set()
    created_files = []

    for prefix, video_list in video_groups.items():
        if len(video_list) < 2:
            log_message(f"\n--- ℹ️  Bỏ qua nhóm '{prefix.strip('_')}' vì chỉ có 1 video ---")
            continue
        log_message(f"\n--- 🔄 Đang xử lý nhóm '{prefix.strip('_')}' (lô #{batch_index}) ---")
        temp_files_to_delete: List[Path] = []
        try:
            arc = settings.get('aspect_ratio_choice')
            target_w, target_h = (1080, 1920) if arc == 'd' else (1920, 1080)
            orig_gain = settings.get('orig_gain', 1.0)
            
            videos_to_process_orig = video_list if isinstance(video_source, list) else sort_videos_numerically(video_list)
            
            for p in videos_to_process_orig: processed_files_in_run.add(p)

            log_message(f"  - Phát hiện {len(videos_to_process_orig)} video. Thứ tự ghép:")
            for i, p in enumerate(videos_to_process_orig): log_message(f"    {i+1}. {p.name}")
            log_message("Bước 1: Chuẩn hóa các video đầu vào...")
            normalized_videos, failed_videos = [], []
            for video_path in videos_to_process_orig:
                norm_path = DIR_TEMP_OUTPUT / f"NORM_{batch_index}_{video_path.name}"
                temp_files_to_delete.append(norm_path)
                if normalize_video(video_path, norm_path, target_w, target_h, orig_gain, clip_duration=None):
                    normalized_videos.append(norm_path)
                else:
                    failed_videos.append(video_path)
            for video_path in failed_videos:
                shutil.move(str(video_path), str(DIR_VIDEO_ERRORS / video_path.name))
                log_message(f"  - Đã di chuyển file lỗi '{video_path.name}' vào thư mục '{DIR_VIDEO_ERRORS.name}'.")
            if len(normalized_videos) < 2:
                log_message(f"  - ⚠️ Không đủ video hợp lệ để ghép. Bỏ qua nhóm này.")
                continue
            log_message("Bước 2: Ghép nối tiếp...")
            concatenated_file = DIR_TEMP_OUTPUT / f"TEMP_CONCAT_{batch_index}.mp4"
            temp_files_to_delete.append(concatenated_file)
            if not concat_sequential_videos(normalized_videos, concatenated_file):
                raise Exception("Ghép nối tiếp thất bại.")
            current_processed_file = concatenated_file
            if settings.get('watermark_text'):
                log_message("Bước 3: Thêm watermark...")
                watermarked_file = DIR_TEMP_OUTPUT / f"TEMP_WM_{batch_index}.mp4"
                temp_files_to_delete.append(watermarked_file)
                add_watermark_cmd(current_processed_file, watermarked_file, settings.get('watermark_text', ""), settings.get('font_path', "C:/Windows/Fonts/Arial.ttf"))
                current_processed_file = watermarked_file
            output_filename = f"{prefix.strip('_')}_final.mp4"
            final_output_path = DIR_VIDEO_OUTPUT / output_filename
            if settings.get('use_bgm'):
                log_message("Bước 4: Trộn nhạc nền...")
                bgm_path = settings.get('bgm_path') if settings.get('bgm_mode') == 'single' else pick_random_bgm()
                if not bgm_path:
                    log_message("⚠️ Không tìm thấy BGM.")
                    shutil.move(str(current_processed_file), str(final_output_path))
                else:
                    log_message(f"🔊 Đã chọn BGM: {Path(bgm_path).name}")
                    mix_bgm_cmd(current_processed_file, final_output_path, bgm_path, settings.get('bgm_gain', 0.1))
            else: shutil.move(str(current_processed_file), str(final_output_path))
            
            created_files.append(final_output_path)
            log_message(f"✅ Hoàn thành nhóm '{prefix.strip('_')}': {final_output_path.name}")
        except Exception as e:
            log_message(f"❌ LỖI khi xử lý nhóm '{prefix.strip('_')}': {e}")
            log_message("  - Các file nguồn cho nhóm này sẽ KHÔNG bị di chuyển.")
        finally:
            log_message("  - Dọn dẹp file tạm...")
            for temp_file in temp_files_to_delete: temp_file.unlink(missing_ok=True)
        batch_index += 1
    return processed_files_in_run, created_files

# ==============================================================================
# PHẦN 2: GIAO DIỆN NGƯỜI DÙNG (UI)
# ==============================================================================
class DraggableListbox(Listbox):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.bind("<Button-1>", self.on_drag_start)
        self.bind("<B1-Motion>", self.on_drag_motion)
        self.drag_start_index = None
        self.paths = []

    def update_list(self, paths: List[Path]):
        self.paths = list(paths)
        self.delete(0, 'end')
        for path in self.paths:
            self.insert('end', path.name)

    def on_drag_start(self, event):
        self.drag_start_index = self.nearest(event.y)

    def on_drag_motion(self, event):
        if self.drag_start_index is None:
            return
        
        new_index = self.nearest(event.y)
        if new_index != self.drag_start_index:
            dragged_path = self.paths.pop(self.drag_start_index)
            self.paths.insert(new_index, dragged_path)
            
            dragged_item = self.get(self.drag_start_index)
            self.delete(self.drag_start_index)
            self.insert(new_index, dragged_item)
            
            self.drag_start_index = new_index
            self.selection_clear(0, 'end')
            self.selection_set(new_index)
            self.activate(new_index)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Video Joiner Tool")
        self.geometry("1200x800")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.processing_thread = None
        self.selected_files_paths = []
        
        self.mode_var = ctk.StringVar(value="random")
        self.aspect_ratio_var = ctk.StringVar(value="d")
        self.watermark_var = ctk.BooleanVar(value=False)
        self.watermark_text_var = ctk.StringVar()
        self.font_path_var = ctk.StringVar(value="C:/Windows/Fonts/Arial.ttf")
        self.use_bgm_var = ctk.BooleanVar(value=True)
        self.bgm_mode_var = ctk.StringVar(value="random")
        self.bgm_path_var = ctk.StringVar(value="")
        self.boost_audio_var = ctk.BooleanVar(value=False)
        self.bgm_gain_var = ctk.DoubleVar(value=0.1)
        self.orig_gain_var = ctk.DoubleVar(value=1.1)
        self.num_to_combine_var = ctk.IntVar(value=3)
        self.transition_var = ctk.StringVar(value="fade")
        self.transition_duration_var = ctk.DoubleVar(value=1.0)
        self.input_source_var = ctk.StringVar(value="folder")

        self.bgm_percent_var = ctk.StringVar(value="10%")
        self.orig_gain_percent_var = ctk.StringVar(value="+10%")
        self.bgm_filename_var = ctk.StringVar(value="Ngẫu nhiên")
        self.summary_text_var = ctk.StringVar()

        self.TRANSITIONS = ['fade', 'dissolve', 'pixelize', 'wipeleft', 'wiperight', 'wipeup', 'wipedown', 'slideleft', 'slideright', 'slideup', 'slidedown', 'circleopen', 'circleclose', 'horzopen', 'vertopen', 'diagtl', 'radial', 'fadewhite', 'fadeblack', 'none']

        self.create_widgets()
        self.load_settings()
        self.update_ui_state()

    def create_widgets(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.tab_view = ctk.CTkTabview(self, anchor="w")
        self.tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.tab_view.add("Main")
        self.tab_view.add("Settings")
        self.tab_view.set("Main")

        self.create_main_tab()
        self.create_settings_tab()

    def create_main_tab(self):
        tab_main = self.tab_view.tab("Main")
        tab_main.grid_columnconfigure(0, weight=1)
        tab_main.grid_columnconfigure(1, weight=1)
        tab_main.grid_rowconfigure(2, weight=1) # Log row expands

        input_source_frame = ctk.CTkFrame(tab_main)
        input_source_frame.grid(row=0, column=0, padx=10, pady=(10,5), sticky="nsew", rowspan=4)
        input_source_frame.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(input_source_frame, text="Nguồn Video", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        ctk.CTkRadioButton(input_source_frame, text="Dùng thư mục Video_Input", variable=self.input_source_var, value="folder", command=self.update_ui_state).pack(anchor="w", padx=10, pady=(5,0))
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

        summary_frame = ctk.CTkFrame(tab_main, fg_color=("gray85", "gray15"))
        summary_frame.grid(row=0, column=1, padx=10, pady=10, sticky="new")
        ctk.CTkLabel(summary_frame, text="Tóm Tắt Cài Đặt", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(5,2), padx=10)
        ctk.CTkLabel(summary_frame, textvariable=self.summary_text_var, justify="left", anchor="w", font=ctk.CTkFont(size=14)).pack(pady=(2,10), padx=10, fill="x")
        
        audio_quick_frame = ctk.CTkFrame(tab_main)
        audio_quick_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        audio_quick_frame.grid_columnconfigure((0, 1), weight=1)
        
        bgm_quick_frame = ctk.CTkFrame(audio_quick_frame)
        bgm_quick_frame.grid(row=0, column=0, padx=(0,5), sticky="ew")
        ctk.CTkLabel(bgm_quick_frame, text="Âm lượng nhạc nền").pack()
        bgm_slider_frame = ctk.CTkFrame(bgm_quick_frame, fg_color="transparent")
        bgm_slider_frame.pack(fill="x", padx=10)
        self.bgm_slider_main = ctk.CTkSlider(bgm_slider_frame, from_=0, to=1, variable=self.bgm_gain_var, command=self.update_audio_labels)
        self.bgm_slider_main.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(bgm_slider_frame, textvariable=self.bgm_percent_var, width=40).pack(side="right")
        
        orig_gain_quick_frame = ctk.CTkFrame(audio_quick_frame)
        orig_gain_quick_frame.grid(row=0, column=1, padx=(5,0), sticky="ew")
        ctk.CTkLabel(orig_gain_quick_frame, text="Tăng âm gốc").pack()
        orig_slider_frame = ctk.CTkFrame(orig_gain_quick_frame, fg_color="transparent")
        orig_slider_frame.pack(fill="x", padx=10)
        self.orig_gain_slider_main = ctk.CTkSlider(orig_slider_frame, from_=1.0, to=3.0, variable=self.orig_gain_var, command=self.update_audio_labels)
        self.orig_gain_slider_main.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(orig_slider_frame, textvariable=self.orig_gain_percent_var, width=40).pack(side="right")

        self.log_textbox = ctk.CTkTextbox(tab_main, state="disabled", wrap="word", font=("Courier New", 12))
        self.log_textbox.grid(row=2, column=1, padx=10, pady=5, sticky="nsew")
        self.after(100, self.process_log_queue)

        self.start_button = ctk.CTkButton(tab_main, text="Bắt đầu xử lý với cài đặt đã chọn", command=self.start_processing, height=40, font=ctk.CTkFont(size=16, weight="bold"))
        self.start_button.grid(row=3, column=1, padx=10, pady=10, sticky="sew")
        
        completed_frame = ctk.CTkFrame(tab_main)
        completed_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        completed_frame.grid_rowconfigure(1, weight=1)
        completed_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(completed_frame, text="Video đã hoàn thành", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=5)
        
        completed_list_frame = ctk.CTkFrame(completed_frame)
        completed_list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        self.completed_listbox = Listbox(completed_list_frame, background="#2B2B2B", foreground="white", borderwidth=0, highlightthickness=0, selectbackground="#1F6AA5")
        self.completed_listbox.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        completed_scrollbar = ctk.CTkScrollbar(completed_list_frame, command=self.completed_listbox.yview)
        completed_scrollbar.pack(side="right", fill="y")
        self.completed_listbox.configure(yscrollcommand=completed_scrollbar.set)
        
        self.context_menu = tk.Menu(self, tearoff=0, background="#2B2B2B", foreground="white", activebackground="#1F6AA5")
        self.context_menu.add_command(label="▶️ Mở Video", command=self.open_selected_video)
        self.context_menu.add_command(label="📂 Mở Thư mục chứa File", command=self.open_folder_location)
        self.completed_listbox.bind("<Button-3>", self.show_context_menu)
        
        self.refresh_completed_button = ctk.CTkButton(completed_frame, text="Làm mới", command=self.update_completed_list)
        self.refresh_completed_button.grid(row=2, column=0, columnspan=2, padx=10, pady=(5,10), sticky="ew")

    def create_settings_tab(self):
        tab_settings = self.tab_view.tab("Settings")
        tab_settings.grid_columnconfigure((0, 1), weight=1)

        PAD_Y = 5
        PAD_X = 10

        left_frame = ctk.CTkFrame(tab_settings)
        left_frame.grid(row=0, column=0, padx=PAD_X, pady=PAD_Y, sticky="new")

        mode_frame = ctk.CTkFrame(left_frame)
        mode_frame.pack(fill="x", padx=PAD_X, pady=PAD_Y)
        ctk.CTkLabel(mode_frame, text="Chế độ & Hiệu ứng", font=ctk.CTkFont(weight="bold")).pack()
        ctk.CTkRadioButton(mode_frame, text="Ghép ngẫu nhiên các nhóm", variable=self.mode_var, value="random", command=self.update_ui_state).pack(anchor="w", padx=PAD_X)
        ctk.CTkRadioButton(mode_frame, text="Ghép nối tiếp cùng nhóm", variable=self.mode_var, value="sequential", command=self.update_ui_state).pack(anchor="w", padx=PAD_X)
        
        self.random_mode_frame = ctk.CTkFrame(mode_frame, fg_color="transparent")
        self.random_mode_frame.pack(fill="x", padx=PAD_X)
        ctk.CTkLabel(self.random_mode_frame, text="Số video ghép (ngẫu nhiên):").pack(anchor="w", pady=(5,0))
        self.num_to_combine_entry = ctk.CTkEntry(self.random_mode_frame, textvariable=self.num_to_combine_var)
        self.num_to_combine_entry.pack(fill="x", pady=5)
        ctk.CTkLabel(self.random_mode_frame, text="Hiệu ứng chuyển cảnh:").pack(anchor="w", pady=(5,0))
        self.transition_menu = ctk.CTkOptionMenu(self.random_mode_frame, variable=self.transition_var, values=self.TRANSITIONS, command=lambda _: self.update_summary_text())
        self.transition_menu.pack(fill="x", pady=5)

        audio_frame = ctk.CTkFrame(left_frame)
        audio_frame.pack(fill="x", padx=PAD_X, pady=PAD_Y)
        ctk.CTkLabel(audio_frame, text="Cài đặt Âm thanh", font=ctk.CTkFont(weight="bold")).pack()
        
        ctk.CTkCheckBox(audio_frame, text="Chèn nhạc nền", variable=self.use_bgm_var, command=self.update_ui_state).pack(anchor="w", padx=PAD_X, pady=(5,0))
        self.bgm_options_frame = ctk.CTkFrame(audio_frame, fg_color="transparent")
        self.bgm_options_frame.pack(fill="x", padx=PAD_X)
        ctk.CTkRadioButton(self.bgm_options_frame, text="Ngẫu nhiên", variable=self.bgm_mode_var, value="random", command=self.update_ui_state).pack(side="left", padx=5)
        ctk.CTkRadioButton(self.bgm_options_frame, text="Chọn tệp", variable=self.bgm_mode_var, value="single", command=self.update_ui_state).pack(side="left", padx=5)
        self.select_bgm_button = ctk.CTkButton(self.bgm_options_frame, text="...", width=30, command=self.select_bgm_file)
        self.select_bgm_button.pack(side="left", padx=5)
        self.bgm_filename_label = ctk.CTkLabel(self.bgm_options_frame, textvariable=self.bgm_filename_var, text_color="gray", wraplength=150)
        self.bgm_filename_label.pack(side="left", padx=5)
        
        ctk.CTkCheckBox(audio_frame, text="Tăng âm thanh gốc", variable=self.boost_audio_var, command=self.update_ui_state).pack(anchor="w", padx=PAD_X, pady=(5,0))

        right_frame = ctk.CTkFrame(tab_settings)
        right_frame.grid(row=0, column=1, padx=PAD_X, pady=PAD_Y, sticky="new")
        
        visuals_frame = ctk.CTkFrame(right_frame)
        visuals_frame.pack(fill="x", padx=PAD_X, pady=PAD_Y)
        ctk.CTkLabel(visuals_frame, text="Hình ảnh & Tùy chỉnh", font=ctk.CTkFont(weight="bold")).pack()

        ctk.CTkLabel(visuals_frame, text="Tỷ lệ khung hình:").pack(anchor="w", padx=PAD_X, pady=(5,0))
        ar_frame = ctk.CTkFrame(visuals_frame, fg_color="transparent")
        ar_frame.pack(fill="x", padx=PAD_X)
        ctk.CTkRadioButton(ar_frame, text="Dọc (9:16)", variable=self.aspect_ratio_var, value="d", command=self.update_ui_state).pack(side="left", padx=5)
        ctk.CTkRadioButton(ar_frame, text="Ngang (16:9)", variable=self.aspect_ratio_var, value="n", command=self.update_ui_state).pack(side="left", padx=5)

        ctk.CTkCheckBox(visuals_frame, text="Chèn Watermark (Tên kênh)", variable=self.watermark_var, command=self.update_ui_state).pack(anchor="w", padx=PAD_X, pady=(5,0))
        self.wm_text_entry = ctk.CTkEntry(visuals_frame, textvariable=self.watermark_text_var, placeholder_text="Nhập tên kênh...")
        self.wm_text_entry.pack(fill="x", padx=PAD_X, pady=5)
        font_frame = ctk.CTkFrame(visuals_frame, fg_color="transparent")
        font_frame.pack(fill="x", padx=PAD_X, pady=5)
        ctk.CTkButton(font_frame, text="Chọn Font", command=self.select_font).pack(side="left")
        self.font_label = ctk.CTkLabel(font_frame, textvariable=self.font_path_var, wraplength=180, justify="left", text_color="gray")
        self.font_label.pack(side="left", padx=5)

    def show_input_list_context_menu(self, event):
        selection = self.file_status_listbox.curselection()
        if not selection:
            self.file_status_listbox.selection_set(self.file_status_listbox.nearest(event.y))
        
        if self.file_status_listbox.curselection():
            self.input_list_context_menu.post(event.x_root, event.y_root)

    def show_context_menu(self, event):
        selection = self.completed_listbox.curselection()
        if not selection:
            self.completed_listbox.selection_set(self.completed_listbox.nearest(event.y))
        
        if self.completed_listbox.curselection():
            self.context_menu.post(event.x_root, event.y_root)

    def update_audio_labels(self, *_):
        self.bgm_percent_var.set(f"{int(self.bgm_gain_var.get() * 100)}%")
        if self.boost_audio_var.get():
            boost_val = (self.orig_gain_var.get() - 1.0)
            self.orig_gain_percent_var.set(f"+{int(boost_val * 100)}%")
        else:
            self.orig_gain_percent_var.set("0%")
        self.update_summary_text()

    def update_file_status_list(self):
        if self.input_source_var.get() == "folder":
            if not DIR_VIDEO_INPUT.exists():
                self.file_status_listbox.update_list([])
                self.file_status_listbox.insert('end', f"Thư mục '{DIR_VIDEO_INPUT.name}' không tồn tại.")
                return
            
            files = sort_videos_numerically(list(DIR_VIDEO_INPUT.glob("*.mp4")))
            self.file_status_listbox.update_list(files)
            if not files: self.file_status_listbox.insert('end', "Thư mục Video_Input trống.")
        
        elif self.input_source_var.get() == "files":
            self.file_status_listbox.update_list(self.selected_files_paths)
            if not self.selected_files_paths: self.file_status_listbox.insert('end', "Chưa có tệp nào được chọn.")
    
    def update_completed_list(self):
        self.completed_listbox.delete(0, 'end')
        if not DIR_VIDEO_OUTPUT.exists(): return
        
        files = sorted([f for f in DIR_VIDEO_OUTPUT.glob("*.mp4") if f.is_file()], key=os.path.getmtime, reverse=True)
        if not files: self.completed_listbox.insert('end', "Chưa có video nào được tạo.")
        else:
            for f in files: self.completed_listbox.insert('end', f.name)
            
    def update_summary_text(self):
        mode = "Ngẫu nhiên" if self.mode_var.get() == "random" else "Nối tiếp"
        aspect = "Dọc 9:16" if self.aspect_ratio_var.get() == "d" else "Ngang 16:9"
        
        bgm_status = "Tắt"
        if self.use_bgm_var.get():
            bgm_mode = "Ngẫu nhiên" if self.bgm_mode_var.get() == 'random' else f"Tệp ({self.bgm_filename_var.get()})"
            bgm_status = f"Bật ({bgm_mode})"

        audio_boost = f"Tăng âm gốc: {self.orig_gain_percent_var.get()}" if self.boost_audio_var.get() else "Tăng âm gốc: Tắt"
        watermark = f"Watermark: '{self.watermark_text_var.get()}'" if self.watermark_var.get() and self.watermark_text_var.get() else "Watermark: Tắt"
        
        summary = (
            f"∙ Chế độ: {mode}\n"
            f"∙ Tỷ lệ: {aspect}\n"
            f"∙ Nhạc nền: {bgm_status}\n"
            f"∙ Âm thanh: {audio_boost}\n"
            f"∙ {watermark}"
        )
        self.summary_text_var.set(summary)

    def update_ui_state(self, *_):
        is_random_mode = self.mode_var.get() == "random"
        new_state = "normal" if is_random_mode else "disabled"
        self.num_to_combine_entry.configure(state=new_state)
        self.transition_menu.configure(state=new_state)
        
        use_wm = self.watermark_var.get()
        self.wm_text_entry.configure(state="normal" if use_wm else "disabled")
        
        use_bgm = self.use_bgm_var.get()
        for widget in self.bgm_options_frame.winfo_children(): widget.configure(state="normal" if use_bgm else "disabled")
        self.bgm_slider_main.configure(state="normal" if use_bgm else "disabled")
        
        is_bgm_single_mode = self.bgm_mode_var.get() == "single"
        self.select_bgm_button.configure(state="normal" if use_bgm and is_bgm_single_mode else "disabled")

        if self.bgm_mode_var.get() == 'random':
            self.bgm_filename_var.set("Ngẫu nhiên")
        else:
            if self.bgm_path_var.get():
                self.bgm_filename_var.set(os.path.basename(self.bgm_path_var.get()))
            else:
                self.bgm_filename_var.set("Chưa chọn file")

        boost_audio = self.boost_audio_var.get()
        self.orig_gain_slider_main.configure(state="normal" if boost_audio else "disabled")
        if not boost_audio: self.orig_gain_var.set(1.0)
        elif self.orig_gain_var.get() == 1.0: self.orig_gain_var.set(1.1)
        
        is_folder_mode = self.input_source_var.get() == "folder"
        self.select_files_button.configure(state="disabled" if is_folder_mode else "normal")
        if is_folder_mode:
            self.selected_files_paths.clear()

        self.update_file_status_list()
        self.update_audio_labels()
        self.update_completed_list()
        self.update_summary_text()

    def select_font(self):
        filepath = filedialog.askopenfilename(title="Chọn file font", filetypes=(("Font files", "*.ttf *.otf"), ("All files", "*.*")))
        if filepath: self.font_path_var.set(filepath); self.update_summary_text()

    def select_files(self):
        filepaths = filedialog.askopenfilenames(title="Chọn các tệp video", filetypes=(("Video files", "*.mp4"), ("All files", "*.*")))
        if filepaths:
            self.selected_files_paths = [Path(p) for p in filepaths]
            self.update_file_status_list()

    def select_bgm_file(self):
        filepath = filedialog.askopenfilename(title="Chọn tệp nhạc nền", filetypes=(("Audio files", "*.mp3 *.wav *.m4a *.aac *.flac"), ("All files", "*.*")))
        if filepath:
            self.bgm_path_var.set(filepath)
            self.bgm_filename_var.set(os.path.basename(filepath))
            log_message(f"Đã chọn nhạc nền: {os.path.basename(filepath)}")
            self.update_summary_text()
    
    def _get_selected_video_path(self):
        selected_indices = self.completed_listbox.curselection()
        if not selected_indices:
            log_message("⚠️ Vui lòng chọn một video từ danh sách.")
            return None
        filename = self.completed_listbox.get(selected_indices[0])
        filepath = DIR_VIDEO_OUTPUT / filename
        if not filepath.exists():
            log_message(f"❌ Lỗi: Không tìm thấy file {filepath}")
            self.update_completed_list()
            return None
        return filepath

    def open_selected_video(self):
        filepath = self._get_selected_video_path()
        if filepath:
            log_message(f"Đang mở video: {filepath}")
            if sys.platform == "win32": os.startfile(filepath)
            elif sys.platform == "darwin": subprocess.call(["open", filepath])
            else: subprocess.call(["xdg-open", filepath])

    def open_folder_location(self):
        filepath = self._get_selected_video_path()
        if filepath:
            folder_path = filepath.parent
            log_message(f"Đang mở thư mục: {folder_path}")
            if sys.platform == "win32": os.startfile(folder_path)
            elif sys.platform == "darwin": subprocess.call(["open", folder_path])
            else: subprocess.call(["xdg-open", folder_path])
            
    def process_log_queue(self):
        try:
            while True:
                msg = log_queue.get_nowait()
                self.log_textbox.configure(state="normal")
                self.log_textbox.insert("end", msg + "\n")
                self.log_textbox.configure(state="disabled")
                self.log_textbox.see("end")
        except queue.Empty: pass
        finally: self.after(100, self.process_log_queue)

    def start_processing(self, force_mode=None, force_source=None):
        if self.processing_thread and self.processing_thread.is_alive():
            log_message("!!! Đang có một tiến trình chạy. Vui lòng đợi hoàn tất.")
            return

        video_source = force_source
        if not video_source:
            if self.input_source_var.get() == "folder": video_source = DIR_VIDEO_INPUT
            elif self.input_source_var.get() == "files":
                if not self.file_status_listbox.paths:
                    log_message("⚠️ Lỗi: Bạn đã chọn chế độ 'Tệp tin' nhưng chưa chọn tệp nào.")
                    return
                video_source = self.file_status_listbox.paths
        
        if self.use_bgm_var.get() and self.bgm_mode_var.get() == "single" and not self.bgm_path_var.get():
             log_message("⚠️ Lỗi: Bạn đã chọn chế độ nhạc nền 'Chọn tệp' nhưng chưa chọn tệp nhạc nào.")
             return

        settings = self.get_settings_from_ui()
        if force_mode:
            settings['mode'] = force_mode

        self.save_settings(settings)
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")
        self.start_button.configure(state="disabled", text="Đang xử lý...")
        self.processing_thread = threading.Thread(target=self.run_worker, args=(settings, video_source), daemon=True)
        self.processing_thread.start()
        
    def start_sequential_join_from_list(self):
        if len(self.file_status_listbox.paths) < 2:
            log_message("⚠️ Cần ít nhất 2 video trong danh sách để thực hiện ghép nối tiếp.")
            return
        
        self.start_processing(force_mode="sequential", force_source=self.file_status_listbox.paths)

    def run_worker(self, settings, video_source):
        processed_files = set()
        created_files = []
        try:
            if settings.get('mode') == 'sequential':
                processed_files, created_files = run_mode_sequential(settings, video_source)
            else:
                processed_files, created_files = run_mode_random(settings, video_source)
            
            if self.input_source_var.get() == "files" and processed_files:
                log_message("Di chuyển các tệp nguồn đã xử lý xong...")
                for f_path in processed_files:
                    try: shutil.move(str(f_path), str(DIR_DONE_INPUT / f_path.name))
                    except Exception as e: log_message(f"  - Lỗi khi di chuyển file {f_path.name}: {e}")

            log_message("\n🎉 Xử lý hoàn tất!")
        except Exception as e:
            log_message(f"\n❌ LỖI KHÔNG MONG MUỐN: {e}")
        finally:
            self.after(0, self.on_processing_finished, created_files)

    def on_processing_finished(self, created_files):
        self.start_button.configure(state="normal", text="Bắt đầu xử lý với cài đặt đã chọn")
        self.update_file_status_list()
        self.update_completed_list()
        
        if created_files:
            latest_file = created_files[-1]
            try:
                items = self.completed_listbox.get(0, "end")
                if latest_file.name in items:
                    idx = items.index(latest_file.name)
                    self.completed_listbox.selection_clear(0, "end")
                    self.completed_listbox.selection_set(idx)
                    self.completed_listbox.activate(idx)
                    self.completed_listbox.see(idx)
            except ValueError: pass

    def get_settings_from_ui(self) -> dict:
        return {
            'mode': self.mode_var.get(),
            'aspect_ratio_choice': self.aspect_ratio_var.get(),
            'watermark_text': self.watermark_text_var.get() if self.watermark_var.get() else "",
            'font_path': self.font_path_var.get(),
            'transition': self.transition_var.get(),
            'transition_duration': self.transition_duration_var.get(),
            'use_bgm': self.use_bgm_var.get(),
            'bgm_mode': self.bgm_mode_var.get(),
            'bgm_path': self.bgm_path_var.get(),
            'bgm_gain': self.bgm_gain_var.get(),
            'orig_gain': self.orig_gain_var.get(),
            'num_to_combine': self.num_to_combine_var.get()
        }

    def save_settings(self, settings: dict):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
            log_message(f"INFO: Cài đặt đã được lưu vào file '{SETTINGS_FILE}'.")
        except IOError as e:
            log_message(f"ERROR: Không thể lưu file cài đặt: {e}")

    def load_settings(self):
        settings_path = Path(SETTINGS_FILE)
        if not settings_path.exists(): return
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            self.mode_var.set(settings.get('mode', 'random'))
            self.aspect_ratio_var.set(settings.get('aspect_ratio_choice', 'd'))
            wm_text = settings.get('watermark_text', "")
            self.watermark_var.set(bool(wm_text))
            self.watermark_text_var.set(wm_text)
            self.font_path_var.set(settings.get('font_path', "C:/Windows/Fonts/Arial.ttf"))
            self.transition_var.set(settings.get('transition', 'fade'))
            self.transition_duration_var.set(settings.get('transition_duration', 1.0))
            self.use_bgm_var.set(settings.get('use_bgm', True))
            self.bgm_mode_var.set(settings.get('bgm_mode', 'random'))
            self.bgm_path_var.set(settings.get('bgm_path', ''))
            self.bgm_gain_var.set(settings.get('bgm_gain', 0.1))
            orig_gain = settings.get('orig_gain', 1.0)
            self.boost_audio_var.set(orig_gain > 1.0)
            self.orig_gain_var.set(orig_gain if orig_gain > 1.0 else 1.1)
            self.num_to_combine_var.set(settings.get('num_to_combine', 3))
            log_message("INFO: Đã tải cài đặt từ lần chạy trước.")
        except (json.JSONDecodeError, IOError, KeyError) as e:
            log_message(f"ERROR: Không thể tải file cài đặt. Sử dụng cài đặt mặc định. Lỗi: {e}")
        finally:
            pass

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()