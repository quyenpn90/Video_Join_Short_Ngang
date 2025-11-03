#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import shutil
import subprocess
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ======================
# CONFIG (de tuy bien)
# ======================
VIDEO_CODEC    = "libx264"
CRF            = "23"
PRESET         = "veryfast"
PIX_FMT        = "yuv420p"
AUDIO_CODEC    = "aac"
AUDIO_BITRATE  = "192k"
LOOP_BGM       = True
SETTINGS_FILE  = "last_settings.json"

# ======================
# Thu muc (ngay canh .py)
# ======================
BASE = Path.cwd()
DIR_VIDEO_INPUT      = BASE / "Video_Input"
DIR_BGM              = BASE / "Sound_Background"
DIR_TEMP_OUTPUT      = BASE / "Temp_Output"
DIR_DONE_INPUT       = BASE / "DONE" / "Video_Input"
DIR_VIDEO_OUTPUT     = BASE / "Video_Output"
DIR_VIDEO_ERRORS     = BASE / "Video_Errors"

for d in [DIR_VIDEO_INPUT, DIR_BGM, DIR_TEMP_OUTPUT, DIR_DONE_INPUT, DIR_VIDEO_OUTPUT, DIR_VIDEO_ERRORS]:
    d.mkdir(parents=True, exist_ok=True)

# ======================
# Helpers
# ======================
def run(cmd: List[str]):
    cmd_str_list = [str(c) for c in cmd]
    print("RUNNING: " + " ".join(cmd_str_list))
    # Using Popen for better error visibility in console
    process = subprocess.Popen(cmd_str_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

def ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path)
        ], text=True, stderr=subprocess.PIPE)
        duration = float(out.strip())
        if duration <= 0: return 0.0
        return duration
    except Exception:
        return 0.0

def pick_random_bgm() -> Optional[Path]:
    cands = [p for p in DIR_BGM.glob("*") if p.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".flac"}]
    return random.choice(cands) if cands else None

def group_videos_by_prefix(input_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    # Regex da duoc cai tien de bat ca A_1, A_01, A_001...
    pat = re.compile(r"^(.+?_)\d+\.mp4$", re.IGNORECASE)
    for f in input_dir.glob("*.mp4"):
        m = pat.match(f.name)
        if m:
            key = m.group(1)
            groups.setdefault(key, []).append(f)
    return groups

def sort_videos_numerically(video_paths: List[Path]) -> List[Path]:
    """Sap xep video theo so trong ten file (vi du: A_1, A_2, A_10)"""
    pat = re.compile(r'_(\d+)\.mp4$', re.IGNORECASE)
    def get_number(path: Path):
        match = pat.search(path.name)
        return int(match.group(1)) if match else 0
    return sorted(video_paths, key=get_number)


def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        print(f"INFO: Cai dat da duoc luu vao file '{SETTINGS_FILE}'.")
    except IOError as e:
        print(f"ERROR: Khong the luu file cai dat: {e}")

def load_settings() -> Optional[dict]:
    settings_path = Path(SETTINGS_FILE)
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError): return None
    return None

# ======================
# Core FFmpeg Commands
# ======================
def normalize_video(in_path: Path, out_path: Path, target_w: int, target_h: int, orig_gain: float, clip_duration: Optional[float] = None) -> bool:
    """Chuan hoa video. Neu clip_duration la None, se giu nguyen thoi luong goc."""
    duration_info = f"{clip_duration:.2f}s (forced)" if clip_duration else "original duration"
    print(f"  - Dang chuan hoa file: {in_path.name} -> {duration_info}")
    try:
        v_filters = [f"fps=30", f"format={PIX_FMT}"]
        a_filters = [f"volume={orig_gain:.2f}", "aresample=44100"]

        if clip_duration is not None and clip_duration > 0:
            v_filters.extend([
                f"tpad=stop_mode=clone:stop_duration={clip_duration}",
                f"trim=duration={clip_duration}"
            ])
            a_filters.append(f"atrim=duration={clip_duration}")

        v_filters.extend([
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease",
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black",
            "setsar=1"
        ])

        filter_complex = f"[0:v]{','.join(v_filters)}[v];[0:a]{','.join(a_filters)}[a]"
        
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(in_path), "-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]", "-c:v", VIDEO_CODEC, "-preset", PRESET, "-crf", CRF, "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, str(out_path)]
        run(cmd)
        return True
    except Exception as e:
        print(f"  - ❌ LOI CHUAN HOA file {in_path.name}: {e}")
        return False

def build_pair_concat_cmd(segment1: Path, segment2: Path, out_path: Path, transition: str, transition_duration: float) -> bool:
    print(f"  - Dang ghep cap: {segment1.name} + {segment2.name}")
    try:
        duration1 = ffprobe_duration(segment1)
        duration2 = ffprobe_duration(segment2)
        if duration1 == 0 or duration2 == 0:
            raise RuntimeError(f"Khong the lay thoi luong cua file tam.")

        effective_transition = transition
        if not math.isclose(duration1, duration2, rel_tol=0.1):
            print(f"  - Canh bao: Phat hien 2 video khong can bang ({duration1:.2f}s vs {duration2:.2f}s). Tu dong ghep noi cung.")
            effective_transition = "none"

        if effective_transition != 'none':
            offset = duration1 - transition_duration
            filter_complex = (
                f"[0:v][1:v]xfade=transition={effective_transition}:duration={transition_duration}:offset={offset},format={PIX_FMT}[v];"
                f"[0:a][1:a]acrossfade=d={transition_duration}[a]"
            )
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
        print(f"  - ❌ LOI khi ghep cap: {e}")
        return False

def concat_sequential_videos(video_list: List[Path], out_path: Path) -> bool:
    """Ghep mot list video da chuan hoa theo phuong phap concat demuxer."""
    print(f"  - Dang ghep noi tiep {len(video_list)} video...")
    list_file = out_path.with_suffix(".txt")
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for video in video_list:
                f.write(f"file '{video.resolve()}'\n")
        
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_path)]
        run(cmd)
        return True
    except Exception as e:
        print(f"  - ❌ LOI khi ghep noi tiep: {e}")
        return False
    finally:
        if list_file.exists(): list_file.unlink()

def add_watermark_cmd(in_path: Path, out_path: Path, watermark_text: str, font_path: str):
    font_path_escaped = font_path.replace("\\", "/").replace(":", "\\:")
    watermark_filter = (f"drawtext=text='{watermark_text}':fontfile='{font_path_escaped}':fontsize=60:fontcolor=white@0.3:x='(w-tw)/2+(w-tw)/2*sin(t/5)':y='(h-th)/3+(h-th)/4*cos(t/7)'")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(in_path), "-vf", watermark_filter, "-c:v", VIDEO_CODEC, "-preset", PRESET, "-crf", CRF, "-c:a", "copy", str(out_path)]
    run(cmd)

def mix_bgm_cmd(video_path: Path, final_path: Path, bgm_path: Path, bgm_gain: float):
    try:
        video_duration_str = subprocess.check_output([ "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video_path) ], text=True, stderr=subprocess.PIPE)
        video_duration = float(video_duration_str.strip())
    except Exception as e:
        raise RuntimeError(f"Could not get final video duration: {e}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path)]
    if LOOP_BGM: cmd += ["-stream_loop", "-1", "-i", str(bgm_path)]
    else: cmd += ["-i", str(bgm_path)]
    filter_complex = (f"[0:a]aresample=async=1:first_pts=0[a0];[1:a]aresample=async=1:first_pts=0,volume={bgm_gain:.2f}[bg];[a0][bg]amix=inputs=2:duration=longest:normalize=0[aout]")
    cmd += ["-filter_complex", filter_complex, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, "-t", f"{video_duration:.3f}", str(final_path)]
    run(cmd)

def get_user_settings() -> dict:
    settings = {}
    while True:
        mode = input("Chon che do ghep video:\n  1. Ghep ngau nhien tu cac nhom khac nhau (A_1 + B_1 + C_1...)\n  2. Ghep noi tiep cac video trong cung mot nhom (A_1 + A_2 + A_3...)\nLua chon cua ban (1/2): ").strip()
        if mode == '1':
            settings['mode'] = 'random'
            break
        elif mode == '2':
            settings['mode'] = 'sequential'
            break
        print("Lua chon khong hop le.")

    while True:
        choice = input("1. Chon ty le khung hinh (d=doc 9:16, n=ngang 16:9): ").strip().lower()
        if choice in ['d', 'n']:
            settings['aspect_ratio_choice'] = choice
            break
        print("Lua chon khong hop le.")
    
    while True:
        choice = input("2. Co chen watermark (ten kenh) khong? (y/n): ").strip().lower()
        if choice in ['y', 'n']:
            settings['watermark_text'] = ""
            settings['font_path'] = "C:/Windows/Fonts/Arial.ttf"
            if choice == 'y':
                settings['watermark_text'] = input("   - Nhap ten kenh: ").strip().replace("'", "")
                font_path_in = input("   - Nhap duong dan file font (de trong de dung Arial mac dinh): ").strip()
                if font_path_in: settings['font_path'] = font_path_in
            break
        print("Lua chon khong hop le.")

    TRANSITIONS = [
        ('fade', 'Mo dan (co ban)'), ('dissolve', 'Tan bien (hoa tan vao nhau)'),
        ('pixelize', 'Hieu ung vo pixel (game)'), ('wipeleft', 'Quet tu phai qua trai'),
        ('wiperight', 'Quet tu trai qua phai'), ('wipeup', 'Quet tu duoi len'),
        ('wipedown', 'Quet tu tren xuong'), ('slideleft', 'Truot tu phai qua trai'),
        ('slideright', 'Truot tu trai qua phai'), ('slideup', 'Truot tu duoi len'),
        ('slidedown', 'Truot tu tren xuong'), ('circleopen', 'Vong tron mo ra (hoat hinh)'),
        ('circleclose', 'Vong tron dong lai'), ('horzopen', 'Mo ngang (nhu rem cua)'),
        ('vertopen', 'Mo doc (tu giua ra)'), ('diagtl', 'Quet cheo (tren-trai)'),
        ('radial', 'Quet tron (nhu kim dong ho)'), ('fadewhite', 'Mo dan qua mau trang'),
        ('fadeblack', 'Mo dan qua mau den')
    ]
    
    # Hieu ung chuyen canh chi ap dung cho che do ghep ngau nhien
    if settings['mode'] == 'random':
        while True:
            choice = input("3. Co them hieu ung chuyen canh khong? (y/n): ").strip().lower()
            if choice in ['y', 'n']:
                settings['transition'] = "none"
                if choice == 'y':
                    print("   - Cac hieu ung co san:")
                    for i, (name, desc) in enumerate(TRANSITIONS):
                        print(f"     {i}: {name.ljust(15)} ({desc})")
                    try:
                        t_idx = int(input("   - Chon so hieu ung: ").strip())
                        settings['transition'] = TRANSITIONS[t_idx][0]
                    except (ValueError, IndexError):
                        print("   - Lua chon khong hop le, se khong dung hieu ung.")
                break
            print("Lua chon khong hop le.")
        settings['transition_duration'] = 1.0
    else:
        # Che do noi tiep khong can hieu ung phuc tap
        settings['transition'] = "none"
        settings['transition_duration'] = 0.0

    while True:
        choice = input("4. Co chen nhac nen khong? (y/n): ").strip().lower()
        if choice in ['y', 'n']:
            settings['use_bgm'] = (choice == 'y')
            if settings['use_bgm']:
                while True:
                    try:
                        settings['bgm_gain'] = float(input("   - Am luong nhac nen (vi du: 0.1): ").strip())
                        break
                    except ValueError: print("Vui long nhap mot so.")
            break
        print("Lua chon khong hop le.")
    
    while True:
        choice = input("5. Co tang am thanh goc khong? (y/n): ").strip().lower()
        if choice in ['y', 'n']:
            settings['orig_gain'] = 1.0
            if choice == 'y':
                while True:
                    try:
                        settings['orig_gain'] = float(input("   - Muc tang (vi du: 1.1): ").strip())
                        break
                    except ValueError: print("Vui long nhap mot so.")
            break
        print("Lua chon khong hop le.")
    
    if settings['mode'] == 'random':
        while True:
            try:
                settings['num_to_combine'] = int(input("6. Ghep may video (tu cac nhom KHAC NHAU) vao lam mot? (vi du: 3, 5): ").strip())
                if settings['num_to_combine'] > 1: break
                print("So luong video de ghep phai lon hon 1.")
            except ValueError: print("Vui long nhap mot so nguyen.")
        
    return settings

def display_summary(settings: dict):
    arc = settings.get('aspect_ratio_choice')
    target_w, target_h = (1080, 1920) if arc == 'd' else (1920, 1080)
    mode = settings.get('mode', 'random')
    
    print("\n--- CAI DAT HIEN TAI ---")
    if mode == 'random':
        nc = settings.get('num_to_combine', 2)
        print(f"Che do ghep:          Ghep {nc} video (tu cac nhom khac nhau) thanh mot")
        tr, td = settings.get('transition', 'none'), settings.get('transition_duration', 1.0)
        print(f"Chuyen canh:          {tr} ({td}s)")
        print("(Thoi luong clip se duoc tu dong xac dinh theo video dau tien)")
    else:
        print("Che do ghep:          Ghep noi tiep cac video trong CUNG MOT NHOM")
        print("(Thoi luong clip se la tong thoi luong cua cac video con)")

    print(f"Ty le khung hinh:     {'Doc 9:16' if arc == 'd' else 'Ngang 16:9'} ({target_w}x{target_h})")
    wt = settings.get('watermark_text', "")
    fp = settings.get('font_path', 'default')
    print(f"Watermark:            {'Co' if wt else 'Khong'}" + (f" ('{wt}', Font: {fp})" if wt else ""))
    ub, bg = settings.get('use_bgm', False), settings.get('bgm_gain', 0.0)
    print(f"Chen nhac nen:        {'Co' if ub else 'Khong'}" + (f" (am luong {bg*100:.0f}%)" if ub else ""))
    og = settings.get('orig_gain', 1.0)
    print(f"Tang am goc:          {'Co' if og > 1.0 else 'Khong'}" + (f" (+{int((og - 1.0) * 100)}%)" if og > 1.0 else ""))
    print("----------------------")

def run_mode_random(settings: dict):
    print("\n🚀 Bat dau xu ly hang loat (Che do Ngau nhien)...")
    video_groups = group_videos_by_prefix(DIR_VIDEO_INPUT)
    group_keys = list(video_groups.keys())
    num_to_combine = settings.get('num_to_combine', 2)
    
    if len(group_keys) < num_to_combine:
        print(f"⚠️ Khong du nhom video. Can {num_to_combine} nhom, chi co {len(group_keys)}.")
        return
        
    random.shuffle(group_keys)
    
    batch_index = 1
    while len(group_keys) >= num_to_combine:
        print(f"\n--- 🔄 Dang xu ly lo #{batch_index} ---")
        selected_keys = [group_keys.pop(0) for _ in range(num_to_combine)]
        
        temp_files_to_delete: List[Path] = []
        try:
            arc = settings.get('aspect_ratio_choice')
            target_w, target_h = (1080, 1920) if arc == 'd' else (1920, 1080)
            orig_gain = settings.get('orig_gain', 1.0)
            
            videos_to_process_orig = [random.choice(video_groups[key]) for key in selected_keys]

            print("Buoc 1: Chuan hoa cac video dau vao...")
            clip_duration = ffprobe_duration(videos_to_process_orig[0])
            if clip_duration == 0:
                print(f"  - ❌ LOI: Khong do duoc thoi luong file chuan {videos_to_process_orig[0].name}. Bo qua lo nay.")
                for video_path in videos_to_process_orig:
                    shutil.move(str(video_path), str(DIR_VIDEO_ERRORS / video_path.name))
                continue
            
            print(f"  - Thoi luong chuan cho lo nay la: {clip_duration:.2f} giay.")
            
            normalized_videos: List[Path] = []
            failed_videos: List[Path] = []

            for video_path in videos_to_process_orig:
                norm_path = DIR_TEMP_OUTPUT / f"NORM_{batch_index}_{video_path.name}"
                temp_files_to_delete.append(norm_path)
                if normalize_video(video_path, norm_path, target_w, target_h, orig_gain, clip_duration):
                    normalized_videos.append(norm_path)
                else:
                    failed_videos.append(video_path)
            
            for video_path in failed_videos:
                shutil.move(str(video_path), str(DIR_VIDEO_ERRORS / video_path.name))
                print(f"  - Da di chuyen file loi '{video_path.name}' vao thu muc '{DIR_VIDEO_ERRORS.name}'.")

            if len(normalized_videos) < 2:
                print(f"  - ⚠️ Khong du video hop le de ghep. Bo qua lo nay.")
                continue

            current_level_videos = normalized_videos
            level = 1
            while len(current_level_videos) > 1:
                print(f"Buoc 2.{level}: Ghep cap vong {level}...")
                next_level_videos = []
                for i in range(0, len(current_level_videos), 2):
                    if i + 1 < len(current_level_videos):
                        clip1, clip2 = current_level_videos[i], current_level_videos[i+1]
                        pair_out_path = DIR_TEMP_OUTPUT / f"TEMP_{batch_index}_level{level}_pair{i//2}.mp4"
                        temp_files_to_delete.append(pair_out_path)
                        
                        if not build_pair_concat_cmd(clip1, clip2, pair_out_path, settings.get('transition', 'none'), settings.get('transition_duration', 1.0)):
                            raise Exception("Ghep cap that bai.")
                        next_level_videos.append(pair_out_path)
                    else:
                        next_level_videos.append(current_level_videos[i])
                current_level_videos = next_level_videos
                level += 1
            
            final_concatenated_video = current_level_videos[0]
            current_processed_file = final_concatenated_video
            
            if settings.get('watermark_text'):
                print("Buoc 3: Them watermark...")
                watermarked_file = DIR_TEMP_OUTPUT / f"TEMP_WM_{batch_index}.mp4"
                temp_files_to_delete.append(watermarked_file)
                add_watermark_cmd(current_processed_file, watermarked_file, settings.get('watermark_text', ""), settings.get('font_path', "C:/Windows/Fonts/Arial.ttf"))
                current_processed_file = watermarked_file

            output_name_parts = [Path(p).stem.split('_')[0] for p in videos_to_process_orig if p not in failed_videos]
            output_filename = "_".join(output_name_parts) + ".mp4"
            final_output_path = DIR_VIDEO_OUTPUT / output_filename
            
            if settings.get('use_bgm'):
                print("Buoc 4: Tron nhac nen...")
                bgm = pick_random_bgm()
                if not bgm:
                    print("⚠️ Khong tim thay BGM.")
                    shutil.move(str(current_processed_file), str(final_output_path))
                else:
                    print(f"🔊 Da chon BGM: {bgm.name}")
                    mix_bgm_cmd(current_processed_file, final_output_path, bgm, settings.get('bgm_gain', 0.1))
            else:
                shutil.move(str(current_processed_file), str(final_output_path))

            for video_path in videos_to_process_orig:
                if video_path not in failed_videos:
                    shutil.move(str(video_path), str(DIR_DONE_INPUT / video_path.name))
            
            print(f"✅ Hoan thanh lo #{batch_index}: {final_output_path.name}")
        except Exception as e:
            print(f"❌ LOI khi xu ly lo #{batch_index}: {e}")
            print("  - Cac file nguon cho lo nay se khong bi di chuyen.")
        finally:
            print("  - Don dep file tam...")
            for temp_file in temp_files_to_delete:
                temp_file.unlink(missing_ok=True)
        batch_index += 1
        
    if group_keys:
        print(f"\nℹ️ Con du {len(group_keys)} nhom video, khong du de tao lo tiep theo.")

def run_mode_sequential(settings: dict):
    print("\n🚀 Bat dau xu ly hang loat (Che do Noi tiep)...")
    video_groups = group_videos_by_prefix(DIR_VIDEO_INPUT)

    if not video_groups:
        print("⚠️ Khong tim thay nhom video nao de xu ly (dinh dang ten file phai la Ten_So.mp4).")
        return
        
    batch_index = 1
    for prefix, video_list in video_groups.items():
        if len(video_list) < 2:
            print(f"\n--- ℹ️  Bo qua nhom '{prefix.strip('_')}' vi chi co 1 video ---")
            continue

        print(f"\n--- 🔄 Dang xu ly nhom '{prefix.strip('_')}' (lo #{batch_index}) ---")
        
        temp_files_to_delete: List[Path] = []
        try:
            arc = settings.get('aspect_ratio_choice')
            target_w, target_h = (1080, 1920) if arc == 'd' else (1920, 1080)
            orig_gain = settings.get('orig_gain', 1.0)
            
            # Sap xep video theo so
            videos_to_process_orig = sort_videos_numerically(video_list)
            print(f"  - Phat hien {len(videos_to_process_orig)} video. Thu tu ghep:")
            for i, p in enumerate(videos_to_process_orig):
                print(f"    {i+1}. {p.name}")


            print("Buoc 1: Chuan hoa cac video dau vao...")
            normalized_videos: List[Path] = []
            failed_videos: List[Path] = []

            for video_path in videos_to_process_orig:
                norm_path = DIR_TEMP_OUTPUT / f"NORM_{batch_index}_{video_path.name}"
                temp_files_to_delete.append(norm_path)
                # Pass None cho clip_duration de giu nguyen thoi luong
                if normalize_video(video_path, norm_path, target_w, target_h, orig_gain, clip_duration=None):
                    normalized_videos.append(norm_path)
                else:
                    failed_videos.append(video_path)
            
            for video_path in failed_videos:
                shutil.move(str(video_path), str(DIR_VIDEO_ERRORS / video_path.name))
                print(f"  - Da di chuyen file loi '{video_path.name}' vao thu muc '{DIR_VIDEO_ERRORS.name}'.")

            if len(normalized_videos) < 2:
                print(f"  - ⚠️ Khong du video hop le de ghep. Bo qua nhom nay.")
                continue

            print("Buoc 2: Ghep noi tiep...")
            concatenated_file = DIR_TEMP_OUTPUT / f"TEMP_CONCAT_{batch_index}.mp4"
            temp_files_to_delete.append(concatenated_file)
            if not concat_sequential_videos(normalized_videos, concatenated_file):
                raise Exception("Ghep noi tiep that bai.")
            
            current_processed_file = concatenated_file
            
            if settings.get('watermark_text'):
                print("Buoc 3: Them watermark...")
                watermarked_file = DIR_TEMP_OUTPUT / f"TEMP_WM_{batch_index}.mp4"
                temp_files_to_delete.append(watermarked_file)
                add_watermark_cmd(current_processed_file, watermarked_file, settings.get('watermark_text', ""), settings.get('font_path', "C:/Windows/Fonts/Arial.ttf"))
                current_processed_file = watermarked_file

            output_filename = f"{prefix.strip('_')}_final.mp4"
            final_output_path = DIR_VIDEO_OUTPUT / output_filename
            
            if settings.get('use_bgm'):
                print("Buoc 4: Tron nhac nen...")
                bgm = pick_random_bgm()
                if not bgm:
                    print("⚠️ Khong tim thay BGM.")
                    shutil.move(str(current_processed_file), str(final_output_path))
                else:
                    print(f"🔊 Da chon BGM: {bgm.name}")
                    mix_bgm_cmd(current_processed_file, final_output_path, bgm, settings.get('bgm_gain', 0.1))
            else:
                shutil.move(str(current_processed_file), str(final_output_path))

            # Di chuyen file da xu ly xong
            for video_path in videos_to_process_orig:
                if video_path not in failed_videos:
                    shutil.move(str(video_path), str(DIR_DONE_INPUT / video_path.name))
            
            print(f"✅ Hoan thanh nhom '{prefix.strip('_')}': {final_output_path.name}")
        except Exception as e:
            print(f"❌ LOI khi xu ly nhom '{prefix.strip('_')}': {e}")
            print("  - Cac file nguon cho nhom nay se khong bi di chuyen.")
        finally:
            print("  - Don dep file tam...")
            for temp_file in temp_files_to_delete:
                temp_file.unlink(missing_ok=True)
        batch_index += 1

def run_interactive_mode():
    print("--- 🎬 Video Processing Script ---")
    settings = None
    last_settings = load_settings()
    if last_settings:
        print("\n--- Phat hien cai dat tu lan chay truoc ---")
        display_summary(last_settings)
        while True:
            use_last = input("Ban co muon su dung lai cai dat nay khong? (y/n): ").strip().lower()
            if use_last in ['y', 'n']:
                if use_last == 'y': settings = last_settings
                break
            print("Lua chon khong hop le.")
    if not settings:
        print("\n--- Vui long thiet lap cac tuy chon moi ---")
        settings = get_user_settings()

    display_summary(settings)
    proceed = input("Bat dau xu ly voi cac cai dat tren? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Da huy.")
        return
        
    save_settings(settings)
    
    # Chon che do chay
    if settings.get('mode') == 'sequential':
        run_mode_sequential(settings)
    else: # Mac dinh la random
        run_mode_random(settings)

    print("\n🎉 Xu ly hoan tat!")

if __name__ == "__main__":
    run_interactive_mode()