# Phiên bản tự động quét /videos và /shorts, thêm cột "Loại".
# Sửa lỗi báo hỏng khi tải MP4 (do check file M4A)
# Thêm tính năng Quản lý List Link
# YÊU CẦU MỚI: Chia đôi UI (Thêm link | Quản lý list) + Tăng kích thước Form
# YÊU CẦU MỚI 2: Thu gọn Proxy/Luồng, Ưu tiên List Video
# YÊU CẦU MỚI 3: Tinh chỉnh Text cột Status, Content Status, Size Header
# YÊU CẦU MỚI 4: Đổi tên Header 'Content Status'->'Sub', 'Status'->'Status', Text 'Gốc'->'Sẵn'
# YÊU CẦU MỚI 5: Quản lý Proxy List bằng file proxies.txt
# YÊU CẦU MỚI 6 (10/20): Thêm Cache Thumbnail (7 ngày) + Context Menu (Xem/Tải Thumbnail)

import customtkinter as ctk
# Thêm messagebox để xác nhận xóa
from tkinter import filedialog, ttk, Menu, messagebox
import threading
import queue
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import sys
import subprocess
import re
import concurrent.futures
import webbrowser
import math
import tempfile
import shutil
import unicodedata # Để chuẩn hóa unicode cho tên file
import time # Thêm để quản lý cache
import urllib.request # Thêm để tải thumbnail

try:
    import yt_dlp
    from yt_dlp.utils import DownloadError # Import lỗi cụ thể
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    DownloadError = Exception # Định nghĩa là Exception chung nếu chưa cài yt-dlp

# --- Cài đặt Cache Thumbnail ---
CACHE_PATH = Path.cwd() / "downloader_cache"
CACHE_DURATION_DAYS = 7
# ------------------------------

# --- Hàm trợ giúp để làm sạch tên file/thư mục ---
def sanitize_filename(text: str, replace_with: str = "_") -> str:
    """Làm sạch chuỗi để an toàn khi dùng làm tên file."""
    if not isinstance(text, str): text = str(text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    sanitized = re.sub(r'[\\/*?:"<>|]', replace_with, text)
    sanitized = sanitized.strip().rstrip('.')
    if replace_with: sanitized = re.sub(f'{re.escape(replace_with)}+', replace_with, sanitized)
    return sanitized if sanitized else "downloaded_file"

# --- LỚP DIALOG ĐỂ CHỌN PHỤ ĐỀ (Logic ưu tiên đã sửa) ---
class SubtitleDialog(ctk.CTkToplevel):
    def __init__(self, master, sub_list: Dict[str, str], original_language_code: str = None):
        super().__init__(master)
        self.title("Chọn Ngôn ngữ Phụ đề")
        self.geometry("350x150")
        self.transient(master)
        self.grab_set()
        self.selection = None
        self.sub_map = sub_list
        display_names = list(self.sub_map.keys())
        self.label = ctk.CTkLabel(self, text="Vui lòng chọn ngôn ngữ phụ đề để tải:")
        self.label.pack(padx=20, pady=(20, 10))
        if display_names:
            self.option_menu = ctk.CTkOptionMenu(self, values=display_names)
            self.option_menu.pack(padx=20, pady=5, fill="x")
            best_match = next((name for name in display_names if "(Original)" in name), None)
            if not best_match and original_language_code:
                best_match_main_manual = next((name for name in display_names if f"({original_language_code})" in name and "(Original)" not in name and "(Tự động)" not in name), None)
                if best_match_main_manual: best_match = best_match_main_manual
                else:
                    best_match_main_auto = next((name for name in display_names if f"({original_language_code})" in name and "(Tự động)" in name), None)
                    if best_match_main_auto: best_match = best_match_main_auto
                    else:
                        base_lang = original_language_code.split('-')[0]
                        best_match = next((name for name in display_names if f"({base_lang})" in name), None)
            if not best_match: best_match = next((name for name in display_names if "(vi)" in name or "Vietnamese" in name), None)
            if not best_match: best_match = next((name for name in display_names if "(en)" in name or "English" in name), None)
            if best_match and best_match in display_names: self.option_menu.set(best_match)
            elif display_names: self.option_menu.set(display_names[0])
        else:
            self.no_subs_label = ctk.CTkLabel(self, text="Không có phụ đề nào.")
            self.no_subs_label.pack(padx=20, pady=5)
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.pack(padx=20, pady=10, fill="x")
        self.ok_button = ctk.CTkButton(self.button_frame, text="Tải", command=self._on_ok)
        if not display_names: self.ok_button.configure(state="disabled")
        self.ok_button.pack(side="right", padx=(10, 0))
        self.cancel_button = ctk.CTkButton(self.button_frame, text="Hủy", command=self.destroy, fg_color="gray")
        self.cancel_button.pack(side="right")
    def _on_ok(self):
        if hasattr(self, 'option_menu'): self.selection = self.sub_map.get(self.option_menu.get())
        self.destroy()
    def get_selection(self): return self.selection

# --- Các hàm định dạng (Không thay đổi) ---
def format_number(n):
    if n is None: return "N/A"
    try:
        n = int(n); f = float(n)
        if n >= 1_000_000_000: return f"{f / 1_000_000_000:.1f}B"
        if n >= 1_000_000: return f"{f / 1_000_000:.1f}M"
        if n >= 1_000: return f"{f / 1_000:.1f}K"
        return str(n)
    except: return "N/A"

def format_size(size_bytes):
    if size_bytes is None: return "N/A"
    try:
        size_bytes = float(size_bytes); base = 1024
        if size_bytes == 0: return "0 B"
        if size_bytes >= base ** 3: return f"{size_bytes / base ** 3:.2f} GB"
        if size_bytes >= base ** 2: return f"{size_bytes / base ** 2:.2f} MB"
        if size_bytes >= base: return f"{size_bytes / base:.2f} KB"
        return f"{int(size_bytes)} B"
    except: return "N/A"

def format_duration(seconds: Optional[float]) -> str:
    if seconds is None: return "N/A"
    try:
        seconds = int(seconds); hours = seconds // 3600
        minutes = (seconds % 3600) // 60; secs = seconds % 60
        if hours > 0: return f"{hours:02}:{minutes:02}:{secs:02}"
        else: return f"{minutes:02}:{secs:02}"
    except: return "N/A"

# --- HÀM PHÂN TÍCH VTT (DỰA TRÊN CODE BẠN CUNG CẤP) ---
def _clean_vtt_tags(text):
    pattern = re.compile(r'<[^>]+>')
    return pattern.sub('', text)

def _parse_vtt_file_to_clean_text(vtt_file_path: str) -> str:
    cleaned_lines, seen_lines = [], set()
    try:
        with open(vtt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if ('-->' in line or not stripped or stripped.isdigit() or
                    stripped == 'WEBVTT' or stripped.startswith("Kind:") or
                    stripped.startswith("Language:")): continue
                clean_line = _clean_vtt_tags(stripped)
                if clean_line and clean_line not in seen_lines:
                    cleaned_lines.append(clean_line)
                    seen_lines.add(clean_line)
        return ' '.join(cleaned_lines)
    except FileNotFoundError: print(f"[LỖI] Không tìm thấy VTT: '{vtt_file_path}'"); return ""
    except Exception as e: print(f"[LỖI] Lỗi phân tích VTT: {e}"); return ""

# --- Worker Process (Lấy đầy đủ thông tin) ---
# Sửa đổi để nhận proxy_url đã được định dạng
def scan_worker_process(url_list, proxy_url, detail_queue, log_queue, thread_count):
    def log(msg): log_queue.put(msg)
    def fetch_single_video_details(entry, default_video_type="Video"):
        video_url_to_fetch = entry.get('webpage_url') or entry.get('original_url') or entry.get('url')
        if not video_url_to_fetch:
            log(f"❌ Worker Thread: Không tìm thấy URL trong entry: {entry.get('id') or entry}")
            raise ValueError("Missing video URL")
        yt_opts = {
            'quiet': True, 'no_warnings': True, 'extract_flat': False, 'forcejson': True,
            'fields': [
                'id', 'title', 'channel', 'upload_date', 'view_count', 'like_count',
                'comment_count', 'webpage_url', 'extractor_key', 'filesize_approx', 'duration',
                'description', 'tags', 'thumbnail', 'language', 'subtitles', 'automatic_captions'
            ]
        }
        # Sử dụng proxy_url đã định dạng
        if proxy_url: yt_opts['proxy'] = proxy_url
        try:
            with yt_dlp.YoutubeDL(yt_opts) as ydl:
                details = ydl.extract_info(video_url_to_fetch, download=False)
                # Thêm/Ghi đè 'video_type' vào chi tiết
                if details:
                    # Quyết định loại video dựa trên thời lượng nếu là video đơn lẻ
                    if default_video_type == "Video": # Chỉ kiểm tra nếu nó không phải từ tab /shorts
                        duration = details.get('duration')
                        if duration is not None and duration <= 60:
                            details['video_type'] = "Shorts"
                        else:
                            details['video_type'] = "Video"
                    else: # Nếu nó đến từ tab /shorts, nó là "Shorts"
                        details['video_type'] = default_video_type
                return details
        except Exception as e:
            log(f"❌ Worker Thread Lỗi lấy chi tiết URL '{video_url_to_fetch}': {type(e).__name__} - {e}")
            raise

    try:
        ydl_opts_flat = {'quiet': True, 'extract_flat': 'in_playlist', 'force_generic_extractor': False}
        # Sử dụng proxy_url đã định dạng
        if proxy_url: ydl_opts_flat['proxy'] = proxy_url
        global_entry_offset = 0
        total_videos_fetched_details = 0
        processed_urls = set()

        with yt_dlp.YoutubeDL(ydl_opts_flat) as ydl:
            for url in url_list:
                if url in processed_urls: continue
                processed_urls.add(url)
                log(f"Worker: Đang quét URL: {url}")

                # --- LOGIC MỚI: XÁC ĐỊNH LOẠI VIDEO ---
                video_type = "Video" # Mặc định
                if url.endswith('/shorts'):
                    video_type = "Shorts"
                # --- KẾT THÚC LOGIC MỚI ---

                current_channel_entries = []
                try:
                    info = ydl.extract_info(url, download=False)
                    if not info: continue
                    if info.get('_type') == 'playlist' or 'entries' in info:
                        if entries := info.get('entries'):
                            for entry in entries: # Thêm loại video vào từng entry
                                if entry: entry['video_type'] = video_type
                            current_channel_entries.extend(filter(None, entries))
                        else: continue
                    else: # Là video đơn lẻ
                        info['video_type'] = video_type # Sẽ được kiểm tra lại bằng duration sau
                        current_channel_entries.append(info)
                except Exception as e: log(f"❌ Worker: Lỗi quét URL: {url} - {e}"); continue
                if not current_channel_entries: continue

                detail_queue.put(("POPULATE_APPEND", current_channel_entries))
                log(f"Worker: Tìm thấy {len(current_channel_entries)} video. Đang lấy chi tiết...")

                with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                    future_to_entry = {
                        # Truyền video_type mặc định vào hàm fetch
                        executor.submit(fetch_single_video_details, entry, entry.get('video_type', 'Video')): (i + global_entry_offset, entry)
                        for i, entry in enumerate(current_channel_entries) if entry
                    }
                    for future in concurrent.futures.as_completed(future_to_entry):
                        index, entry = future_to_entry[future]
                        try:
                            video_details = future.result()
                            if video_details: detail_queue.put(("UPDATE", (str(index), video_details)))
                        except Exception: pass
                        finally:
                            total_videos_fetched_details += 1
                            log_queue.put(f"PROGRESS:{total_videos_fetched_details}")
                global_entry_offset += len(current_channel_entries)
        log(f"✅ Worker: Hoàn thành quét tất cả URL.")
    except Exception as e: log(f"❌ Worker: Lỗi nghiêm trọng: {e}")
    finally: detail_queue.put(("FINISH_SCAN", None))

# --- Lớp Giao Diện Chính ---
class DownloaderTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) # Dòng 1 (list_frame) sẽ co giãn chính
        self.grid_rowconfigure(0, weight=0) # Dòng 0 (input_frame) cố định
        self.grid_rowconfigure(2, weight=0) # Dòng 2 (bottom_frame) cố định

        self.scan_process = None
        self.download_thread = None
        self.content_download_thread = None
        self.log_queue = mp.Queue()
        self.detail_queue = mp.Queue()
        self.status_queue = queue.Queue()
        self.tree_item_map: Dict[str, Dict[str, Any]] = {}
        self.sort_column = 'index'
        self.sort_reverse = False
        self.output_path_var = ctk.StringVar(value=str(Path.cwd() / "Downloads"))
        # --- Proxy Management ---
        self.proxy_file_path = Path.cwd() / "proxies.txt"
        self.proxy_list = ["Kết nối trực tiếp"] # Danh sách proxy đọc từ file
        self.selected_proxy_var = ctk.StringVar(value=self.proxy_list[0])
        # ---
        self.total_videos_var = ctk.StringVar(value="Tổng video: 0")
        self.thread_count_var = ctk.StringVar(value="4")
        self.quality_var = ctk.StringVar(value="Best")

        # --- Thêm cho tính năng Quản lý List Link ---
        self.manage_link_path = Path.cwd() / "ManageLink"
        self.manage_link_path.mkdir(exist_ok=True)
        self.link_list_var = ctk.StringVar(value="[Chọn list link]")
        self.link_lists: Dict[str, Path] = {} # Lưu {tên_list: đường_dẫn_file}
        # --- Kết thúc ---
        
        # --- Thêm cho Cache Thumbnail ---
        self.cache_path = CACHE_PATH
        self.cache_path.mkdir(exist_ok=True)
        # -------------------------------

        self.create_widgets() # Tạo widget trước khi load proxy
        self._load_proxies() # Load proxy list lần đầu

        if not YT_DLP_AVAILABLE:
            self.log_message("="*50 + "\nCẢNH BÁO: 'yt-dlp' chưa được cài đặt.\n" + "="*50)
            self.scan_button.configure(state="disabled")
            if hasattr(self, 'start_download_button'): self.start_download_button.configure(state="disabled")
        
        # Chạy dọn dẹp cache cũ trong 1 thread riêng
        threading.Thread(target=self._cleanup_old_cache, daemon=True).start()
        
        self.after(100, self.process_queues)

    def log_message(self, msg: str): self.log_queue.put(msg)

    def process_queues(self):
        try:
            while not self.log_queue.empty():
                msg = self.log_queue.get_nowait()
                if isinstance(msg, str) and msg.startswith("PROGRESS:"):
                    try:
                        count = int(msg.split(':')[1])
                        total = len(self.tree_item_map)
                        if total > 0: self.scan_button.configure(text=f"Tải chi tiết {count}/{total} ({int(count/total*100)}%)")
                        else: self.scan_button.configure(text=f"Tải chi tiết {count}/?")
                    except: self.scan_button.configure(text=f"Tải chi tiết...")
                else:
                    self.log_textbox.configure(state="normal")
                    self.log_textbox.insert("end", str(msg) + "\n")
                    self.log_textbox.configure(state="disabled")
                    self.log_textbox.see("end")
            for _ in range(50):
                if self.detail_queue.empty(): break
                signal, data = self.detail_queue.get_nowait()
                if signal == "POPULATE_APPEND": self._append_to_list(data)
                elif signal == "UPDATE":
                    iid, details = data
                    if details: self._update_treeview_row(iid, details)
                elif signal == "FINISH_SCAN":
                    self.scan_button.configure(state="normal", text="Quét Video")
                    self.total_videos_var.set(f"Tổng video: {len(self.tree_item_map)}")
            while not self.status_queue.empty():
                iid, column, status_text = self.status_queue.get_nowait()
                if self.video_tree.exists(iid):
                    self.video_tree.set(iid, column, status_text)
                    if column == 'content_status' and iid in self.tree_item_map:
                        self.tree_item_map[iid]['content_status_text'] = status_text
                    elif column == 'status' and iid in self.tree_item_map:
                        self.tree_item_map[iid]['status_text'] = status_text
        except queue.Empty: pass
        except Exception as e: print(f"Lỗi trong process_queues: {e}")
        finally: self.after(100, self.process_queues)


    # ========================================================================
    # ===== HÀM create_widgets ĐÃ TỐI ƯU UI/UX ===============================
    # ========================================================================
    def create_widgets(self):
        # --- Khung nhập liệu chính (TOP) ---
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=0, column=0, padx=10, pady=(10,5), sticky="nsew") # Giảm pady bottom
        input_frame.grid_columnconfigure(0, weight=1)

        # --- Khung chứa 2 cột (Thêm link | Quản lý) ---
        top_controls_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        top_controls_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=(0,5))
        top_controls_frame.grid_columnconfigure(0, weight=1) # Cột 1 co giãn
        top_controls_frame.grid_columnconfigure(1, weight=1) # Cột 2 co giãn

        # --- Cột 1: Thêm Link ---
        add_link_frame = ctk.CTkFrame(top_controls_frame)
        add_link_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        add_link_frame.grid_rowconfigure(1, weight=1)
        add_link_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(add_link_frame, text="Nhập Links (mỗi link một dòng):", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(5,0), sticky="w")
        self.url_textbox = ctk.CTkTextbox(add_link_frame, height=120) # Giảm chiều cao một chút
        self.url_textbox.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew")

        # --- Cột 2: Quản lý List Link ---
        manage_link_frame = ctk.CTkFrame(top_controls_frame)
        manage_link_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        manage_link_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(manage_link_frame, text="Quản lý List Link:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(5,5))

        self.link_list_menu = ctk.CTkOptionMenu(manage_link_frame, variable=self.link_list_var, command=self._load_link_list)
        self.link_list_menu.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Khung cho 2 nút Save/Delete
        manage_button_frame = ctk.CTkFrame(manage_link_frame, fg_color="transparent")
        manage_button_frame.grid(row=2, column=0, sticky="ew", pady=5) # Giảm pady
        manage_button_frame.grid_columnconfigure(0, weight=1)
        manage_button_frame.grid_columnconfigure(1, weight=1)

        self.save_list_button = ctk.CTkButton(manage_button_frame, text="💾 Lưu List", command=self._save_link_list)
        self.save_list_button.grid(row=0, column=0, padx=(10,5), sticky="ew")

        self.delete_list_button = ctk.CTkButton(manage_button_frame, text="❌ Xóa List", fg_color="#D32F2F", hover_color="#B71C1C", command=self._delete_link_list)
        self.delete_list_button.grid(row=0, column=1, padx=(5,10), sticky="ew")

        # --- Khung Cài đặt (Proxy, Luồng) & Nút Quét ---
        settings_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        settings_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(0,5))
        settings_frame.grid_columnconfigure(1, weight=1) # Cho OptionMenu Proxy co giãn

        # --- Proxy Selection ---
        ctk.CTkLabel(settings_frame, text="Proxy:").pack(side="left", padx=(0, 5))
        self.proxy_menu = ctk.CTkOptionMenu(settings_frame, variable=self.selected_proxy_var, values=self.proxy_list) # values sẽ được cập nhật bởi _load_proxies
        self.proxy_menu.pack(side="left", padx=(0, 5), fill="x", expand=True)
        self.edit_proxy_button = ctk.CTkButton(settings_frame, text="Sửa", width=50, command=self._open_proxy_file)
        self.edit_proxy_button.pack(side="left", padx=(0, 5))
        self.refresh_proxy_button = ctk.CTkButton(settings_frame, text="Làm mới", width=80, command=self._load_proxies)
        self.refresh_proxy_button.pack(side="left", padx=(0, 15))
        # ---

        # Số luồng
        ctk.CTkLabel(settings_frame, text="Luồng quét:").pack(side="left", padx=(0, 5))
        self.thread_entry = ctk.CTkEntry(settings_frame, textvariable=self.thread_count_var, width=40) # Giảm width
        self.thread_entry.pack(side="left", padx=(0, 15))

        # Nút Quét
        self.scan_button = ctk.CTkButton(settings_frame, text="Quét Video", command=self._start_scan, width=120) # Chiều rộng cố định
        self.scan_button.pack(side="left", padx=(0, 0)) # Bỏ padx phải

        # --- Khung danh sách video (MIDDLE) ---
        list_frame = ctk.CTkFrame(self)
        list_frame.grid(row=1, column=0, padx=10, pady=0, sticky="nsew") # Dòng này weight=1, bỏ pady top
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(1, weight=1) # Treeview sẽ co giãn trong frame này

        # Khung action của list (Xóa, Chọn All, Bỏ chọn)
        list_actions_frame = ctk.CTkFrame(list_frame, fg_color="transparent")
        list_actions_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=(5,5), sticky="ew") # Thêm columnspan=2
        ctk.CTkButton(list_actions_frame, text="Xóa danh sách", width=120, command=self._clear_list).pack(side="left", padx=(0, 5))
        ctk.CTkButton(list_actions_frame, text="Chọn tất cả", width=120, command=self._select_all_tree).pack(side="left", padx=5)
        ctk.CTkButton(list_actions_frame, text="Bỏ chọn tất cả", width=120, command=self._deselect_all_tree).pack(side="left", padx=5)

        # Cấu hình style Treeview (không đổi)
        style = ttk.Style()
        style.theme_use("default")
        bg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        text_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        header_bg = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        style.configure("Treeview", background=bg_color, foreground=text_color, fieldbackground=bg_color, borderwidth=0, rowheight=25)
        style.map('Treeview', background=[('selected', ctk.ThemeManager.theme["CTkButton"]["hover_color"][1])])
        style.configure("Treeview.Heading", background=header_bg, foreground=text_color, relief="flat", font=('Segoe UI', 10, 'bold'))
        style.map("Treeview.Heading", background=[('active', ctk.ThemeManager.theme["CTkButton"]["hover_color"][1])])

        # Treeview (columns definition không đổi)
        columns = ('index', 'status', 'content_status', 'title', 'type', 'language', 'duration', 'size', 'channel', 'views', 'likes', 'comments', 'date')
        self.video_tree = ttk.Treeview(list_frame, columns=columns, show='headings', selectmode="extended")

        # --- YÊU CẦU: Tinh chỉnh Headers ---
        self.video_tree.heading('index', text='#', command=lambda: self._sort_tree_column('index', False))
        self.video_tree.column('index', width=40, stretch=False, anchor='center')
        self.video_tree.heading('status', text='Status', command=lambda: self._sort_tree_column('status', False)) # Đổi Header
        self.video_tree.column('status', width=70, stretch=False, anchor='center')
        self.video_tree.heading('content_status', text='Sub', command=lambda: self._sort_tree_column('content_status', False)) # Đổi Header
        self.video_tree.column('content_status', width=70, stretch=False, anchor='center')
        self.video_tree.heading('title', text='Tiêu đề', command=lambda: self._sort_tree_column('title', False))
        self.video_tree.column('title', width=280, stretch=True)
        self.video_tree.heading('type', text='Loại', command=lambda: self._sort_tree_column('type', False))
        self.video_tree.column('type', width=70, stretch=False, anchor='center')
        self.video_tree.heading('language', text='Ngôn ngữ', command=lambda: self._sort_tree_column('language', False))
        self.video_tree.column('language', width=60, stretch=False, anchor='center')
        self.video_tree.heading('duration', text='Thời lượng', command=lambda: self._sort_tree_column('duration', True))
        self.video_tree.column('duration', width=80, stretch=False, anchor='center')
        self.video_tree.heading('size', text='Size', command=lambda: self._sort_tree_column('size', True)) # Xác nhận Header
        self.video_tree.column('size', width=100, stretch=False, anchor='e')
        self.video_tree.heading('channel', text='Kênh', command=lambda: self._sort_tree_column('channel', False))
        self.video_tree.column('channel', width=150, stretch=False)
        self.video_tree.heading('views', text='Lượt xem', command=lambda: self._sort_tree_column('views', True))
        self.video_tree.column('views', width=80, stretch=False, anchor='e')
        self.video_tree.heading('likes', text='Thích', command=lambda: self._sort_tree_column('likes', True))
        self.video_tree.column('likes', width=80, stretch=False, anchor='e')
        self.video_tree.heading('comments', text='Bình luận', command=lambda: self._sort_tree_column('comments', True))
        self.video_tree.column('comments', width=80, stretch=False, anchor='e')
        self.video_tree.heading('date', text='Ngày đăng', command=lambda: self._sort_tree_column('date', True))
        self.video_tree.column('date', width=100, stretch=False, anchor='center')
        # --- Hết phần tinh chỉnh Headers ---

        self.video_tree.grid(row=1, column=0, sticky="nsew")
        v_scroll = ctk.CTkScrollbar(list_frame, command=self.video_tree.yview)
        v_scroll.grid(row=1, column=1, sticky='ns')
        self.video_tree.configure(yscrollcommand=v_scroll.set)

        # Context Menu (Cập nhật)
        self.context_menu = Menu(self, tearoff=0, background=bg_color, foreground=text_color)
        self.context_menu.add_command(label="Tải Video Đã Chọn", command=self.start_download)
        self.context_menu.add_command(label="Tải Phụ đề SRT...", command=self._prompt_for_subtitles)
        self.context_menu.add_command(label="Tải Nội dung (TXT)", command=self._start_content_download)
        self.context_menu.add_separator()
        # --- Thêm tính năng Thumbnail ---
        self.context_menu.add_command(label="Xem Thumbnail", command=self._review_thumbnail)
        self.context_menu.add_command(label="Tải Thumbnail...", command=self._download_thumbnail)
        self.context_menu.add_separator()
        # ---
        self.context_menu.add_command(label="Copy Description", command=self._copy_description)
        self.context_menu.add_command(label="Copy Keywords", command=self._copy_keywords)
        self.context_menu.add_command(label="Copy Video URL", command=self._copy_video_url)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Mở URL trong trình duyệt", command=self._open_selected_url)
        
        self.video_tree.bind("<Button-3>", self._show_context_menu)
        self.video_tree.bind("<Double-1>", self._on_double_click)

        # --- Khung dưới cùng (BOTTOM) ---
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.grid(row=2, column=0, padx=10, pady=(5,10), sticky="nsew") # Dòng này weight=0
        bottom_frame.grid_columnconfigure(0, weight=1)

        # Đường dẫn lưu
        path_frame = ctk.CTkFrame(bottom_frame)
        path_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        path_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(path_frame, text="Chọn Thư Mục...", command=self.select_output_folder).grid(row=0, column=0, padx=(0,5))
        ctk.CTkEntry(path_frame, textvariable=self.output_path_var, state="readonly").grid(row=0, column=1, sticky="ew")
        ctk.CTkButton(path_frame, text="Mở Thư Mục", command=self.open_output_folder).grid(row=0, column=2, padx=5)

        # Hành động tải
        dl_actions_frame = ctk.CTkFrame(bottom_frame)
        dl_actions_frame.grid(row=1, column=0, sticky="ew")
        dl_actions_frame.grid_columnconfigure(1, weight=1) # Progress bar co giãn
        self.total_videos_label = ctk.CTkLabel(dl_actions_frame, textvariable=self.total_videos_var, anchor="w")
        self.total_videos_label.pack(side="left", padx=5)
        self.progress_bar = ctk.CTkProgressBar(dl_actions_frame)
        self.progress_bar.set(0)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(10,5))
        self.start_download_button = ctk.CTkButton(dl_actions_frame, text="Tải Video Đã Chọn", height=30, font=ctk.CTkFont(size=14, weight="bold"), command=self.start_download)
        self.start_download_button.pack(side="left", padx=(0,5))
        quality_menu = ctk.CTkOptionMenu(dl_actions_frame, variable=self.quality_var, values=["Best", "4K", "2K", "1080p", "720p"], width=100)
        quality_menu.pack(side="left", padx=(0, 5))
        ctk.CTkLabel(dl_actions_frame, text="Chất lượng:").pack(side="left") # Rút gọn text

        # Log Textbox
        self.log_textbox = ctk.CTkTextbox(bottom_frame, state="disabled", wrap="word", font=("Courier New", 11), height=100) # Giảm chiều cao một chút
        self.log_textbox.grid(row=2, column=0, sticky="nsew", pady=(10,0))

        # --- Tải danh sách link lần đầu ---
        self._refresh_link_lists() # Load link lists

    # ========================================================================
    # ===== HẾT HÀM create_widgets ===========================================
    # ========================================================================

    # --- HÀM MỚI CHO QUẢN LÝ PROXY ---
    def _load_proxies(self):
        """Đọc file proxies.txt và cập nhật OptionMenu."""
        self.proxy_list = ["Kết nối trực tiếp"] # Luôn có lựa chọn này
        try:
            if self.proxy_file_path.exists():
                with open(self.proxy_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        proxy_str = line.strip()
                        if proxy_str and not proxy_str.startswith('#'): # Bỏ qua dòng trống và comment
                            self.proxy_list.append(proxy_str)
            else:
                 # Tạo file nếu chưa có
                 with open(self.proxy_file_path, 'w', encoding='utf-8') as f:
                     f.write("# Định dạng: IP:port hoặc IP:port:user:pass (mỗi proxy một dòng)\n")
                     f.write("# Ví dụ: 127.0.0.1:8080\n")
                     f.write("# Ví dụ: 192.168.1.1:1080:myuser:mypass\n")
                 self.log_message(f"ℹ️ Đã tạo file proxy mặc định: {self.proxy_file_path}")

            # Cập nhật OptionMenu
            current_selection = self.selected_proxy_var.get()
            self.proxy_menu.configure(values=self.proxy_list)

            # Giữ lại lựa chọn cũ nếu nó vẫn còn trong danh sách mới
            if current_selection in self.proxy_list:
                self.selected_proxy_var.set(current_selection)
            else:
                self.selected_proxy_var.set(self.proxy_list[0]) # Mặc định chọn "Kết nối trực tiếp"

            self.log_message(f"✅ Đã làm mới danh sách proxy ({len(self.proxy_list)-1} proxies).")

        except Exception as e:
            self.log_message(f"❌ Lỗi khi tải file proxy: {e}")
            self.proxy_menu.configure(values=["[Lỗi tải proxies]"])
            self.selected_proxy_var.set("[Lỗi tải proxies]")

    def _open_proxy_file(self):
        """Mở file proxies.txt bằng trình soạn thảo mặc định."""
        if not self.proxy_file_path.exists():
            self._load_proxies() # Tạo file nếu chưa có
            if not self.proxy_file_path.exists(): # Vẫn lỗi?
                self.log_message(f"❌ Không thể tạo hoặc tìm thấy file proxy: {self.proxy_file_path}")
                return

        try:
            self.log_message(f"Đang mở file proxy: {self.proxy_file_path}")
            if sys.platform == "win32":
                os.startfile(self.proxy_file_path)
            elif sys.platform == "darwin":
                subprocess.call(["open", self.proxy_file_path])
            else:
                subprocess.call(["xdg-open", self.proxy_file_path])
            self.log_message("ℹ️ Sau khi sửa file proxy, nhấn nút 'Làm mới' để cập nhật danh sách.")
        except Exception as e:
            self.log_message(f"❌ Lỗi khi mở file proxy: {e}")

    def _get_formatted_proxy(self) -> Optional[str]:
        """Lấy proxy được chọn và định dạng lại cho yt-dlp."""
        selected = self.selected_proxy_var.get()
        if selected == "Kết nối trực tiếp" or not selected or selected.startswith("["):
            return None # Không dùng proxy

        parts = selected.split(':')
        if len(parts) < 2:
            self.log_message(f"⚠️ Định dạng proxy không hợp lệ: {selected}. Bỏ qua proxy.")
            return None # Định dạng sai

        ip = parts[0].strip()
        port = parts[1].strip()
        user = None
        password = None

        if len(parts) == 4:
            user = parts[2].strip()
            password = parts[3].strip()

        if user and password:
            # Mặc định dùng http, yt-dlp sẽ tự xử lý https nếu cần
            proxy_url = f"http://{user}:{password}@{ip}:{port}"
        elif user and not password:
             self.log_message(f"⚠️ Proxy có user nhưng thiếu password: {selected}. Bỏ qua proxy.")
             return None # Thiếu pass
        else:
            proxy_url = f"http://{ip}:{port}"

        return proxy_url

    # --- KẾT THÚC HÀM QUẢN LÝ PROXY ---

    # --- HÀM MỚI QUẢN LÝ CACHE THUMBNAIL ---

    def _cleanup_old_cache(self):
        """Xóa các file thumbnail cũ hơn CACHE_DURATION_DAYS ngày."""
        if not self.cache_path.exists():
            return
        
        cutoff_time = time.time() - (CACHE_DURATION_DAYS * 24 * 60 * 60)
        cleaned_count = 0
        try:
            for f in self.cache_path.glob("*.jpg"):
                if f.is_file():
                    if os.path.getmtime(f) < cutoff_time:
                        os.remove(f)
                        cleaned_count += 1
            if cleaned_count > 0:
                self.log_message(f"ℹ️ [Cache] Đã dọn dẹp {cleaned_count} thumbnail cũ.")
        except Exception as e:
            self.log_message(f"❌ [Cache] Lỗi khi dọn dẹp cache: {e}")

    def _download_and_cache_thumbnail(self, iid: str) -> Optional[str]:
        """
        Lấy thumbnail từ cache. Nếu không có hoặc hết hạn, tải về và cache lại.
        Trả về đường dẫn file local (str) hoặc None nếu thất bại.
        """
        item_data = self.tree_item_map.get(iid)
        if not item_data:
            self.log_message("❌ [Cache] Lỗi: Không tìm thấy dữ liệu cho item.")
            return None
        
        video_id = item_data.get('id')
        thumb_url = item_data.get('thumbnail')

        if not video_id or not thumb_url:
            self.log_message("ℹ️ [Cache] Video này không có thông tin ID hoặc thumbnail URL.")
            return None
        
        cache_file_path = self.cache_path / f"{video_id}.jpg"
        cutoff_time = time.time() - (CACHE_DURATION_DAYS * 24 * 60 * 60)

        # 1. Kiểm tra cache
        if cache_file_path.exists():
            try:
                if os.path.getmtime(cache_file_path) < cutoff_time:
                    self.log_message(f"ℹ️ [Cache] Thumbnail cho {video_id} đã cũ, đang tải lại...")
                    os.remove(cache_file_path)
                else:
                    self.log_message(f"ℹ️ [Cache] Đã tìm thấy thumbnail trong cache: {cache_file_path.name}")
                    return str(cache_file_path)
            except Exception as e:
                self.log_message(f"❌ [Cache] Lỗi khi kiểm tra file cache: {e}")
                # Thử xóa file lỗi và tải lại
                try: os.remove(cache_file_path)
                except: pass

        # 2. Tải về nếu không có cache
        self.log_message(f"ℹ️ [Cache] Đang tải thumbnail từ: {thumb_url}")
        
        try:
            proxy_url = self._get_formatted_proxy()
            handlers = []
            if proxy_url:
                # urllib cần proxy ở định dạng http://... hoặc https://...
                # Hàm _get_formatted_proxy đã trả về đúng định dạng (ví dụ: http://ip:port)
                proxy_dict = {'http': proxy_url, 'https': proxy_url}
                proxy_handler = urllib.request.ProxyHandler(proxy_dict)
                handlers.append(proxy_handler)
            
            opener = urllib.request.build_opener(*handlers)
            # Thêm User-Agent để tránh bị block (lỗi 403 Forbidden)
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')]
            urllib.request.install_opener(opener)

            with urllib.request.urlopen(thumb_url) as response, open(cache_file_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            
            self.log_message(f"ℹ️ [Cache] Đã lưu thumbnail vào cache: {cache_file_path.name}")
            return str(cache_file_path)

        except Exception as e:
            self.log_message(f"❌ [Cache] Lỗi khi tải thumbnail: {e}")
            # Xóa file rác nếu tải lỗi
            if cache_file_path.exists():
                try: os.remove(cache_file_path)
                except: pass
            return None

    def _review_thumbnail(self):
        """Hàm gọi từ context menu để xem thumbnail."""
        selected_iids = self.video_tree.selection()
        if not selected_iids:
            self.log_message("⚠️ Vui lòng chọn một video.")
            return
        iid = selected_iids[0]
        
        self.log_message("Đang chuẩn bị xem thumbnail...")
        # Chạy trong thread để không block UI khi tải
        threading.Thread(target=self._review_thumbnail_worker, args=(iid,), daemon=True).start()

    def _review_thumbnail_worker(self, iid):
        """Worker thread để lấy và mở thumbnail."""
        thumb_path = self._download_and_cache_thumbnail(iid)
        
        if thumb_path:
            try:
                self.log_message(f"Đang mở thumbnail: {thumb_path}")
                if sys.platform == "win32": os.startfile(thumb_path)
                elif sys.platform == "darwin": subprocess.call(["open", thumb_path])
                else: subprocess.call(["xdg-open", thumb_path])
            except Exception as e:
                self.log_message(f"❌ Lỗi khi mở thumbnail: {e}")
        else:
            self.log_message(f"❌ Không thể lấy thumbnail để xem.")

    def _download_thumbnail(self):
        """Hàm gọi từ context menu để tải thumbnail về máy."""
        selected_iids = self.video_tree.selection()
        if not selected_iids:
            self.log_message("⚠️ Vui lòng chọn một video.")
            return
        iid = selected_iids[0]
        item_data = self.tree_item_map.get(iid)
        if not item_data: return

        # Tạo tên file gợi ý
        safe_title = sanitize_filename(item_data.get('title', 'thumbnail'))
        video_id = item_data.get('id', 'default')
        suggested_filename = f"{safe_title} [{video_id}].jpg"

        # Hỏi người dùng lưu file ở đâu
        save_path = filedialog.asksaveasfilename(
            title="Lưu Thumbnail",
            initialfile=suggested_filename,
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )

        if not save_path:
            self.log_message("ℹ️ Đã hủy thao tác tải thumbnail.")
            return

        self.log_message(f"Đang chuẩn bị tải thumbnail về {save_path}...")
        # Chạy trong thread để không block UI
        threading.Thread(target=self._download_thumbnail_worker, args=(iid, save_path), daemon=True).start()

    def _download_thumbnail_worker(self, iid, save_path):
        """Worker thread để lấy thumbnail từ cache và copy ra vị trí lưu."""
        thumb_path_from_cache = self._download_and_cache_thumbnail(iid)
        
        if thumb_path_from_cache:
            try:
                shutil.copy2(thumb_path_from_cache, save_path)
                self.log_message(f"✅ Đã tải thumbnail thành công về: {save_path}")
            except Exception as e:
                self.log_message(f"❌ Lỗi khi lưu thumbnail: {e}")
        else:
            self.log_message(f"❌ Không thể tải thumbnail.")

    # --- KẾT THÚC HÀM QUẢN LÝ CACHE THUMBNAIL ---


    # --- CÁC HÀM MỚI ĐỂ QUẢN LÝ LIST LINK ---

    def _refresh_link_lists(self):
        """Quét thư mục ManageLink và cập nhật OptionMenu."""
        self.link_lists = {}
        try:
            # Sắp xếp file theo tên
            files = sorted(self.manage_link_path.glob("*.txt"), key=lambda f: f.stem)
            for f in files:
                self.link_lists[f.stem] = f

            names = list(self.link_lists.keys())

            if not names:
                names = ["[Không có list nào]"]
                self.link_list_menu.configure(state="disabled", values=names)
                self.delete_list_button.configure(state="disabled")
                self.link_list_var.set(names[0])
            else:
                current_val = self.link_list_var.get()
                self.link_list_menu.configure(state="normal", values=names)
                self.delete_list_button.configure(state="normal")
                # Giữ giá trị đang chọn nếu nó vẫn tồn tại
                if current_val not in names and current_val != "[Chọn list link]": # Đừng chọn lại nếu giá trị là placeholder
                     self.link_list_var.set(names[0]) # Mặc định chọn cái đầu tiên nếu list cũ bị xóa

            # Cập nhật lại giá trị hiển thị trên menu nếu biến đã thay đổi
            self.link_list_menu.set(self.link_list_var.get())


        except Exception as e:
            self.log_message(f"❌ Lỗi khi làm mới danh sách link: {e}")
            self.link_list_menu.configure(state="disabled", values=["[Lỗi tải list]"])
            self.link_list_var.set("[Lỗi tải list]")

    def _load_link_list(self, selected_name: str):
        """Tải nội dung của file .txt đã chọn vào textbox."""
        if selected_name in self.link_lists:
            try:
                filepath = self.link_lists[selected_name]
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.url_textbox.delete("1.0", "end")
                self.url_textbox.insert("1.0", content)
                self.log_message(f"ℹ️ Đã tải list: {selected_name}")
                self.link_list_var.set(selected_name) # Đảm bảo biến được cập nhật
            except Exception as e:
                self.log_message(f"❌ Lỗi khi tải list '{selected_name}': {e}")
        elif selected_name == "[Không có list nào]" or selected_name == "[Chọn list link]" or selected_name == "[Lỗi tải list]":
             self.url_textbox.delete("1.0", "end") # Xóa nội dung nếu chọn các mục placeholder

    def _save_link_list(self):
        """Lưu nội dung textbox hiện tại vào một file .txt."""
        content = self.url_textbox.get("1.0", "end-1c").strip()
        if not content:
            self.log_message("⚠️ Không có link nào để lưu.")
            return

        dialog = ctk.CTkInputDialog(text="Nhập tên cho List Link này:", title="Lưu List Link")
        list_name = dialog.get_input()

        if not list_name:
            return # User canceled

        safe_name = sanitize_filename(list_name.strip(), replace_with=" ") # Thêm strip()
        if not safe_name:
            self.log_message("⚠️ Tên list không hợp lệ.")
            return

        # Kiểm tra trùng tên
        is_overwrite = False
        filepath = self.manage_link_path / f"{safe_name}.txt"
        if filepath.exists():
             confirm_overwrite = messagebox.askyesno("Xác nhận Ghi đè", f"List '{safe_name}' đã tồn tại.\nBạn có muốn ghi đè không?")
             if not confirm_overwrite:
                 return
             is_overwrite = True

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            log_msg = f"✅ Đã {'ghi đè' if is_overwrite else 'lưu'} list: {safe_name}.txt"
            self.log_message(log_msg)
            # Chỉ cập nhật var và refresh nếu tên mới hoặc ghi đè thành công
            self.link_list_var.set(safe_name)
            self._refresh_link_lists() # Tải lại danh sách

        except Exception as e:
            self.log_message(f"❌ Lỗi khi lưu list '{safe_name}': {e}")

    def _delete_link_list(self):
        """Xóa file .txt của list đang chọn."""
        selected_name = self.link_list_var.get()
        if selected_name not in self.link_lists:
            self.log_message("⚠️ Không có list nào được chọn để xóa.")
            return

        confirm = messagebox.askyesno("Xác nhận Xóa", f"Bạn có chắc chắn muốn xóa list: '{selected_name}' không?\nThao tác này không thể hoàn tác.")

        if not confirm:
            return

        try:
            filepath = self.link_lists[selected_name]
            os.remove(filepath)
            self.log_message(f"✅ Đã xóa list: {selected_name}")
            # Sau khi xóa, đặt lại giá trị mặc định và xóa textbox
            self.link_list_var.set("[Chọn list link]")
            self.url_textbox.delete("1.0", "end")
            self._refresh_link_lists() # Tải lại danh sách

        except Exception as e:
            self.log_message(f"❌ Lỗi khi xóa list '{selected_name}': {e}")

    # --- KẾT THÚC CÁC HÀM QUẢN LÝ LIST LINK ---

    # def _get_download_format(self):
    #     quality = self.quality_var.get()
    #     if quality == "Best": return 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
    #     resolutions = {"4K": "2160", "2K": "1440", "1080p": "1080", "720p": "720"}
    #     height = resolutions.get(quality, "1080")
    #     return f'bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={height}][ext=mp4]/best[ext=mp4]/best'

    def _get_download_format(self):
        """
        Tạo chuỗi định dạng linh hoạt hơn cho yt-dlp để tránh lỗi "format not available".
        Hàm này sẽ ưu tiên chất lượng tốt nhất, bất kể container (mp4/webm),
        và dựa vào 'merge_output_format' để có được file mp4 cuối cùng.
        """
        quality = self.quality_var.get()
        # Chuỗi định dạng chung: tải video tốt nhất + audio tốt nhất, sau đó fallback về file gộp sẵn tốt nhất.
        base_format = 'bestvideo+bestaudio/best'

        if quality == "Best":
            return base_format

        resolutions = {"4K": "2160", "2K": "1440", "1080p": "1080", "720p": "720"}
        height = resolutions.get(quality, "1080")
        # Thêm điều kiện về chiều cao vào chuỗi định dạng chung.
        return f'bestvideo[height<={height}]+bestaudio/best[height<={height}]'
    
    def _start_scan(self):
        if self.scan_process and self.scan_process.is_alive():
            self.log_message("!!! Đang quét. Vui lòng đợi.")
            return
        urls_text = self.url_textbox.get("1.0", "end-1c").strip()
        raw_urls = [url.strip() for url in re.split(r'[\n,]', urls_text) if url.strip()]
        unique_urls = list(dict.fromkeys(raw_urls))
        if not unique_urls: self.log_message("⚠️ Vui lòng nhập URL."); return

        # --- LOGIC MỞ RỘNG URL KÊNH ---
        final_urls_to_scan = []
        for url in unique_urls:
            # Kiểm tra xem có phải là URL kênh YouTube (handle, /c/, /user/)
            is_yt_channel = "youtube.com" in url and \
                            (('/@' in url or '/c/' in url or '/user/' in url) or \
                             (not 'watch?v=' in url and not '/playlist?list=' in url and not '/shorts/' in url and not '/videos' in url))

            # Kiểm tra xem có phải là URL kênh TikTok (handle)
            is_tiktok_channel = "tiktok.com" in url and \
                                (('/@' in url) and \
                                 (not '/video/' in url))

            if is_yt_channel:
                self.log_message(f"ℹ️ Phát hiện kênh YouTube: {url}. Tự động quét /videos và /shorts.")
                url_base = url.rstrip('/')
                final_urls_to_scan.append(f"{url_base}/videos")
                final_urls_to_scan.append(f"{url_base}/shorts")
            elif is_tiktok_channel:
                 self.log_message(f"ℹ️ Phát hiện kênh TikTok: {url}. Đang quét kênh.")
                 final_urls_to_scan.append(url) # yt-dlp tự xử lý kênh TikTok
            else:
                # Là link video đơn, playlist, hoặc đã là tab cụ thể
                final_urls_to_scan.append(url)
        # --- KẾT THÚC LOGIC MỞ RỘNG ---

        if not final_urls_to_scan:
            self.log_message("⚠️ Không tìm thấy URL hợp lệ để quét."); return

        self._clear_list()
        self.log_message(f"UI: Bắt đầu quét {len(final_urls_to_scan)} nguồn (đã mở rộng)...")
        self.scan_button.configure(state="disabled", text="Đang khởi tạo...")
        try:
            thread_count = int(self.thread_count_var.get())
            if thread_count <= 0: thread_count = 1
        except ValueError:
            thread_count = 4; self.log_message("⚠️ Số luồng không hợp lệ, dùng mặc định (4)."); self.thread_count_var.set("4")
        while not self.log_queue.empty(): self.log_queue.get()
        while not self.detail_queue.empty(): self.detail_queue.get()

        # Lấy proxy đã định dạng
        formatted_proxy = self._get_formatted_proxy()

        # Gửi danh sách URL đã mở rộng và proxy đã định dạng
        self.scan_process = mp.Process(target=scan_worker_process, args=(final_urls_to_scan, formatted_proxy, self.detail_queue, self.log_queue, thread_count), daemon=True)
        self.scan_process.start()

    # --- Các hàm còn lại giữ nguyên (từ _append_to_list đến hết) ---
    def _append_to_list(self, entries):
        start_index = len(self.tree_item_map)
        for i, entry in enumerate(entries):
            if not entry: continue
            title = entry.get('title') or entry.get('fulltitle') or entry.get('id') or 'N/A'
            iid = str(start_index + i)
            video_type = entry.get('video_type', 'Video') # Lấy loại video
            # Thêm placeholder cho cột type và language
            values = (start_index + i + 1, "Đang quét...", "", title, video_type, "...", "...", "...", "...", "...", "...", "...")
            try:
                self.video_tree.insert('', 'end', iid=iid, values=values)
                self.tree_item_map[iid] = {
                    '_entry': entry, 'index': start_index + i + 1, 'title': title, 'status': 'pending',
                    'id': entry.get('id'), 'webpage_url': entry.get('url'),
                    'video_type': video_type # Lưu lại loại video
                }
            except Exception as e: print(f"Lỗi chèn dòng: {e} - Data: {values}")
        self.total_videos_var.set(f"Tổng video: {len(self.tree_item_map)}")

    def _update_treeview_row(self, iid, details):
        if not self.video_tree.exists(iid): return
        self.tree_item_map[iid].update(details)
        self.tree_item_map[iid]['status'] = 'ready' # Trạng thái nội bộ vẫn là 'ready'
        
        # --- LƯU LẠI THUMBNAIL URL (QUAN TRỌNG) ---
        self.tree_item_map[iid]['thumbnail'] = details.get('thumbnail')
        # ----------------------------------------

        # --- YÊU CẦU: Tinh chỉnh Text cột Content Status (Sub) ---
        content_status_text = "Không có" # Mặc định
        is_tiktok = details.get('extractor_key', '').lower() == 'tiktok'
        if not is_tiktok:
            if details.get('subtitles'):
                content_status_text = "Sẵn" # Đổi từ Gốc -> Sẵn
            elif details.get('automatic_captions'):
                content_status_text = "Tự động"
        else:
            content_status_text = "N/A (TikTok)"
        # ---
        self.tree_item_map[iid]['content_status_text'] = content_status_text

        # --- YÊU CẦU: Tinh chỉnh Text cột Status ---
        status_display_text = "✅" # Chỉ hiển thị icon khi sẵn sàng (status='ready')
        # ---
        self.tree_item_map[iid]['status_text'] = status_display_text # Lưu lại text (dùng để sort)

        values = (
            self.tree_item_map[iid].get('index', ''),
            status_display_text, # Hiển thị text đã sửa
            content_status_text, # Hiển thị text đã sửa
            details.get('title', self.tree_item_map[iid].get('title', 'N/A')),
            details.get('video_type', 'Video'),
            details.get('language', 'N/A'),
            format_duration(details.get('duration')), format_size(details.get('filesize_approx')),
            details.get('channel', 'N/A'), format_number(details.get('view_count')),
            format_number(details.get('like_count')), format_number(details.get('comment_count')),
            details.get('upload_date', 'N/A')
        )
        try: self.video_tree.item(iid, values=values)
        except Exception as e: print(f"Lỗi cập nhật dòng {iid}: {e} - Values: {values}")

    def _on_double_click(self, event):
        iid = self.video_tree.focus();
        if not iid: return
        item_data = self.tree_item_map.get(iid, {}); column_id = self.video_tree.identify_column(event.x)
        content_column_index = '#3' # Index cột 'content_status'
        filepath_to_open, file_type = None, None
        if column_id == content_column_index and 'content_filepath' in item_data: filepath_to_open, file_type = item_data['content_filepath'], "nội dung TXT"
        elif column_id == content_column_index and 'subtitle_filepath' in item_data: filepath_to_open, file_type = item_data['subtitle_filepath'], "phụ đề SRT"
        elif item_data.get('status') == 'success' and 'filepath' in item_data: filepath_to_open, file_type = item_data['filepath'], "video"
        if filepath_to_open:
            if os.path.exists(filepath_to_open):
                self.log_message(f"Đang mở file {file_type}: {filepath_to_open}")
                try:
                    if sys.platform == "win32": os.startfile(filepath_to_open)
                    elif sys.platform == "darwin": subprocess.call(["open", filepath_to_open])
                    else: subprocess.call(["xdg-open", filepath_to_open])
                except Exception as e: self.log_message(f"❌ Lỗi khi mở file: {e}")
            else: self.log_message(f"⚠️ Không tìm thấy file {file_type}: {filepath_to_open}")
        elif column_id == content_column_index: self.log_message(f"ℹ️ Chưa có file nội dung/phụ đề cho video này.")

    def _open_selected_url(self):
        self._copy_video_url(log_only=True)
        try:
            url = self.clipboard_get()
            if url and url.startswith("http"):
                try: self.log_message(f"Đang mở URL: {url}"); webbrowser.open(url, new=2)
                except Exception as e: self.log_message(f"❌ Lỗi khi mở URL: {e}")
        except Exception: self.log_message(f"⚠️ Không thể lấy URL từ clipboard.")

    def _prompt_for_subtitles(self):
        iid = self.video_tree.focus();
        if not iid: self.log_message("⚠️ Vui lòng chọn một video để tải phụ đề."); return
        item_data = self.tree_item_map.get(iid)
        if not item_data or not item_data.get('webpage_url') or item_data.get('status') == 'pending': self.log_message("⚠️ Chi tiết video chưa được quét hoặc thiếu URL."); return
        url, channel_name = item_data.get('webpage_url'), item_data.get('channel', 'Unknown_Channel')
        extractor_key = item_data.get('extractor_key', '').lower()
        if extractor_key == 'tiktok': self.log_message("ℹ️ TikTok không hỗ trợ tải phụ đề."); self.status_queue.put((iid, 'content_status', "N/A (TikTok)")); return
        self.status_queue.put((iid, 'content_status', "Đang quét...")) # Rút gọn text
        threading.Thread(target=self._fetch_and_show_sub_dialog, args=(iid, url, channel_name), daemon=True).start()

    def _fetch_and_show_sub_dialog(self, iid, url, channel_name):
        try:
            self.log_message(f"Đang lấy danh sách phụ đề cho: {url}")
            item_data = self.tree_item_map.get(iid, {})
            # Tận dụng dữ liệu đã quét
            info = {
                'language': item_data.get('language'),
                'subtitles': item_data.get('subtitles'),
                'automatic_captions': item_data.get('automatic_captions'),
                'title': item_data.get('title'),
                'extractor_key': item_data.get('extractor_key')
            }

            # Lấy proxy đã định dạng
            formatted_proxy = self._get_formatted_proxy()

            if info['subtitles'] is None and info['automatic_captions'] is None:
                self.log_message("   - Dữ liệu phụ đề chưa có, đang quét lại...")
                ydl_opts = {'listsubtitles': True, 'quiet': True, 'no_warnings': True, 'fields': ['language', 'subtitles', 'automatic_captions', 'title', 'extractor_key']}
                if formatted_proxy: ydl_opts['proxy'] = formatted_proxy # Sử dụng proxy đã định dạng
                with yt_dlp.YoutubeDL(ydl_opts) as ydl: info = ydl.extract_info(url, download=False)
                self.tree_item_map[iid].update(info) # Cập nhật lại map
            sub_map = {}
            if subs := info.get('subtitles'):
                for lang_code, sub_list in subs.items():
                    if any(s.get('ext') in ['vtt', 'srt'] for s in sub_list):
                        lang_name = sub_list[0].get('name', lang_code); sub_map[f"{lang_name} ({lang_code}) (Original)"] = lang_code
            if autos := info.get('automatic_captions'):
                for lang_code, sub_list in autos.items():
                    base_lang = lang_code.split('-')[0]
                    manual_exists = any(base_lang == mc.split('-')[0] for mc in subs.keys()) if subs else False
                    if not manual_exists and any(s.get('ext') in ['vtt', 'srt'] for s in sub_list):
                        lang_name = sub_list[0].get('name', lang_code); sub_map[f"{lang_name} ({lang_code}) (Tự động)"] = lang_code
            original_language = info.get('language')
            if not sub_map:
                self.log_message(f"ℹ️ Không tìm thấy phụ đề phù hợp cho video: {info.get('title', url)}"); self.status_queue.put((iid, 'content_status', "Không có")); return # Sửa text
            self.after(0, self._show_sub_dialog, iid, url, sub_map, channel_name, original_language)
        except Exception as e:
            self.log_message(f"❌ Lỗi khi lấy danh sách phụ đề: {e}"); self.status_queue.put((iid, 'content_status', "❌ Lỗi quét")) # Sửa text

    def _show_sub_dialog(self, iid, url, sub_map, channel_name, original_language):
        dialog = SubtitleDialog(self, sub_map, original_language)
        self.wait_window(dialog)
        selected_lang = dialog.get_selection()
        if selected_lang: self._start_subtitle_download(iid, url, selected_lang, channel_name)
        else:
            # Nếu hủy dialog, phục hồi trạng thái content ban đầu
            original_content_status = self.tree_item_map.get(iid, {}).get('content_status_text', 'Không có')
            self.status_queue.put((iid, 'content_status', original_content_status))

    def _start_subtitle_download(self, iid, url, lang_code, channel_name):
        sanitized_channel_name = sanitize_filename(channel_name)
        base_path = Path(self.output_path_var.get())
        subtitle_path = base_path / "Subtitles" / sanitized_channel_name
        subtitle_path.mkdir(parents=True, exist_ok=True)
        self.status_queue.put((iid, 'content_status', f"Đang tải SRT {lang_code}..."))
        threading.Thread(target=self._subtitle_download_worker, args=(iid, url, lang_code, subtitle_path), daemon=True).start()

    def _subtitle_download_worker(self, iid, url, lang_code, output_path: Path):
        try:
            item_data = self.tree_item_map.get(iid, {})
            safe_title = sanitize_filename(item_data.get('title', 'Unknown Title'))
            video_id = item_data.get('id', 'UnknownID')

            # Lấy proxy đã định dạng
            formatted_proxy = self._get_formatted_proxy()

            ydl_opts = {
                'skip_download': True, 'writesubtitles': True, 'writeautomaticsub': True,
                'subtitleslangs': [lang_code], 'subtitlesformat': 'srt',
                'outtmpl': str(output_path / f'{safe_title} [{video_id}].%(lang)s.%(ext)s'),
                'quiet': True, 'no_warnings': True,
            }
            if formatted_proxy: ydl_opts['proxy'] = formatted_proxy # Sử dụng proxy đã định dạng
            final_sub_path_srt, download_info = None, {}
            def get_sub_filename_hook(d):
                nonlocal final_sub_path_srt, download_info
                if d['status'] == 'finished':
                    download_info = d.get('info_dict', {})
                    subs_dict = download_info.get('requested_subtitles',{})
                    if subs_dict and lang_code in subs_dict:
                        fpath = subs_dict[lang_code].get('filepath')
                        if fpath and fpath.lower().endswith('.srt'): final_sub_path_srt = fpath
            ydl_opts['progress_hooks'] = [get_sub_filename_hook]
            self.log_message(f"Đang tải SRT cho ngôn ngữ {lang_code}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
            if not final_sub_path_srt and download_info:
                base_lang_code = lang_code.split('-')[0]
                expected_filename_full = f"{safe_title} [{video_id}].{lang_code}.srt"
                expected_filename_base = f"{safe_title} [{video_id}].{base_lang_code}.srt"
                potential_path_full, potential_path_base = output_path / expected_filename_full, output_path / expected_filename_base
                if potential_path_full.exists(): final_sub_path_srt = str(potential_path_full)
                elif potential_path_base.exists(): final_sub_path_srt = str(potential_path_base)
                else:
                    for f in output_path.glob(f"*{video_id}*{lang_code}*.srt"): final_sub_path_srt = str(f); break
                    if not final_sub_path_srt:
                        for f in output_path.glob(f"*{video_id}*{base_lang_code}*.srt"): final_sub_path_srt = str(f); break
            if final_sub_path_srt and os.path.exists(final_sub_path_srt):
                self.tree_item_map[iid]['subtitle_filepath'] = final_sub_path_srt
                sub_filename = os.path.basename(final_sub_path_srt)
                self.status_queue.put((iid, 'content_status', f"✅ SRT: {sub_filename}"))
                self.log_message(f"✅ Tải SRT '{lang_code}' thành công!")
            else: raise Exception(f"Không tìm thấy file SRT đã tải cho '{lang_code}'. Đã tìm trong {output_path}")
        except Exception as e:
            self.log_message(f"❌ Lỗi khi tải phụ đề SRT '{lang_code}': {e}")
            self.status_queue.put((iid, 'content_status', "❌ Lỗi SRT")) # Sửa text

    def _start_content_download(self):
        if self.content_download_thread and self.content_download_thread.is_alive():
            self.log_message("!!! Đang có một tiến trình tải nội dung. Vui lòng đợi."); return
        selected_iids = self.video_tree.selection()
        if not selected_iids: self.log_message("⚠️ Vui lòng chọn ít nhất một video để tải nội dung."); return
        videos_to_process = []
        for iid in selected_iids:
            item_data = self.tree_item_map.get(iid, {})
            extractor_key = item_data.get('extractor_key', '').lower()
            if extractor_key == 'tiktok':
                self.log_message(f"ℹ️ Bỏ qua tải nội dung cho video TikTok: {item_data.get('title', '#'+iid)}")
                self.status_queue.put((iid, 'content_status', "N/A (TikTok)")); continue
            # Cho phép tạo lại TXT nếu video đã sẵn sàng hoặc đã thành công/lỗi
            if item_data.get('status') in ['ready', 'success', 'error'] and item_data.get('webpage_url'):
                self.status_queue.put((iid, 'content_status', '⏳ Chờ TXT')) # Sửa text
                item_data['status_content'] = 'queued'; videos_to_process.append((iid, item_data))
            else: self.log_message(f"⚠️ Video '{item_data.get('title', '#'+iid)}' chưa sẵn sàng, đang quét hoặc đang chờ, bỏ qua.")
        if not videos_to_process: return
        if not self.output_path_var.get(): self.log_message("⚠️ Vui lòng chọn thư mục lưu."); return
        self.progress_bar.set(0)
        output_path = Path(self.output_path_var.get())
        self.content_download_thread = threading.Thread(target=self._content_download_worker, args=(videos_to_process, output_path), daemon=True)
        self.content_download_thread.start()

    def _content_download_worker(self, videos_to_process, base_output_path: Path):
        self.log_message(f"Bắt đầu tạo {len(videos_to_process)} file nội dung TXT...")
        total, completed = len(videos_to_process), 0
        for i, (iid, video_info) in enumerate(videos_to_process):
            txt_filepath_result = None # Đổi tên biến để rõ ràng hơn
            try:
                url, channel_name = video_info.get('webpage_url'), video_info.get('channel', 'Unknown')
                video_title, video_id = video_info.get('title', 'Unknown'), video_info.get('id', 'Unknown')
                if not url: raise ValueError("URL không hợp lệ")
                self.status_queue.put((iid, 'content_status', 'Đang xử lý TXT...'))
                txt_filepath_result = self._download_vtt_and_parse_to_txt(
                    url=url, video_title=video_title, video_id=video_id,
                    channel_name=channel_name, base_output_path=base_output_path,
                    video_info_map=video_info # Truyền toàn bộ thông tin đã quét
                )
                if isinstance(txt_filepath_result, str) and os.path.exists(txt_filepath_result): # Thành công, trả về path
                    self.tree_item_map[iid]['content_filepath'] = txt_filepath_result
                    txt_filename = os.path.basename(txt_filepath_result)
                    self.status_queue.put((iid, 'content_status', f"✅ TXT: {txt_filename}"))
                    self.tree_item_map[iid]['status_content'] = 'success'
                elif txt_filepath_result is False: # Không có phụ đề
                    self.status_queue.put((iid, 'content_status', 'Không có')); # Sửa text
                    self.tree_item_map[iid]['status_content'] = 'no_sub'
                else: # Lỗi (trả về None)
                    self.status_queue.put((iid, 'content_status', '❌ Lỗi TXT'));
                    self.tree_item_map[iid]['status_content'] = 'error'
            except Exception as e:
                self.log_message(f"--- ❌ Lỗi khi xử lý TXT cho video #{i+1}: {video_info.get('title', 'N/A')} - {e} ---")
                self.status_queue.put((iid, 'content_status', "❌ Lỗi TXT"))
                if iid in self.tree_item_map: self.tree_item_map[iid]['status_content'] = 'error'
            finally:
                completed += 1; progress = completed / total; self.after(0, self.progress_bar.set, progress)
        self.log_message("\n📄📄📄 Đã xử lý xong tất cả nội dung TXT! 📄📄📄")

    def _download_vtt_and_parse_to_txt(self, url: str, video_title:str, video_id:str, channel_name: str, base_output_path: Path, video_info_map: Dict[str, Any]) -> Optional[str or bool]: # Sửa kiểu trả về
        temp_sub_dir = tempfile.mkdtemp(prefix="txt_sub_")
        target_lang, vtt_filepath = None, None
        try:
            self.log_message(f"   - Lấy thông tin phụ đề cho video ID: {video_id}")
            # Dùng dữ liệu đã quét, không gọi API nữa
            manual_subs = video_info_map.get('subtitles', {})
            auto_subs = video_info_map.get('automatic_captions', {})
            available_lang_codes = set(manual_subs.keys()) | set(auto_subs.keys())

            if not available_lang_codes:
                 self.log_message(f"   - ℹ️ Không tìm thấy phụ đề (thủ công hoặc tự động).")
                 return False # Trả về False để worker biết là không có sub, không phải lỗi

            # Priority 1: Find any 'Original' manual subtitle
            for lang_code in manual_subs.keys():
                if '-orig' in lang_code or any('(original)' in sub.get('name', '').lower() for sub in manual_subs[lang_code]):
                    target_lang = lang_code; self.log_message(f"   - Tìm thấy 'Original': '{target_lang}'."); break
            # Priority 2: Use video's main language
            if not target_lang:
                main_lang = video_info_map.get('language')
                if main_lang and main_lang in available_lang_codes: target_lang = main_lang; self.log_message(f"   - Sử dụng ngôn ngữ chính: '{target_lang}'")
                elif main_lang and main_lang.split('-')[0] in available_lang_codes: target_lang = main_lang.split('-')[0]; self.log_message(f"   - Sử dụng ngôn ngữ chính (base): '{target_lang}'")
            # Priority 3 & 4: Fallback to vi, then en
            if not target_lang:
                if 'vi' in available_lang_codes: target_lang = 'vi'
                elif 'en' in available_lang_codes: target_lang = 'en'
                if target_lang: self.log_message(f"   - Sử dụng fallback: '{target_lang}'")
            if not target_lang:
                self.log_message(f"   - ℹ️ Không tìm thấy phụ đề phù hợp.");
                return False # Trả về False

            self.log_message(f"   - Đang tải VTT cho ngôn ngữ: '{target_lang}'...")
            temp_vtt_outtmpl = os.path.join(temp_sub_dir, f"{video_id or 'temp'}.%(ext)s")

            # Lấy proxy đã định dạng
            formatted_proxy = self._get_formatted_proxy()

            opts_sub = {
                'quiet': True, 'no_warnings': True, 'skip_download': True, 'writesubtitles': True,
                'writeautomaticsub': True, 'subtitleslangs': [target_lang], 'subtitlesformat': 'vtt',
                'outtmpl': temp_vtt_outtmpl.replace('.vtt', ''),
            }
            if formatted_proxy: opts_sub['proxy'] = formatted_proxy # Sử dụng proxy đã định dạng

            with yt_dlp.YoutubeDL(opts_sub) as ydl_sub:
                ydl_sub.extract_info(url, download=True) # download=True để tải sub

            downloaded_files = os.listdir(temp_sub_dir)
            if not downloaded_files: raise DownloadError(f"Tải VTT cho '{target_lang}' thất bại.")
            vtt_filepath = os.path.join(temp_sub_dir, downloaded_files[0])
            self.log_message(f"   - Đã tải VTT: {downloaded_files[0]}")

            # Gọi hàm phân tích dựa trên code bạn cung cấp
            clean_text = _parse_vtt_file_to_clean_text(vtt_filepath)

            if clean_text:
                sanitized_channel = sanitize_filename(channel_name)
                txt_output_dir = base_output_path / "TXT_Content" / sanitized_channel
                txt_output_dir.mkdir(parents=True, exist_ok=True)
                safe_video_title = sanitize_filename(video_title if video_title else video_id)
                txt_filename = f"{safe_video_title}.txt"
                txt_filepath_obj = txt_output_dir / txt_filename
                with open(txt_filepath_obj, 'w', encoding='utf-8') as f_txt:
                    f_txt.write(clean_text)
                self.log_message(f"   - ✅ Đã tạo file TXT: {str(txt_filepath_obj)}")
                return str(txt_filepath_obj) # Trả về đường dẫn file TXT
            else:
                self.log_message(f"   - ⚠️ Lỗi: Không thể phân tích nội dung từ file VTT: {vtt_filepath}");
                return None # Trả về None nếu lỗi phân tích
        except DownloadError as e:
            self.log_message(f"   - ❌ Lỗi DownloadError khi xử lý TXT: {e}");
            return None # Trả về None nếu lỗi tải VTT
        except Exception as e:
            self.log_message(f"   - ❌ Lỗi chung khi xử lý TXT: {type(e).__name__} - {e}");
            return None # Trả về None nếu lỗi khác
        finally:
            if os.path.exists(temp_sub_dir): shutil.rmtree(temp_sub_dir)

    def start_download(self):
        if self.download_thread and self.download_thread.is_alive(): self.log_message("!!! Đang tải. Vui lòng đợi."); return
        selected_iids = self.video_tree.selection()
        if not selected_iids: self.log_message("⚠️ Vui lòng chọn video."); return
        videos_to_download = []
        for iid in selected_iids:
            item_data = self.tree_item_map.get(iid, {})
            # Cho phép tải lại cả video đã thành công/lỗi
            if item_data.get('status') in ['ready', 'success', 'error']:
                self.status_queue.put((iid, 'status', '⏳ Chờ tải'));
                item_data['status'] = 'queued'; videos_to_download.append((iid, item_data))
            else: self.log_message(f"⚠️ Video '{item_data.get('title', '#'+iid)}' đang quét hoặc đã trong hàng chờ, bỏ qua.")
        if not videos_to_download: return
        if not self.output_path_var.get(): self.log_message("⚠️ Vui lòng chọn thư mục lưu."); return
        self.start_download_button.configure(state="disabled", text="Đang Tải...")
        self.progress_bar.set(0)
        self.download_thread = threading.Thread(target=self._download_worker, args=(videos_to_download, Path(self.output_path_var.get())), daemon=True)
        self.download_thread.start()

    def _download_worker(self, videos_to_download, output_path: Path):
        self.log_message(f"Bắt đầu tải {len(videos_to_download)} video...")
        for i, (iid, video) in enumerate(videos_to_download):
            try: self._download_single_video(iid, video, output_path, i, len(videos_to_download))
            except Exception as e:
                self.log_message(f"--- ❌ Lỗi khi tải video #{i+1}/{len(videos_to_download)}: {video.get('title', 'N/A')} - {e} ---")
                self.status_queue.put((iid, 'status', "❌ Lỗi"))
                if iid in self.tree_item_map: self.tree_item_map[iid]['status'] = 'error'
        self.log_message("\n🎉🎉🎉 Đã tải xong tất cả các video đã chọn! 🎉🎉🎉")
        self.after(0, self.on_download_finished)

    def _download_single_video(self, iid, video_info, output_path, index, total):
        url = video_info.get('webpage_url')
        if not url: raise ValueError("URL không hợp lệ")
        self.log_message(f"\n--- 📥 Bắt đầu tải video #{index+1}/{total}: {video_info.get('title', url)} ---")

        # Biến cục bộ để lưu đường dẫn file từ hook
        final_filepath_from_hook = None

        def my_hook(d):
            nonlocal final_filepath_from_hook
            if d['status'] == 'downloading':
                # Logic thanh tiến trình (không đổi)
                total_bytes_est = d.get('total_bytes_estimate'); total_bytes = d.get('total_bytes', total_bytes_est)
                downloaded_bytes = d.get('downloaded_bytes', 0)
                if total_bytes and total_bytes > 0:
                    percent = downloaded_bytes / total_bytes * 100
                    self.after(0, self.progress_bar.set, downloaded_bytes / total_bytes)
                    self.status_queue.put((iid, 'status', f"Đang tải {percent:.1f}%"))

            elif d['status'] == 'finished':
                fpath = d.get('info_dict', {}).get('filepath')
                if fpath:
                    final_filepath_from_hook = fpath
                self.status_queue.put((iid, 'status', "Đang xử lý..."))

            elif d['status'] == 'error':
                self.log_message(f"--- ⚠️ Lỗi hook yt-dlp cho video #{index+1} ---")

        # --- Thiết lập đường dẫn và tùy chọn (không đổi) ---
        channel_name = video_info.get('channel', 'Unknown_Channel'); sanitized_channel = sanitize_filename(channel_name)
        video_output_dir = output_path / "Videos" / sanitized_channel
        video_output_dir.mkdir(parents=True, exist_ok=True)
        safe_title = sanitize_filename(video_info.get('title', 'Unknown Title')); video_id = video_info.get('id', 'UnknownID')

        # --- Xác định các đường dẫn file cuối cùng CÓ THỂ CÓ ---
        expected_final_path_mp4 = video_output_dir / f"{safe_title} [{video_id}].mp4"
        expected_final_path_mkv = video_output_dir / f"{safe_title} [{video_id}].mkv"
        expected_final_path_webm = video_output_dir / f"{safe_title} [{video_id}].webm"

        # Lấy proxy đã định dạng
        formatted_proxy = self._get_formatted_proxy()

        ydl_opts = {
            'outtmpl': str(video_output_dir / f'{safe_title} [{video_id}].%(ext)s'),
            'progress_hooks': [my_hook], 'quiet': True, 'no_warnings': True,
            'ignoreerrors': True, 'noprogress': True, 'format': self._get_download_format()
        }
        if formatted_proxy: ydl_opts['proxy'] = formatted_proxy # Sử dụng proxy đã định dạng

        # --- Bắt đầu tải ---
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])

        # --- Logic kiểm tra file sau khi tải (Đã sửa đổi) ---
        final_filepath_to_check = final_filepath_from_hook
        if not final_filepath_to_check or not os.path.exists(final_filepath_to_check):
            if expected_final_path_mp4.exists():
                final_filepath_to_check = str(expected_final_path_mp4)
            elif expected_final_path_mkv.exists():
                final_filepath_to_check = str(expected_final_path_mkv)
            elif expected_final_path_webm.exists():
                final_filepath_to_check = str(expected_final_path_webm)
            else:
                pass

        if final_filepath_to_check and os.path.exists(final_filepath_to_check):
            self.log_message(f"--- ✅ Hoàn thành video #{index+1}/{total} ---")
            self.status_queue.put((iid, 'status', "✅ Xong"))
            if iid in self.tree_item_map:
                self.tree_item_map[iid]['status'] = 'success'
                self.tree_item_map[iid]['filepath'] = final_filepath_to_check
        else:
            self.log_message(f"--- ⚠️ Lỗi xử lý sau tải video #{index+1}/{total}. Không tìm thấy file video cuối cùng. ---")
            self.log_message(f"   (Đã kiểm tra đường dẫn từ hook: {final_filepath_from_hook})")
            self.log_message(f"   (Đã kiểm tra đường dẫn dự kiến: {expected_final_path_mp4}, {expected_final_path_mkv}, {expected_final_path_webm})")
            self.status_queue.put((iid, 'status', "❌ Lỗi")) # Sửa text
            if iid in self.tree_item_map: self.tree_item_map[iid]['status'] = 'error'

    def on_download_finished(self):
        self.start_download_button.configure(state="normal", text="Tải Video Đã Chọn")
        self.progress_bar.set(0)

    def select_output_folder(self):
        folder_path = filedialog.askdirectory(initialdir=self.output_path_var.get())
        if folder_path: self.output_path_var.set(folder_path)

    def open_output_folder(self):
        path = self.output_path_var.get()
        if os.path.exists(path):
            try:
                if sys.platform == "win32": os.startfile(path)
                elif sys.platform == "darwin": subprocess.call(["open", path])
                else: subprocess.call(["xdg-open", path])
            except Exception as e: self.log_message(f"❌ Lỗi khi mở thư mục: {e}")
        else: self.log_message(f"⚠️ Thư mục '{path}' không tồn tại.")

    def _clear_list(self):
        for i in self.video_tree.get_children(): self.video_tree.delete(i)
        self.tree_item_map.clear()
        self.total_videos_var.set("Tổng video: 0") # Sửa lại text
        self.scan_button.configure(text="Quét Video") # Sửa lại text

    def _select_all_tree(self): self.video_tree.selection_set(self.video_tree.get_children())
    def _deselect_all_tree(self): self.video_tree.selection_set()

    def _show_context_menu(self, event):
        selected_iids = self.video_tree.selection()
        if selected_iids:
            # Cho phép tải lại/copy info nếu video đã sẵn sàng hoặc đã tải xong/lỗi
            allow_action = any(self.tree_item_map.get(iid, {}).get('status') in ['ready', 'success', 'error'] for iid in selected_iids)
            is_tiktok = any(self.tree_item_map.get(iid, {}).get('extractor_key','').lower() == 'tiktok' for iid in selected_iids)
            # Kiểm tra xem có thumbnail URL không
            has_thumb = any(self.tree_item_map.get(iid, {}).get('thumbnail') for iid in selected_iids)

            state = "normal" if allow_action else "disabled"
            sub_state = "normal" if allow_action and not is_tiktok else "disabled"
            thumb_state = "normal" if allow_action and has_thumb else "disabled" # Thêm state cho thumbnail

            self.context_menu.entryconfigure("Tải Video Đã Chọn", state=state)
            self.context_menu.entryconfigure("Tải Phụ đề SRT...", state=sub_state)
            self.context_menu.entryconfigure("Tải Nội dung (TXT)", state=sub_state)
            # Cập nhật state cho thumbnail
            self.context_menu.entryconfigure("Xem Thumbnail", state=thumb_state)
            self.context_menu.entryconfigure("Tải Thumbnail...", state=thumb_state)
            #
            self.context_menu.entryconfigure("Copy Description", state=state)
            self.context_menu.entryconfigure("Copy Keywords", state=state)
            self.context_menu.entryconfigure("Copy Video URL", state=state)
            self.context_menu.entryconfigure("Mở URL trong trình duyệt", state=state)
            self.context_menu.tk_popup(event.x_root, event.y_root)

    def _sort_tree_column(self, col, is_numeric):
        key_map = {
            'size': 'filesize_approx', 'views': 'view_count', 'likes': 'like_count',
            'comments': 'comment_count', 'date': 'upload_date', 'duration': 'duration',
            'language': 'language'
        }
        # Sửa key cho cột content_status thành 'Sub' để khớp header
        # data_key = key_map.get(col, col if col not in ['status', 'content_status'] else f'{col}_text')
        data_key = key_map.get(col, col if col != 'content_status' else 'content_status_text')
        if col == 'status': # Cột Status giờ chỉ có icon hoặc text lỗi/đang tải
             data_key = 'status_text' # Dùng text lưu trữ để sort

        try:
            data = []
            for iid in self.video_tree.get_children():
                item_data = self.tree_item_map.get(iid, {})
                val = item_data.get(data_key)
                if is_numeric:
                    try: numeric_val = float(val) if val is not None and isinstance(val, (int, float, str)) and str(val).replace('.','',1).replace('-','',1).isdigit() else 0.0
                    except: numeric_val = 0.0
                    data.append((numeric_val, iid))
                else: data.append((str(val) if val is not None else "", iid))
        except Exception as e: self.log_message(f"Sorting error: {e}"); return

        reverse_sort = self.sort_reverse
        if self.sort_column == col: self.sort_reverse = not self.sort_reverse; reverse_sort = self.sort_reverse
        else: self.sort_column = col; self.sort_reverse = False; reverse_sort = False
        data.sort(key=lambda t: t[0], reverse=reverse_sort)
        for i, (val, iid) in enumerate(data): self.video_tree.move(iid, '', i)
        for c in self.video_tree['columns']:
            current_text = self.video_tree.heading(c)['text'].replace(' ▼','').replace(' ▲','')
            self.video_tree.heading(c, text=current_text)
        new_heading = self.video_tree.heading(col)['text'] + (' ▼' if reverse_sort else ' ▲')
        self.video_tree.heading(col, text=new_heading)

    def _copy_description(self):
        selected_iids = self.video_tree.selection()
        if not selected_iids: self.log_message("⚠️ Vui lòng chọn video."); return
        iid = selected_iids[0]
        item_data = self.tree_item_map.get(iid)
        if not item_data or item_data.get('status') not in ['ready', 'success', 'error']: self.log_message("⚠️ Chi tiết video chưa sẵn sàng."); return
        description = item_data.get('description', '')
        if description:
            try:
                self.clipboard_clear(); self.clipboard_append(description)
                self.log_message(f"✅ Đã copy Description cho: {item_data.get('title', iid)}")
            except Exception as e: self.log_message(f"❌ Lỗi copy description: {e}")
        else: self.log_message("ℹ️ Không có description.")

    def _copy_keywords(self):
        selected_iids = self.video_tree.selection()
        if not selected_iids: self.log_message("⚠️ Vui lòng chọn video."); return
        iid = selected_iids[0]
        item_data = self.tree_item_map.get(iid)
        if not item_data or item_data.get('status') not in ['ready', 'success', 'error']: self.log_message("⚠️ Chi tiết video chưa sẵn sàng."); return
        tags = item_data.get('tags')
        if tags and isinstance(tags, list):
            keywords_text = ", ".join(tags)
            try:
                self.clipboard_clear(); self.clipboard_append(keywords_text)
                self.log_message(f"✅ Đã copy Keywords cho: {item_data.get('title', iid)}")
            except Exception as e: self.log_message(f"❌ Lỗi copy keywords: {e}")
        else: self.log_message("ℹ️ Không có keywords/tags.")

    def _copy_video_url(self, log_only=False):
        selected_iids = self.video_tree.selection()
        if not selected_iids:
            if not log_only: self.log_message("⚠️ Vui lòng chọn video."); return
            return
        iid = selected_iids[0]
        item_data = self.tree_item_map.get(iid)
        # Chỉ cần có webpage_url là copy được, không cần check status
        if not item_data or not item_data.get('webpage_url'):
            if not log_only: self.log_message("⚠️ Không có dữ liệu URL cho video."); return
            return
        url = item_data.get('webpage_url')
        if url:
            try:
                self.clipboard_clear(); self.clipboard_append(url)
                if not log_only: self.log_message(f"✅ Đã copy URL cho: {item_data.get('title', iid)}")
            except Exception as e:
                if not log_only: self.log_message(f"❌ Lỗi copy URL: {e}")
        else:
            if not log_only: self.log_message("ℹ️ Không tìm thấy URL cho video này.")


# --- Chạy ứng dụng ---
if __name__ == '__main__':
    if not YT_DLP_AVAILABLE:
        print("LỖI NGHIÊM TRỌNG: Thư viện yt-dlp là bắt buộc nhưng không tìm thấy.")
        print("Vui lòng cài đặt bằng lệnh: pip install yt-dlp")
        try:
             import tkinter as tk; from tkinter import messagebox
             root = tk.Tk(); root.withdraw()
             messagebox.showerror("Lỗi Thiếu Thư Viện", "Thư viện yt-dlp là bắt buộc...\nVui lòng cài đặt:\npip install yt-dlp")
             root.destroy()
        except ImportError: pass
        sys.exit(1)

    mp.freeze_support()
    app = ctk.CTk()
    app.title("Downloader Tool V9 (Thumbnail Cache)") # Đổi tên
    app.geometry("1400x900")

    try:
        if sys.platform == "win32": from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except Exception as e: print(f"Không thể đặt DPI awareness: {e}")

    downloader_tab = DownloaderTab(master=app)
    downloader_tab.pack(fill="both", expand=True)
    app.mainloop()