#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cho Tab Quản lý Kênh (Phiên bản chuyên nghiệp).
Chỉ cache ảnh thumbnail, luôn tải lại dữ liệu mới và sửa lỗi race condition.
"""

import customtkinter as ctk
from tkinter import ttk, messagebox, Menu
import json
import threading
from pathlib import Path
from datetime import datetime
import queue
import sys
import os
import subprocess
import webbrowser

try:
    import yt_dlp
    from PIL import Image, ImageTk
    import requests
    from io import BytesIO
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

MANAGER_DATA_FILE = Path.cwd() / "channel_manager_data.json"
CACHE_DIR = Path.cwd() / "channel_cache"

def format_number(n):
    if n is None: return "N/A"
    try:
        n = int(n)
        if n >= 1_000_000_000: return f"{n / 1_000_000_000:.2f} B"
        if n >= 1_000_000: return f"{n / 1_000_000:.2f} M"
        if n >= 1_000: return f"{n / 1_000:.1f} K"
        return str(n)
    except (ValueError, TypeError): return "N/A"

class ManageChannelTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(1, weight=1)
        self.ui_queue = queue.Queue(); self.manager_map = {}; self.channel_data_cache = {}
        self.channel_thumbnails = {}; self.video_thumbnails = {}; self.video_data_cache = {}
        
        self.channel_sort_column = None; self.channel_sort_reverse = False
        self.video_sort_column = "upload_date"; self.video_sort_reverse = True
        self.USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        self.current_video_worker_token = 0

        CACHE_DIR.mkdir(exist_ok=True)
        self.log_message(f"INFO: Thư mục cache ảnh: {CACHE_DIR}")

        self.create_widgets()
        self.load_channels_from_file()
        self.after(100, self.process_ui_queue)

    def log_message(self, msg): self.ui_queue.put(("log", msg))

    def process_ui_queue(self):
        try:
            for _ in range(100):
                if self.ui_queue.empty(): break
                msg_type, data = self.ui_queue.get_nowait()
                if msg_type == "log":
                    self.log_textbox.configure(state="normal"); self.log_textbox.insert("end", data + "\n")
                    self.log_textbox.configure(state="disabled"); self.log_textbox.see("end")
                elif msg_type == "update_channel_row":
                    iid, values, image = data
                    if self.channel_tree.exists(iid):
                        if image: self.channel_thumbnails[iid] = image
                        self.channel_tree.item(iid, values=values, image=image or "")
                elif msg_type == "scan_finished":
                    self.load_button.configure(state="normal", text="Tải Dữ Liệu Toàn Bộ")
                    self.log_message("✅ Đã hoàn tất chu trình tải dữ liệu!")
                elif msg_type == "populate_video_placeholders": self._populate_video_list_placeholders(data)
                elif msg_type == "update_video_row":
                    token, iid, values, image = data
                    if token == self.current_video_worker_token and self.video_tree.exists(iid):
                        if image: self.video_thumbnails[iid] = image
                        self.video_tree.item(iid, values=values, image=image or "")
        except queue.Empty: pass
        finally: self.after(100, self.process_ui_queue)

    def create_widgets(self):
        control_frame = ctk.CTkFrame(self)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        control_frame.grid_columnconfigure(5, weight=1)
        ctk.CTkLabel(control_frame, text="Người Quản Lý:").grid(row=0, column=0, padx=(10, 5), pady=10)
        self.manager_var = ctk.StringVar(value="[Tất cả]")
        self.manager_menu = ctk.CTkOptionMenu(control_frame, variable=self.manager_var, command=self.load_channels_from_file)
        self.manager_menu.grid(row=0, column=1, padx=5, pady=10)
        self.load_button = ctk.CTkButton(control_frame, text="Tải Dữ Liệu Toàn Bộ", command=self.start_scan_all_channels)
        self.load_button.grid(row=0, column=2, padx=5, pady=10)
        ctk.CTkLabel(control_frame, text="Cookie File:").grid(row=0, column=3, padx=(20, 5), pady=10)
        self.cookie_path_var = ctk.StringVar(value="cookies.txt")
        self.cookie_entry = ctk.CTkEntry(control_frame, textvariable=self.cookie_path_var)
        self.cookie_entry.grid(row=0, column=4, padx=0, pady=10, sticky="ew")
        self.use_cookie_var = ctk.BooleanVar(value=True)
        self.cookie_checkbox = ctk.CTkCheckBox(control_frame, text="Sử dụng", variable=self.use_cookie_var)
        self.cookie_checkbox.grid(row=0, column=5, padx=(5, 20), pady=10, sticky="w")
        ctk.CTkButton(control_frame, text="Sửa List Kênh...", command=self.open_manager_file).grid(row=0, column=6, padx=5, pady=10, sticky="e")

        main_pane = ctk.CTkFrame(self)
        main_pane.grid(row=1, column=0, padx=10, pady=0, sticky="nsew")
        main_pane.grid_columnconfigure(0, weight=3); main_pane.grid_columnconfigure(1, weight=4); main_pane.grid_rowconfigure(0, weight=1)
        channel_list_frame = ctk.CTkFrame(main_pane)
        channel_list_frame.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="nsew")
        channel_list_frame.grid_rowconfigure(1, weight=1); channel_list_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(channel_list_frame, text="Hệ Thống Kênh (Click header để sort)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        style = ttk.Style()
        style.theme_use("default")
        bg_color=self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"]); text_color=self._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"]); header_bg=self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["fg_color"]); active_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        style.configure("Channel.Treeview", background=bg_color, foreground=text_color, fieldbackground=bg_color, borderwidth=0, rowheight=65)
        style.map('Channel.Treeview', background=[('selected', active_color)])
        style.configure("Channel.Treeview.Heading", background=header_bg, foreground=text_color, relief="flat", font=('Segoe UI', 10, 'bold'))
        style.map("Channel.Treeview.Heading", background=[('active', active_color)])
        
        channel_cols = ('channel_name', 'status', 'platform', 'category', 'subs', 'likes', 'total_videos', 'uploaded_yesterday')
        self.channel_tree = ttk.Treeview(channel_list_frame, columns=channel_cols, show='tree headings', style="Channel.Treeview")
        self.channel_tree.heading('#0', text='Avatar'); self.channel_tree.column('#0', width=70, anchor='center', stretch=False)
        self.channel_tree.heading('channel_name', text='Tên Kênh', command=lambda: self._sort_treeview_column(self.channel_tree, 'channel_name', False)); self.channel_tree.column('channel_name', width=180, stretch=True)
        self.channel_tree.heading('status', text='Trạng thái', command=lambda: self._sort_treeview_column(self.channel_tree, 'status', False)); self.channel_tree.column('status', width=100, anchor='center')
        self.channel_tree.heading('platform', text='Nền tảng', command=lambda: self._sort_treeview_column(self.channel_tree, 'platform', False)); self.channel_tree.column('platform', width=80, anchor='center')
        self.channel_tree.heading('category', text='Thể loại', command=lambda: self._sort_treeview_column(self.channel_tree, 'category', False)); self.channel_tree.column('category', width=100, anchor='center')
        self.channel_tree.heading('subs', text='Sub / Follow', command=lambda: self._sort_treeview_column(self.channel_tree, 'subs', True)); self.channel_tree.column('subs', width=110, anchor='e')
        self.channel_tree.heading('likes', text='Likes Kênh', command=lambda: self._sort_treeview_column(self.channel_tree, 'likes', True)); self.channel_tree.column('likes', width=100, anchor='e')
        self.channel_tree.heading('total_videos', text='Videos', command=lambda: self._sort_treeview_column(self.channel_tree, 'total_videos', True)); self.channel_tree.column('total_videos', width=70, anchor='center')
        self.channel_tree.heading('uploaded_yesterday', text='Up Hôm Qua', command=lambda: self._sort_treeview_column(self.channel_tree, 'uploaded_yesterday', True)); self.channel_tree.column('uploaded_yesterday', width=90, anchor='center')
        self.channel_tree.grid(row=1, column=0, sticky="nsew")
        channel_scroll = ctk.CTkScrollbar(channel_list_frame, command=self.channel_tree.yview); channel_scroll.grid(row=1, column=1, sticky='ns')
        self.channel_tree.configure(yscrollcommand=channel_scroll.set)
        self.channel_tree.bind('<<TreeviewSelect>>', self.on_channel_select)
        self.channel_tree.bind('<Double-1>', self._on_channel_double_click)
        self.channel_context_menu = Menu(self, tearoff=0, background=bg_color, fg=text_color, activebackground=active_color, activeforeground=text_color, relief='flat', borderwidth=0)
        self.channel_context_menu.add_command(label="Tải lại dữ liệu kênh này", command=self.refresh_selected_channel)
        self.channel_tree.bind('<Button-3>', self.show_channel_context_menu)

        video_detail_frame = ctk.CTkFrame(main_pane)
        video_detail_frame.grid(row=0, column=1, padx=(5, 0), pady=0, sticky="nsew")
        video_detail_frame.grid_rowconfigure(1, weight=1); video_detail_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(video_detail_frame, text="Video Kênh (Double-click để mở link)", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        style.configure("Video.Treeview", background=bg_color, foreground=text_color, fieldbackground=bg_color, borderwidth=0, rowheight=72)
        style.map('Video.Treeview', background=[('selected', active_color)])
        style.configure("Video.Treeview.Heading", background=header_bg, foreground=text_color, relief="flat", font=('Segoe UI', 10, 'bold'))
        style.map("Video.Treeview.Heading", background=[('active', active_color)])
        
        video_cols = ('title', 'views', 'upload_date', 'duration', 'likes')
        self.video_tree = ttk.Treeview(video_detail_frame, columns=video_cols, show='tree headings', style="Video.Treeview")
        self.video_tree.heading('#0', text='Thumbnail'); self.video_tree.column('#0', width=130, anchor='center', stretch=False)
        self.video_tree.heading('title', text='Tiêu đề', command=lambda: self._sort_treeview_column(self.video_tree, 'title', False)); self.video_tree.column('title', width=300, stretch=True)
        self.video_tree.heading('views', text='Lượt xem', command=lambda: self._sort_treeview_column(self.video_tree, 'views', True)); self.video_tree.column('views', width=100, anchor='e')
        self.video_tree.heading('upload_date', text='Ngày đăng', command=lambda: self._sort_treeview_column(self.video_tree, 'upload_date', False)); self.video_tree.column('upload_date', width=100, anchor='center')
        self.video_tree.heading('duration', text='Thời lượng', command=lambda: self._sort_treeview_column(self.video_tree, 'duration', True)); self.video_tree.column('duration', width=80, anchor='center')
        self.video_tree.heading('likes', text='Thích', command=lambda: self._sort_treeview_column(self.video_tree, 'likes', True)); self.video_tree.column('likes', width=80, anchor='e')
        self.video_tree.grid(row=1, column=0, sticky="nsew")
        video_scroll = ctk.CTkScrollbar(video_detail_frame, command=self.video_tree.yview); video_scroll.grid(row=1, column=1, sticky='ns')
        self.video_tree.configure(yscrollcommand=video_scroll.set)
        self.video_tree.bind('<Double-1>', self._on_video_double_click)

        self.log_textbox = ctk.CTkTextbox(self, state="disabled", height=120, font=("Courier New", 11))
        self.log_textbox.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

    def _sort_treeview_column(self, tv, col, is_numeric):
        is_channel_tv = (tv == self.channel_tree); sort_column_attr = 'channel_sort_column' if is_channel_tv else 'video_sort_column'; sort_reverse_attr = 'channel_sort_reverse' if is_channel_tv else 'video_sort_reverse'
        current_sort_column = getattr(self, sort_column_attr); current_sort_reverse = getattr(self, sort_reverse_attr)
        reverse = not current_sort_reverse if col == current_sort_column else False
        data = []
        for iid in tv.get_children(''):
            try: col_index = tv['columns'].index(col)
            except ValueError: continue
            value = tv.item(iid, 'values')[col_index]
            if is_numeric:
                try:
                    num_str = str(value).strip().upper(); multiplier = 1
                    if 'B' in num_str: multiplier = 1_000_000_000
                    elif 'M' in num_str: multiplier = 1_000_000
                    elif 'K' in num_str: multiplier = 1_000
                    cleaned_num = float(num_str.replace('B','').replace('M','').replace('K','').strip().replace(',',''))
                    data.append((cleaned_num * multiplier, iid))
                except (ValueError, TypeError): data.append((0, iid))
            else: data.append((str(value).lower(), iid))
        data.sort(reverse=reverse)
        for i, (val, iid) in enumerate(data): tv.move(iid, '', i)
        setattr(self, sort_column_attr, col); setattr(self, sort_reverse_attr, reverse)
        for c in tv['columns']: tv.heading(c, text=tv.heading(c)['text'].replace(' ▼','').replace(' ▲',''))
        tv.heading(col, text=tv.heading(col)['text'] + (' ▼' if reverse else ' ▲'))

    def show_channel_context_menu(self, event):
        iid = self.channel_tree.identify_row(event.y)
        if iid: self.channel_tree.selection_set(iid); self.channel_context_menu.post(event.x_root, event.y_root)

    def _on_channel_double_click(self, event):
        item_id = self.channel_tree.focus()
        if not item_id or item_id not in self.channel_data_cache: return
        channel_info = self.channel_data_cache[item_id]
        if 'webpage_url' in channel_info:
            url = channel_info['webpage_url']
            self.log_message(f"INFO: Đang mở kênh: {url}")
            try: webbrowser.open(url, new=2)
            except Exception as e: self.log_message(f"ERROR: Không thể mở trình duyệt: {e}")

    def _on_video_double_click(self, event):
        item_id = self.video_tree.focus()
        if not item_id or item_id not in self.video_data_cache: return
        video_info = self.video_data_cache.get(item_id)
        url = video_info.get('webpage_url')
        if url:
            self.log_message(f"INFO: Mở link video: {url}")
            try: webbrowser.open(url, new=2)
            except Exception as e: self.log_message(f"ERROR: Không thể mở link video: {e}")

    def load_channels_from_file(self, selected_manager=None):
        try:
            if not MANAGER_DATA_FILE.exists():
                default_data = { "Youtube_AI": [{ "url": "...", "category": "..." }] }
                with open(MANAGER_DATA_FILE, 'w', encoding='utf-8') as f: json.dump(default_data, f, indent=4, ensure_ascii=False)
                self.manager_map = default_data
            else:
                with open(MANAGER_DATA_FILE, 'r', encoding='utf-8-sig') as f: self.manager_map = json.load(f)
            manager_names = ["[Tất cả]"] + sorted(list(self.manager_map.keys()))
            self.manager_menu.configure(values=manager_names)
        except Exception as e:
            self.log_message(f"ERROR: Không thể tải file dữ liệu: {e}"); messagebox.showerror("Lỗi Tải Dữ Liệu", f"Lỗi: {e}"); return
        
        self.channel_tree.delete(*self.channel_tree.get_children()); self.video_tree.delete(*self.video_tree.get_children()); self.channel_data_cache.clear()
        manager_filter = self.manager_var.get()
        channels_to_display = []
        if manager_filter == "[Tất cả]":
            temp_dict = {}
            for channel_list in self.manager_map.values():
                for channel_obj in channel_list:
                    url_key = channel_obj.get("url") if isinstance(channel_obj, dict) else channel_obj
                    if url_key and url_key not in temp_dict: temp_dict[url_key] = channel_obj
            channels_to_display = list(temp_dict.values())
        elif manager_filter in self.manager_map: channels_to_display = self.manager_map[manager_filter]
        for i, channel_entry in enumerate(channels_to_display):
            iid = str(i)
            url, category = (channel_entry.get("url"), channel_entry.get("category", "N/A")) if isinstance(channel_entry, dict) else (channel_entry, "N/A")
            if not url: continue
            self.channel_data_cache[iid] = {"original_url": url, "category": category}
            platform = "TikTok" if "tiktok.com" in url else "YouTube"
            values = (url.split('/')[-1], "Chưa tải", platform, category, "...", "...", "...", "...")
            self.channel_tree.insert('', 'end', iid=iid, values=values, text="")
        self.log_message(f"Đã tải {len(self.channel_tree.get_children())} kênh vào danh sách.")

    def open_manager_file(self):
        self.log_message(f"INFO: Mở file {MANAGER_DATA_FILE}. Hãy nhấn 'Tải Dữ Liệu' sau khi sửa.");
        try:
            if sys.platform == "win32": os.startfile(MANAGER_DATA_FILE)
            elif sys.platform == "darwin": subprocess.call(["open", str(MANAGER_DATA_FILE)])
            else: subprocess.call(["xdg-open", str(MANAGER_DATA_FILE)])
        except Exception as e: self.log_message(f"ERROR: Không thể mở file: {e}")

    def start_scan_all_channels(self):
        iids = self.channel_tree.get_children()
        if not iids: self.log_message("WARNING: Không có kênh nào để tải."); return
        self.log_message(f"Bắt đầu tải dữ liệu cho {len(iids)} kênh..."); self.load_button.configure(state="disabled", text="Đang tải...")
        threading.Thread(target=self._scan_worker, args=(iids,), daemon=True).start()

    def refresh_selected_channel(self):
        iids = self.channel_tree.selection()
        if not iids: self.log_message("WARNING: Vui lòng chọn kênh."); return
        self.log_message(f"Bắt đầu tải lại dữ liệu cho {len(iids)} kênh đã chọn..."); threading.Thread(target=self._scan_worker, args=(iids,), daemon=True).start()

    def _scan_worker(self, iids_to_scan):
        ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': 'in_playlist', 'force_generic_extractor': False, 'http_headers': {'User-Agent': self.USER_AGENT}}
        if self.use_cookie_var.get() and self.cookie_path_var.get():
            cookie_file = Path(self.cookie_path_var.get())
            if cookie_file.exists(): ydl_opts['cookiefile'] = str(cookie_file); self.log_message(f"INFO: Đang sử dụng cookie từ: {cookie_file}")
            else: self.log_message(f"WARNING: File cookie không tồn tại: {cookie_file}.")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for iid in iids_to_scan:
                if iid not in self.channel_data_cache: continue
                url = self.channel_data_cache[iid].get("original_url")
                if not url: continue
                current_values = list(self.channel_tree.item(iid, 'values')); current_values[1] = "Đang tải..."; self.ui_queue.put(("update_channel_row", (iid, tuple(current_values), None)))
                try:
                    info = ydl.extract_info(url, download=False)
                    self._process_channel_info(iid, info)
                except Exception as e:
                    self.log_message(f"ERROR: Lỗi quét kênh {url}: {type(e).__name__}")
                    current_values = list(self.channel_tree.item(iid, 'values')); current_values[1] = "❌ Lỗi 403"; self.ui_queue.put(("update_channel_row", (iid, tuple(current_values), None)))
        
        if self.load_button.cget('state') == 'disabled': self.ui_queue.put(("scan_finished", None))
    
    def _process_channel_info(self, iid, info):
        if not info: return
        self.channel_data_cache[iid].update(info)
        category = self.channel_data_cache[iid].get("category", "N/A")
        channel_thumb_url = info.get('thumbnail')
        channel_thumb_img = None
        if channel_thumb_url:
            try:
                cache_path = CACHE_DIR / f"channel_{info.get('id')}.png"
                if cache_path.exists():
                    img = Image.open(cache_path)
                else:
                    res = requests.get(channel_thumb_url, timeout=5, headers={'User-Agent': self.USER_AGENT})
                    if res.status_code == 200:
                        img = Image.open(BytesIO(res.content)); img.save(cache_path, "PNG")
                img.thumbnail((60, 60), Image.LANCZOS); channel_thumb_img = ImageTk.PhotoImage(img)
            except Exception: pass
        
        channel_name = info.get('channel') or info.get('uploader') or info.get('title', 'N/A')
        platform = "TikTok" if "tiktok" in info.get('webpage_url_domain', '') else "YouTube"
        subs = info.get('channel_follower_count') or info.get('follower_count')
        likes = info.get('like_count')
        video_count = info.get('playlist_count') or info.get('video_count') or len(info.get('entries', []))
        uploaded_yesterday = "N/A"
        
        values = (channel_name, "✅ Sẵn sàng", platform, category, format_number(subs), format_number(likes), format_number(video_count), uploaded_yesterday)
        self.ui_queue.put(("update_channel_row", (iid, values, channel_thumb_img)))

    def on_channel_select(self, event):
        selected_items = self.channel_tree.selection()
        if not selected_items: return
        selected_iid = selected_items[0]
        channel_info = self.channel_data_cache.get(selected_iid)
        
        self.current_video_worker_token += 1
        
        if channel_info and 'entries' in channel_info:
            video_entries = channel_info['entries']
            self.ui_queue.put(("populate_video_placeholders", video_entries))
            threading.Thread(target=self._video_detail_worker, args=(video_entries, self.current_video_worker_token), daemon=True).start()
        else:
            self.video_tree.delete(*self.video_tree.get_children())

    def _populate_video_list_placeholders(self, video_entries):
        self.video_tree.delete(*self.video_tree.get_children()); self.video_thumbnails.clear()
        for i, video in enumerate(video_entries):
            iid = video.get('id')
            if not iid: continue
            self.video_data_cache.setdefault(iid, {}).update(video)
            title = video.get('title', 'Đang tải...')
            values = (title, "...", "...", "...", "...")
            self.video_tree.insert('', 'end', iid=iid, values=values, text="")
    
    def _video_detail_worker(self, video_entries, token):
        ydl_opts = {'quiet': True, 'no_warnings': True, 'http_headers': {'User-Agent': self.USER_AGENT}}
        if self.use_cookie_var.get() and self.cookie_path_var.get():
             cookie_file = Path(self.cookie_path_var.get())
             if cookie_file.exists(): ydl_opts['cookiefile'] = str(cookie_file)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for video_entry in video_entries:
                if token != self.current_video_worker_token:
                    self.log_message("INFO: Tác vụ tải video đã bị hủy do chuyển kênh.")
                    return

                url = video_entry.get('url'); iid = video_entry.get('id')
                if not iid or not url: continue

                try:
                    video_info = ydl.extract_info(url, download=False)
                    if not video_info: raise ValueError("Không có dữ liệu trả về")
                    self._process_video_info(iid, video_info, token)
                except Exception as e:
                    self.log_message(f"ERROR: Lỗi lấy chi tiết video {url}: {type(e).__name__}")
                    if token == self.current_video_worker_token and self.video_tree.exists(iid):
                        current_values = list(self.video_tree.item(iid, 'values')); current_values[1] = "❌ Lỗi"
                        self.ui_queue.put(("update_video_row", (token, iid, tuple(current_values), None)))
                        
    def _get_thumbnail_image(self, item_id, thumb_url, token):
        if not thumb_url or not item_id: return None
        try:
            cache_path = CACHE_DIR / f"video_{item_id}.png"
            if cache_path.exists():
                img = Image.open(cache_path)
                self.log_message(f"INFO: Đã tải thumbnail video {item_id} từ cache.")
            else:
                if token != self.current_video_worker_token: return None
                res = requests.get(thumb_url, stream=True, timeout=5, headers={'User-Agent': self.USER_AGENT})
                if res.status_code == 200:
                    img = Image.open(BytesIO(res.content)); img.save(cache_path, "PNG")
                else: return None

            target_height = 72; aspect_ratio = img.width / img.height
            target_width = int(target_height * aspect_ratio)
            if target_width > 128: target_width = 128; target_height = int(target_width / aspect_ratio)
            img_resized = img.resize((target_width, target_height), Image.LANCZOS)
            return ImageTk.PhotoImage(img_resized)
        except Exception: return None

    def _process_video_info(self, iid, video_info, token):
        if token != self.current_video_worker_token: return
        self.video_data_cache[iid].update(video_info)
        thumbnail_image = self._get_thumbnail_image(iid, video_info.get('thumbnail'), token)
        upload_date = "N/A"
        if video_info.get('upload_date'):
            try: upload_date = datetime.strptime(video_info['upload_date'], '%Y%m%d').strftime('%Y-%m-%d')
            except ValueError: upload_date = video_info['upload_date']
        duration = video_info.get('duration')
        if duration:
            try:
                secs = int(duration); mins, secs = divmod(secs, 60); hours, mins = divmod(mins, 60)
                duration = f"{hours:02}:{mins:02}:{secs:02}" if hours > 0 else f"{mins:02}:{secs:02}"
            except (ValueError, TypeError): duration = "N/A"
        
        values = (video_info.get('title','N/A'), format_number(video_info.get('view_count')), upload_date, duration or 'N/A', format_number(video_info.get('like_count')))
        self.ui_queue.put(("update_video_row", (token, iid, values, thumbnail_image)))


if __name__ == '__main__':
    if not YT_DLP_AVAILABLE:
        print("LỖI: Thiếu thư viện. Vui lòng chạy: pip install yt-dlp Pillow requests")
        sys.exit(1)
    app = ctk.CTk()
    app.title("Test ManageChannelTab"); app.geometry("1800x900")
    tab = ManageChannelTab(app); tab.pack(fill="both", expand=True)
    app.mainloop()