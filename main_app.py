#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File chính để chạy ứng dụng.
Chỉ chứa App shell, Tabview, và các logic chung (như Log processing).
"""

import customtkinter as ctk
import queue

# --- THÊM IMPORT CHO TAB MỚI ---
from manage_channel_tab import ManageChannelTab # <<< THÊM MỚI
# Import các class Frame của từng Tab
from Tool import JoinVideoTab, log_queue as jv_log_queue # Import Tab JoinVideo và log_queue chung
from downloader_tab import DownloaderTab
# Chừa chỗ import JoinStory sau này
# from JoinStory import JoinStoryTab # Ví dụ

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Tool Tổng Hợp")
        self.geometry("1400x900")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sử dụng log_queue chung từ Tool.py
        self.log_queue = jv_log_queue
        
        self.create_widgets()
        
        # Chạy trình xử lý log chung
        self.after(100, self.process_log_queue)

    def create_widgets(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.tab_view = ctk.CTkTabview(self, anchor="w")
        self.tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    # --- Thêm các Tab (Đã sắp xếp lại) ---
        self.tab_view.add("ManageChannel")        # <<< THÊM MỚI
        self.tab_view.add("DownloadVideo")
        self.tab_view.add("JoinVideo")
        self.tab_view.add("JoinStoryVideo")
        self.tab_view.add("Config_JoinStory")
        self.tab_view.set("ManageChannel")        # <<< THAY ĐỔI: Đặt tab mới làm mặc định
        # --- Kết thúc ---

        # --- Gắn các Frame vào Tab ---
        # 0. Tab ManageChannel (Import từ manage_channel_tab.py)
        self.manage_channel_frame = ManageChannelTab(master=self.tab_view.tab("ManageChannel"))
        self.manage_channel_frame.pack(fill="both", expand=True)

        # 1. Tab DownloadVideo (Import từ downloader_tab.py)
        self.downloader_frame = DownloaderTab(master=self.tab_view.tab("DownloadVideo"))
        self.downloader_frame.pack(fill="both", expand=True)
        
        # 2. Tab JoinVideo (Import từ Tool.py)
        self.join_video_frame = JoinVideoTab(master=self.tab_view.tab("JoinVideo"))
        self.join_video_frame.pack(fill="both", expand=True)

        # 3. Tab JoinStory (Placeholder)
        tab_join_story = self.tab_view.tab("JoinStoryVideo")
        tab_join_story.grid_columnconfigure(0, weight=1)
        tab_join_story.grid_rowconfigure(0, weight=1)
        placeholder_label_1 = ctk.CTkLabel(tab_join_story, text="Giao diện Join Story Video sẽ ở đây", font=ctk.CTkFont(size=20))
        placeholder_label_1.grid(row=0, column=0, padx=20, pady=20)
        
        # 4. Tab Config JoinStory (Placeholder)
        tab_config_story = self.tab_view.tab("Config_JoinStory")
        tab_config_story.grid_columnconfigure(0, weight=1)
        tab_config_story.grid_rowconfigure(0, weight=1)
        placeholder_label_2 = ctk.CTkLabel(tab_config_story, text="Cài đặt cho Join Story Video sẽ ở đây", font=ctk.CTkFont(size=20))
        placeholder_label_2.grid(row=0, column=0, padx=20, pady=20)
        
    def process_log_queue(self):
        """Xử lý các thông điệp trong hàng đợi log (import từ Tool.py)."""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                
                # Tìm log_textbox trong JoinVideoTab
                # Đây là cách để logic chung tác động lên 1 tab cụ thể
                if hasattr(self, 'join_video_frame') and hasattr(self.join_video_frame, 'log_textbox'):
                    log_textbox = self.join_video_frame.log_textbox
                    if log_textbox:
                        log_textbox.configure(state="normal")
                        log_textbox.insert("end", msg + "\n")
                        log_textbox.configure(state="disabled")
                        log_textbox.see("end")
                else:
                    # Fallback nếu log_textbox chưa kịp tạo
                    print(msg) 
                        
        except queue.Empty:
            pass
        except Exception as e:
             print(f"Lỗi trong process_log_queue của App: {e}")
        finally:
            self.after(100, self.process_log_queue)


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()