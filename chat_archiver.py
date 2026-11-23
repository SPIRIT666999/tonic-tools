import hashlib
import os
import sys
import tkinter as tk
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "Архив_чатов.txt")
STATE = os.path.join(HERE, "arhiv_state.txt")
POLL_MS = 1200
AUTO_SEC = 60
LUMO_HINTS = ("lumo",)

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32

    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def find_lumo_window():
        found = []

        @EnumWindowsProc
        def cb(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            ln = user32.GetWindowTextLengthW(hwnd)
            if ln <= 0:
                return True
            buf = ctypes.create_unicode_buffer(ln + 1)
            user32.GetWindowTextW(hwnd, buf, ln + 1)
            t = buf.value.lower()
            if any(h in t for h in LUMO_HINTS):
                found.append(hwnd)
            return True

        user32.EnumWindows(cb, 0)
        return found[0] if found else None

    def send_ctrl_key(vk):
        class KI(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]

        class INP(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("ki", KI), ("pad", ctypes.c_byte * 8)]

        arr = (INP * 4)()
        for i, (vk_, down) in enumerate(((0x11, True), (vk_, True), (vk_, False), (0x11, False))):
            arr[i].type = 1
            arr[i].ki.wVk = vk_
            if not down:
                arr[i].ki.dwFlags = 2
        user32.SendInput(4, arr, ctypes.sizeof(INP))

    def press_alt():
        class KI(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]

        class INP(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("ki", KI), ("pad", ctypes.c_byte * 8)]

        arr = (INP * 2)()
        arr[0].type = 1
        arr[0].ki.wVk = 0x12
        arr[1].type = 1
        arr[1].ki.wVk = 0x12
        arr[1].ki.dwFlags = 2
        user32.SendInput(2, arr, ctypes.sizeof(INP))

    def grab_lumo():
        hwnd = find_lumo_window()
        if not hwnd:
            return False
        prev = user32.GetForegroundWindow()
        press_alt()
        user32.SetForegroundWindow(hwnd)
        send_ctrl_key(0x41)
        send_ctrl_key(0x43)
        if prev:
            user32.SetForegroundWindow(prev)
        return True
else:
    def grab_lumo():
        return False


def delta_text(new, prev):
    if prev and prev in new:
        tail = new[new.find(prev) + len(prev):]
        return tail.strip()
    return new.strip()


def load_hashes():
    try:
        with open(STATE, encoding="utf-8") as f:
            return set(f.read().splitlines())
    except OSError:
        return set()


def save_hash(h):
    try:
        with open(STATE, "a", encoding="utf-8") as f:
            f.write(h + "\n")
    except OSError:
        pass


def archive_text(text, hashes, tag=""):
    h = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    if h in hashes:
        return False
    with open(ARCHIVE, "a", encoding="utf-8") as f:
        f.write("=== %s %s ===\n%s\n\n" % (datetime.now().strftime("%d.%m.%Y %H:%M:%S"), tag, text))
    save_hash(h)
    hashes.add(h)
    return True


def main():
    root = tk.Tk()
    root.title("Архив чатов")
    root.geometry("470x220")
    root.attributes("-topmost", True)

    tk.Label(root, text="АРХИВ ЧАТОВ", font=("Consolas", 12, "bold")).pack(pady=(10, 2))
    auto_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Автомат: каждые 60 сек слизывать окно Lumo целиком",
                   variable=auto_var).pack(anchor="w", padx=12)
    tk.Label(root, text="Открыт Lumo - программа сама выделяет и копирует переписку,\nв файл пишется только новое. Можно и вручную: Ctrl+A, Ctrl+C.",
             justify="left").pack(anchor="w", padx=12)

    status_var = tk.StringVar(value="жду...")
    status = tk.Label(root, textvariable=status_var, font=("Consolas", 11), fg="#0a7a0a")
    status.pack(pady=6)

    hashes = load_hashes()
    saved = [0]
    last = [None]
    prev_full = [None]
    auto_pending = [False]

    def do_poll():
        try:
            text = root.clipboard_get()
        except Exception:
            text = ""
        if text and text != last[0]:
            new = text
            tag = "авто"
            if auto_pending[0]:
                auto_pending[0] = False
                body = delta_text(text, prev_full[0])
            else:
                body = text.strip()
                tag = "вручную"
                prev_full[0] = text
            if body and archive_text(body, hashes, tag):
                saved[0] += 1
                status_var.set("сохранено: %d" % saved[0])
            if tag == "авто":
                prev_full[0] = text
            last[0] = text
        root.after(POLL_MS, do_poll)

    def do_auto():
        if auto_var.get():
            if grab_lumo():
                auto_pending[0] = True
        root.after(AUTO_SEC * 1000, do_auto)

    root.after(POLL_MS, do_poll)
    root.after(AUTO_SEC * 1000, do_auto)
    root.mainloop()


if __name__ == "__main__":
    main()
