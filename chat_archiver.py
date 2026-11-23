import hashlib
import os
import re
import sys
import tkinter as tk
from datetime import datetime

import string

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(HERE, "Архив_чатов")
ULT_PATH = r"D:\MAKACIM\02_ProtonDrive\My files\09_АРХИВ_ЧАТОВ_Lumo"


def _protondrive_in(path):
    try:
        for e in os.scandir(path):
            if e.is_dir() and "PROTONDRIVE" in e.name.upper():
                return e.path
    except OSError:
        pass
    return None


def _find_protondrive_root():
    for letter in string.ascii_uppercase:
        drive = letter + ":\\"
        if not os.path.isdir(drive):
            continue
        hit = _protondrive_in(drive)
        if hit:
            return hit
        try:
            tops = [e.path for e in os.scandir(drive)
                    if e.is_dir() and not e.name.startswith((".", "$"))]
        except OSError:
            continue
        for top in tops:
            hit = _protondrive_in(top)
            if hit:
                return hit
    return None

def _setup_archve_dir():
    global ARCHIVE_DIR
    try:
        if os.path.isdir(os.path.dirname(ULT_PATH)) or os.path.isdir(ULT_PATH):
            ARCHIVE_DIR = ULT_PATH
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            return
        root = _find_protondrive_root()
        if root:
            myfiles = root
            try:
                for e in os.scandir(root):
                    if e.is_dir() and e.name.lower() == "my files":
                        myfiles = e.path
                        break
            except OSError:
                pass
            ARCHIVE_DIR = os.path.join(myfiles, "09_АРХИВ_ЧАТОВ_Lumo")
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
    except OSError:
        pass


_setup_archve_dir()
POLL_MS = 1200
DEFAULT_AUTO_SEC = 120
IDLE_MS = 3000
LUMO_HINTS = ("lumo",)
BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def clean_title(t):
    t = re.sub(r"\s*[\|—–\-]\s*Lumo\s*$", "", t, flags=re.I).strip()
    t = BAD_CHARS.sub("_", t).strip(" ._")
    t = t[:60].strip()
    return t or "Без_имени"


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def window_title(hwnd):
        ln = user32.GetWindowTextLengthW(hwnd)
        if ln <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(ln + 1)
        user32.GetWindowTextW(hwnd, buf, ln + 1)
        return buf.value

    def find_lumo_windows():
        found = []

        @EnumWindowsProc
        def cb(hwnd, lparam):
            if user32.IsWindowVisible(hwnd):
                t = window_title(hwnd)
                if t and any(h in t.lower() for h in LUMO_HINTS):
                    found.append((hwnd, t))
            return True

        user32.EnumWindows(cb, 0)
        return found

    def get_idle_ms():
        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        plii = LASTINPUTINFO()
        plii.cbSize = ctypes.sizeof(plii)
        if not user32.GetLastInputInfo(ctypes.byref(plii)):
            return 0
        return int(kernel32.GetTickCount() - plii.dwTime) & 0x7FFFFFFF

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
        wins = find_lumo_windows()
        if not wins:
            return None
        hwnd, title = wins[0]
        prev = user32.GetForegroundWindow()
        press_alt()
        if not user32.SetForegroundWindow(hwnd):
            return None
        send_ctrl_key(0x41)
        send_ctrl_key(0x43)
        if prev:
            user32.SetForegroundWindow(prev)
        return title
else:
    def find_lumo_windows():
        return []

    def get_idle_ms():
        return 10 ** 9

    def grab_lumo():
        return None


UI_NOISE = {
    "новый чат", "все чаты", "проекты", "запланированные", "избранное",
    "недавно", "настройки", "поиск", "выйти", "справка", "поддержка",
    "новые чаты", "chats", "projects", "scheduled", "favorites", "recent",
    "new chat", "all chats", "settings", "search", "lumo", "lumo ai",
}
EMAIL_RE = re.compile(r"[\w.\-+]{1,40}@[\w\-]{2,60}\.[A-Za-z]{2,10}")
STORAGE_RE = re.compile(r"^\d+([.,]\d+)?\s*(гб|gb|мб|mb|тб|tb)$", re.I)


def is_noise_line(ln):
    t = ln.strip().lower()
    if len(t) > 40:
        return False
    if not t:
        return False
    if t in UI_NOISE:
        return True
    if len(t) < 100 and EMAIL_RE.fullmatch(t):
        return True
    if len(t) < 12 and STORAGE_RE.match(t):
        return True
    return False


def chat_name_from_body(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for i, s in enumerate(lines):
        if s.lower() in ("недавно", "recent"):
            for t in lines[i + 1:]:
                if not is_noise_line(t) and len(t) <= 80:
                    return t
    return None


def resolve_chat(title, raw):
    tl = (title or "").lower()
    if "lumo" in tl and re.search(r"assistant|privacy|private", tl):
        nm = chat_name_from_body(raw)
        if nm:
            return clean_title(nm)
    return clean_title(title)


def clean_body(raw):
    lines = raw.splitlines()
    stop = None
    for i, ln in enumerate(lines[:80]):
        if len(ln) < 100 and EMAIL_RE.search(ln):
            before = [x.strip().lower() for x in lines[:i]]
            if any(b in UI_NOISE for b in before):
                stop = i
            break
    if stop is not None:
        lines = lines[stop + 1:]
    out = []
    blank = 0
    for ln in lines:
        if is_noise_line(ln):
            continue
        if not ln.strip():
            blank += 1
            if blank > 2:
                continue
        else:
            blank = 0
        out.append(ln)
    return "\n".join(out).strip()


def delta_text(new, prev):
    if prev and prev in new:
        tail = new[new.find(prev) + len(prev):]
        return tail.strip()
    return new.strip()


def load_hashes(path):
    try:
        with open(path, encoding="utf-8") as f:
            return set(f.read().splitlines())
    except OSError:
        return set()


def archive_text(chat, text, hashes, tag=""):
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    file_txt = os.path.join(ARCHIVE_DIR, chat + ".txt")
    file_st = os.path.join(ARCHIVE_DIR, chat + ".st")
    h = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    if h in hashes:
        return False
    with open(file_txt, "a", encoding="utf-8") as f:
        f.write("=== %s %s ===\n%s\n\n" % (datetime.now().strftime("%d.%m.%Y %H:%M:%S"), tag, text))
    try:
        with open(file_st, "a", encoding="utf-8") as f:
            f.write(h + "\n")
    except OSError:
        pass
    hashes.add(h)
    return True


def main():
    root = tk.Tk()
    root.title("Архив чатов")
    root.geometry("500x280")
    root.attributes("-topmost", True)

    tk.Label(root, text="АРХИВ ЧАТОВ", font=("Consolas", 12, "bold")).pack(pady=(10, 2))
    auto_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Автомат: сам копирует окно Lumo (только пока ты не печатаешь)",
                   variable=auto_var).pack(anchor="w", padx=12)

    row = tk.Frame(root)
    row.pack(anchor="w", padx=12)
    tk.Label(row, text="каждые").pack(side="left")
    sec_var = tk.IntVar(value=DEFAULT_AUTO_SEC)
    tk.Spinbox(row, from_=30, to=600, increment=30, width=5, textvariable=sec_var).pack(side="left", padx=4)
    tk.Label(row, text="секунд").pack(side="left")

    tk.Label(root, text="Каждый чат сохраняется в свой файл\n(имя файла = имя вкладки).\nРучное Ctrl+C тоже попадает в архив,\nв файл по названию окна, откуда копировал.",
             justify="left").pack(anchor="w", padx=12)
    tk.Label(root, text="Склад: " + ARCHIVE_DIR,
             justify="left", font=("Consolas", 8), fg="#555555",
             wraplength=470).pack(anchor="w", padx=12)

    status_var = tk.StringVar(value="жду...")
    status = tk.Label(root, textvariable=status_var, font=("Consolas", 11), fg="#0a7a0a")
    status.pack(pady=6)

    hashes = {}
    prev_full = {}
    state = {"last": None, "pending_title": None, "saved": 0}

    def chat_hashes(chat):
        if chat not in hashes:
            hashes[chat] = load_hashes(os.path.join(ARCHIVE_DIR, chat + ".st"))
        return hashes[chat]

    def store(chat, body, tag, full=None):
        if body and archive_text(chat, body, chat_hashes(chat), tag):
            state["saved"] += 1
            status_var.set("сохранено: %d  |  последний: %s" % (state["saved"], chat))
        if full is not None:
            prev_full[chat] = full

    def do_poll():
        try:
            text = root.clipboard_get()
        except Exception:
            text = ""
        if text and text != state["last"]:
            state["last"] = text
            if state["pending_title"] is not None:
                ptitle, state["pending_title"] = state["pending_title"], None
                chat = resolve_chat(ptitle, text)
                body = clean_body(delta_text(text, prev_full.get(chat)))
                store(chat, body, "авто", text)
            else:
                if sys.platform == "win32":
                    fg = user32.GetForegroundWindow()
                    chat = resolve_chat(window_title(fg), text) if fg else "Без_имени"
                else:
                    chat = "Без_имени"
                store(chat, clean_body(text), "вручную", text)
        root.after(POLL_MS, do_poll)

    def do_auto():
        try:
            sec = max(30, min(600, int(sec_var.get() or DEFAULT_AUTO_SEC)))
        except (ValueError, tk.TclError):
            sec = DEFAULT_AUTO_SEC
        if auto_var.get():
            if get_idle_ms() >= IDLE_MS:
                wins = find_lumo_windows()
                if not wins:
                    if not str(status_var.get()).startswith("окно Lumo"):
                        status_var.set("окно Lumo не найдено")
                else:
                    title = grab_lumo()
                    if title:
                        state["pending_title"] = title
        root.after(sec * 1000, do_auto)

    root.after(POLL_MS, do_poll)
    root.after(DEFAULT_AUTO_SEC * 1000, do_auto)
    root.mainloop()


if __name__ == "__main__":
    main()
