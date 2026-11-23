import hashlib
import os
import tkinter as tk
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "Архив_чатов.txt")
STATE = os.path.join(HERE, "arhiv_state.txt")
POLL_MS = 1200


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


def archive_text(text, hashes):
    h = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    if h in hashes:
        return False
    with open(ARCHIVE, "a", encoding="utf-8") as f:
        f.write("=== %s ===\n%s\n\n" % (datetime.now().strftime("%d.%m.%Y %H:%M:%S"), text))
    save_hash(h)
    hashes.add(h)
    return True


def main():
    root = tk.Tk()
    root.title("Архив чатов")
    root.geometry("430x160")
    root.attributes("-topmost", True)

    tk.Label(root, text="АРХИВ ЧАТОВ ВКЛЮЧЁН", font=("Consolas", 12, "bold")).pack(pady=(14, 4))
    tk.Label(root, text="Открой Lumo, выдели переписку (Ctrl+A) и скопируй (Ctrl+C).\nКаждая копия сама дописывается в файл Архив_чатов.txt\nрядом с программой. Окно можно свернуть.",
             justify="left").pack(padx=12)

    status_var = tk.StringVar(value="жду копирование...")
    status = tk.Label(root, textvariable=status_var, font=("Consolas", 11), fg="#0a7a0a")
    status.pack(pady=8)

    hashes = load_hashes()
    saved = [0]
    last = [None]

    def poll():
        try:
            text = root.clipboard_get()
        except Exception:
            text = ""
        if text and text != last[0]:
            last[0] = text
            if archive_text(text, hashes):
                saved[0] += 1
                status_var.set("сохранено копий: %d" % saved[0])
        root.after(POLL_MS, poll)

    root.after(POLL_MS, poll)
    root.mainloop()


if __name__ == "__main__":
    main()
