# -*- coding: utf-8 -*-
# Тоника аудиотрека в Гц (точность до 3 знака после запятой).
# Формат вывода: чистый текст, десятичный разделитель - запятая.

import sys
import os
import threading
import traceback

import numpy as np

SR = 22050
LOWPASS_HZ = 400.0

FRAME_SEC = 0.5         # длина анализируемого кадра
FRAME_SEC_SHIFT = 0.25  # шаг кадров
FMIN = 40.0
FMAX = 400.0

MIN_SEG_SEC = 2.0       # минимальная длина устойчивого участка
CENTS_TOL = 40.0        # допуск "та же тоника", в центах
RNF = 27.5              # опорная частота для расчёта центов


def load_audio(path):
    import librosa
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
    except Exception:
        import soundfile as sf
        from scipy.signal import resample_poly
        data, sr = sf.read(path, always_2d=True)
        mono = data.mean(axis=1)
        if sr != SR:
            from math import gcd
            g = gcd(int(sr), SR)
            mono = resample_poly(mono, SR // g, int(sr) // g)
        y = mono.astype(np.float32)
    if len(y) < SR:
        raise RuntimeError("Трек слишком короткий.")
    return y


def frames_of(y, n_frame, step):
    n = len(y)
    if n <= n_frame:
        return np.stack([np.pad(y, (0, n_frame - n))])
    idx = range(0, n - n_frame + 1, step)
    return np.stack([y[i:i + n_frame] for i in idx])


def frame_f0s(y):
    from scipy.signal import butter, sosfiltfilt

    sos = butter(8, LOWPASS_HZ / (SR / 2.0), btype="lowpass", output="sos")
    yb = sosfiltfilt(sos, y).astype(np.float64)

    n_frame = int(round(FRAME_SEC * SR))
    step = int(round(FRAME_SEC_SHIFT * SR))
    ys = frames_of(yb, n_frame, step)
    n_frames = ys.shape[0]

    win = np.hanning(n_frame)
    nfft = 65536
    fmax = float(min(10 * FMAX, SR / 2.0 - 1.0))

    n_h = 10
    weights_k = 1.0 / np.arange(1, n_h + 1)
    weights_k /= weights_k.sum()

    def harm_score(mag, freqs, grid):
        score = np.zeros(frame_count_grid(grid))
        for k in range(1, n_h + 1):
            kf = np.minimum(k * grid, fmax)
            v = np.interp(kf, freqs, mag, left=0.0, right=0.0)
            score += weights_k[k - 1] * v
        return score

    def frame_count_grid(grid):
        return grid.size

    coarse = np.arange(FMIN, FMAX + 0.0001, 0.25)
    fine_off = np.arange(-1.5, 1.5001, 0.005)
    f0s = np.empty(n_frames)
    rmss = np.empty(n_frames)

    for i in range(n_frames):
        seg = ys[i] * win
        mag = np.abs(np.fft.rfft(seg, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=1.0 / SR)
        keep = freqs <= fmax + 1.0
        mag, freqs = mag[keep], freqs[keep]

        sc = harm_score(mag, freqs, coarse)
        f0_c = coarse[int(np.argmax(sc))]

        fine = np.clip(f0_c + fine_off, FMIN, FMAX)
        sf = harm_score(mag, freqs, fine)
        f0s[i] = float(fine[int(np.argmax(sf))])
        rmss[i] = float(np.sqrt(np.mean(ys[i] ** 2)))

    times = (np.arange(n_frames) * step + n_frame / 2.0) / SR
    return times, f0s, rmss


def cents(f):
    f = np.asarray(f, dtype=np.float64)
    return 1200.0 * np.log2(np.maximum(f, 1e-9) / RNF)


def circ_diff_c(c1, c2):
    return np.abs(((c1 - c2 + 600.0) % 1200.0) - 600.0)


def refine_freq(y, t0, t1, fc):
    from scipy.signal import butter, sosfiltfilt, hilbert

    i0 = max(0, int(t0 * SR))
    i1 = min(len(y), int(t1 * SR))
    if i1 - i0 < SR // 3:
        return None
    seg = y[i0:i1].astype(np.float64)

    bw = 8.0
    lo = max(FMIN - 5.0, fc - bw) / (SR / 2.0)
    hi = min(LOWPASS_HZ, fc + bw) / (SR / 2.0)
    if hi <= lo:
        return None
    sos = butter(4, [lo, hi], btype="bandpass", output="sos")
    x = sosfiltfilt(sos, seg)
    if np.sqrt(np.mean(x ** 2)) < 0.02 * np.sqrt(np.mean(seg ** 2)) + 1e-12:
        return None

    an = hilbert(x)
    ph = np.unwrap(np.angle(an))
    n_trim = min(int(0.5 * SR), len(ph) // 5)
    if n_trim > 0 and len(ph) - 2 * n_trim > 10:
        ph = ph[n_trim:len(ph) - n_trim]
    if len(ph) < 10:
        return None
    t = np.arange(len(ph)) / SR
    slope, intercept = np.polyfit(t, ph, 1)
    f = float(slope / (2.0 * np.pi))
    resid = float(np.std(ph - (slope * t + intercept)))
    return {"f": f, "resid": resid}


def analyze(path, log):
    log("Загрузка файла...")
    y = load_audio(path)
    dur = len(y) / SR
    log("Трек: %.1f сек. Вычисление линии тона (медленно, подожди)..." % dur)

    times, f0, rms = frame_f0s(y)
    if not np.any(rms > 0):
        raise RuntimeError("Тишина в файле.")
    log("Линия тона готова. Поиск устойчивых участков и модуляций...")

    pcs = np.mod(cents(f0), 1200.0)
    valid = rms >= 0.3 * float(np.max(rms))

    k = 5
    pcs_f = pcs.copy()
    half = k // 2
    for i in range(len(pcs)):
        lo = max(0, i - half)
        hi = min(len(pcs), i + half + 1)
        seg = pcs[lo:hi][valid[lo:hi]]
        if seg.size:
            ang = 2.0 * np.pi * seg / 1200.0
            pcs_f[i] = (np.angle(np.mean(np.exp(1j * ang))) * 1200.0 / (2.0 * np.pi)) % 1200.0

    hold = max(1, int(round(MIN_SEG_SEC / FRAME_SEC_SHIFT)))
    segments = []
    cur_pc = pcs_f[0]
    cur_start = 0
    pend_pc, pend_start, pend_count = None, None, 0

    for i in range(1, len(pcs_f)):
        if not valid[i] or circ_diff_c(pcs_f[i], cur_pc) <= CENTS_TOL:
            pend_pc, pend_count = None, 0
            continue
        if pend_pc is not None and circ_diff_c(pcs_f[i], pend_pc) <= CENTS_TOL:
            pend_count += 1
        else:
            pend_pc, pend_start, pend_count = pcs_f[i], i, 1
        if pend_count >= hold:
            segments.append({"pc": cur_pc, "s0": cur_start, "s1": pend_start})
            cur_pc, cur_start = pend_pc, pend_start
            pend_pc, pend_count = None, 0

    segments.append({"pc": cur_pc, "s0": cur_start, "s1": len(pcs_f)})
    if len(segments) == 1 and (segments[0]["s1"] - segments[0]["s0"]) < hold:
        raise RuntimeError("Не удалось определить тонику: в треке нет держащегося тона.")

    log("Уточнение частоты по нотам (фазовый метод)...")
    for idx, seg in enumerate(segments):
        t0 = times[seg["s0"]]
        t1 = times[seg["s1"] - 1] + FRAME_SEC
        if idx > 0:
            t0 = segments[idx - 1]["t1"]
        seg["t0"], seg["t1"] = t0, min(t1, dur)

        runs = []
        run_frames = []
        run_bin = None
        for i in range(seg["s0"], seg["s1"]):
            ok = valid[i] and circ_diff_c(pcs[i], seg["pc"]) <= CENTS_TOL
            if ok:
                b = np.floor(cents(f0[i]) / 1200.0)
                if run_bin is not None and b != run_bin:
                    runs.append((run_frames, run_bin))
                    run_frames = []
                if run_bin is None:
                    run_bin = b
                run_frames.append(i)
            else:
                if run_frames:
                    runs.append((run_frames, run_bin))
                run_frames, run_bin = [], None
        if run_frames:
            runs.append((run_frames, run_bin))

        cand = []
        for frames_ids, obin in runs:
            if len(frames_ids) < 3:
                continue
            fc_run = float(np.median(f0[frames_ids]))
            ra = max(seg["t0"], times[frames_ids[0]] - 0.2)
            rb = min(seg["t1"], times[frames_ids[-1]] + 0.2)
            ref = refine_freq(y, ra, rb, fc_run)
            if ref is None:
                continue
            cand.append({"bin": float(obin), "f": ref["f"], "resid": ref["resid"],
                         "w": float(np.sum(rms[frames_ids])) / (1.0 + (ref["resid"] / 0.15) ** 4)})

        cand = [c for c in cand if c["resid"] < 5.0]

        if cand:
            bins = np.array([c["bin"] for c in cand])
            ws = np.array([c["w"] for c in cand])
            vals = np.unique(bins)
            bws = np.array([ws[bins == v].sum() for v in vals])
            best_bin = vals[int(np.argmax(bws))]
            sel = [c for c in cand if c["bin"] == best_bin]
            wsum = float(np.sum([c["w"] for c in sel]))
            f_agg = float(np.sum([c["f"] * c["w"] for c in sel]) / wsum)
            resid_w = float(np.sum([c["resid"] * c["w"] for c in sel]) / wsum)
            resid_w = min(resid_w, float(np.max([c["resid"] for c in sel])))
            resid = resid_w
            used_w = float(np.sum([c["w"] for c in sel]))
            total_w = float(np.sum(rms[seg["s0"]:seg["s1"]][valid[seg["s0"]:seg["s1"]]]))
            seg["consistency"] = used_w / max(1.0, total_w)
        else:
            f_agg = RNF * 2.0 ** (seg["pc"] / 1200.0)
            resid = 99.0
            seg["consistency"] = 0.0
        seg["f"] = f_agg
        seg["resid"] = resid
        seg["voiced_frac"] = float(valid[seg["s0"]:seg["s1"]].mean())

    return {"duration": dur, "segments": segments, "single": len(segments) == 1}


def fmt_hz(x):
    return ("%.3f" % x).replace(".", ",")


def fmt_time(t):
    t = int(round(t))
    return "%d:%02d" % (t // 60, t % 60)


def build_report(res, path):
    lines = []
    lines.append("ФАЙЛ: %s" % os.path.basename(path))
    lines.append("ДЛИТЕЛЬНОСТЬ: %s" % fmt_time(res["duration"]))
    lines.append("")
    if res["single"]:
        s = res["segments"][0]
        good = (s["resid"] < 0.5 and s["consistency"] > 0.25 and s["voiced_frac"] > 0.4)
        lines.append("ТОНИКА ВСЕГО ПРОИЗВЕДЕНИЯ: %s Гц" % fmt_hz(s["f"]))
        if not good:
            lines.append("(доверие к цифре низкое: тон нестабилен или бас выражен слабо)")
    else:
        lines.append("ОБНАРУЖЕНЫ МОДУЛЯЦИИ. ТОНИКИ ПО УЧАСТКАМ:")
        for i, s in enumerate(res["segments"], 1):
            good = (s["resid"] < 0.5 and s["consistency"] > 0.25 and s["voiced_frac"] > 0.4)
            trust = "высокое" if good else "низкое"
            lines.append("%d) %s - %s = %s Гц (доверие: %s)" % (
                i, fmt_time(s["t0"]), fmt_time(s["t1"]), fmt_hz(s["f"]), trust))
    return "\n".join(lines)


def run_gui():
    import tkinter as tk
    from tkinter import scrolledtext

    root = tk.Tk()
    root.title("Определение тоники (Гц)")
    root.geometry("760x440")

    state = {"running": False}

    out = scrolledtext.ScrolledText(root, font=("Consolas", 11), state="disabled")
    out.pack(fill="both", expand=True, padx=10, pady=10)

    label = tk.Label(root, text="", font=("Consolas", 12, "bold"))
    label.pack(pady=2)

    def log(msg):
        out.configure(state="normal")
        out.insert("end", msg + "\n")
        out.see("end")
        out.configure(state="disabled")

    def clear():
        out.configure(state="normal")
        out.delete("1.0", "end")
        out.configure(state="disabled")

    def worker(path):
        try:
            res = analyze(path, log)
            rep = build_report(res, path)
            root.after(0, lambda: show(rep))
        except BaseException as e:
            msg = "ОШИБКА: %s\n%s" % (e, traceback.format_exc(limit=2))
            root.after(0, lambda: fail(msg))

    def show(rep):
        state["running"] = False
        label.configure(text="Готово.")
        clear()
        log(rep)
        btn.configure(state="normal")

    def fail(msg):
        state["running"] = False
        label.configure(text="Ошибка анализа.")
        clear()
        log(msg)
        btn.configure(state="normal")

    def on_click():
        if state["running"]:
            return
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Выбери аудиотрек",
            filetypes=[("Аудио", "*.wav *.flac *.mp3 *.m4a *.ogg *.aiff *.aif *.wma"),
                       ("Все файлы", "*.*")],
        )
        if not path:
            return
        state["running"] = True
        btn.configure(state="disabled")
        label.configure(text="Анализирую, подожди...")
        clear()
        threading.Thread(target=worker, args=(path,), daemon=True).start()

    btn = tk.Button(root, text="ВЫБРАТЬ ФАЙЛ И АНАЛИЗИРОВАТЬ",
                    font=("Consolas", 13, "bold"), command=on_click, height=2)
    btn.pack(fill="x", padx=10, pady=(0, 10))
    log("Выбери файл (WAV, FLAC, MP3, M4A, OGG, AIFF) и нажми кнопку.")
    log("Результат: тоника в Гц с точностью до третьего знака после запятой.")
    log("Если в треке модуляции - будут цифры по участкам с границами времени.")
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--gui":
        res = analyze(sys.argv[1], lambda m: print(m))
        print(build_report(res, sys.argv[1]))
    else:
        run_gui()
