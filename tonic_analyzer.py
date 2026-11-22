# -*- coding: utf-8 -*-
# Определение тоники аудиотрека в Гц (точность до 3 знака после запятой).
# Формат вывода: чистый текст, десятичный разделитель - запятая.

import sys
import os
import threading
import traceback

import numpy as np

SR = 22050

FRAME_SEC = 0.5
HOP_SEC = 0.25

FMIN = 40.0
FMAX = 500.0
HARM_K = 14
HARM_TOP = 2000.0
NFFT = 65536
RNF = 27.5

HOLD_SEC = 4.0
MIN_SEG_SEC = 6.0
CENTS_TOL = 45.0
MERGE_TOL = 35.0
TONAL_MIN = 0.30

W_PENALTY = 2.0
MIN_MOD_SEC = 25.0
MOD_MASS = 0.12
CHROMA_STEP_SEC = 5.0
CHROMA_FRAME_SEC = 1.0
KEY_W_SEC = 30.0
KEY_HOP_SEC = 15.0
KEY_CORR_MIN = 0.70
KEY_ROOT_TOL = 50.0


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


def preprocess(y):
    try:
        import librosa
        y_harm, _ = librosa.effects.hpss(y, margin=2.0)
        return np.asarray(y_harm, dtype=np.float32)
    except Exception:
        return y


def frame_features(yb, progress=None):
    from scipy.ndimage import maximum_filter1d

    n_frame = int(round(FRAME_SEC * SR))
    step = int(round(HOP_SEC * SR))
    ys = frames_of(yb, n_frame, step)
    n_frames = ys.shape[0]

    win = np.hanning(n_frame)
    nfft = NFFT
    freqs = np.fft.rfftfreq(nfft, d=1.0 / SR)
    keep = freqs <= (HARM_TOP * 1.4)
    fk = freqs[keep]

    ks = np.arange(1, HARM_K + 1, dtype=float)
    w = 1.0 / ks
    w = w / w.sum()

    coarse = np.arange(FMIN, FMAX + 0.2501, 0.25)
    fine = np.arange(-1.5, 1.5001, 0.005)

    f0s = np.empty(n_frames)
    rmss = np.empty(n_frames)
    tonals = np.empty(n_frames)

    def harm_map(grid, mA):
        g = np.clip(np.outer(grid, ks), 0.0, fk[-1])
        return np.interp(g.ravel(), fk, mA).reshape(grid.size, ks.size)

    def frame_peaks(mag):
        loc = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]))[0] + 1
        band = (fk[loc] >= FMIN * 0.75) & (fk[loc] <= HARM_TOP)
        loc = loc[band]
        if loc.size == 0:
            return fk[:0], mag[:0]
        if loc.size > 50:
            order = np.argsort(mag[loc])[::-1][:50]
            loc = np.sort(loc[order])
        return fk[loc], mag[loc]

    def cand_eval(f_c, mA, harm_lookup, pk_f, pk_a):
        harm = float((harm_lookup * w).sum())
        kf = ks * f_c
        kf = kf[kf <= HARM_TOP]
        if kf.size == 0 or pk_f.size == 0:
            return harm, 0.0
        d = np.abs(pk_f[:, None] - kf[None, :]).min(axis=1)
        tol = 0.04 * pk_f + 2.0
        covered = d <= tol
        miss = ~covered
        pen = float(np.sum(pk_a[miss] * np.clip(d[miss] / (0.12 * pk_f[miss] + 2.0), 0.0, 1.0)))
        tonal = float(pk_a[covered].sum() / (pk_a.sum() + 1e-12))
        return harm - W_PENALTY * pen, tonal

    for i in range(n_frames):
        seg = ys[i] * win
        mag = np.abs(np.fft.rfft(seg, n=nfft))[keep]
        mA = maximum_filter1d(mag, size=9, mode="nearest")
        if mA.max() <= 1e-9:
            f0s[i] = np.nan
            rmss[i] = float(np.sqrt(np.mean(ys[i] ** 2)))
            tonals[i] = 0.0
            continue

        Ah = harm_map(coarse, mA)
        Scoarse = (Ah * w).sum(axis=1)
        smax = float(Scoarse.max())

        jmax = np.where((Scoarse[1:-1] >= Scoarse[:-2]) & (Scoarse[1:-1] > Scoarse[2:]))[0] + 1
        jmax = jmax[Scoarse[jmax] >= 0.30 * smax]
        jmax = jmax[np.argsort(Scoarse[jmax])[::-1][:6]]
        base = set(jmax.tolist())
        base.add(int(np.argmax(Scoarse)))

        cands = set()
        for j in base:
            c = coarse[j]
            for m in (1.0 / 3.0, 0.5, 1.0, 2.0, 3.0):
                v = c * m
                if FMIN <= v <= FMAX:
                    cands.add(round(v, 2))
        cands = np.array(sorted(cands))

        pk_f, pk_m = frame_peaks(mag)
        pa = pk_m / (pk_m.max() + 1e-12) if pk_f.size else pk_m

        if cands.size == 0:
            cands = np.array([coarse[int(np.argmax(Scoarse))]])

        sub = np.clip(np.outer(cands, ks), 0.0, fk[-1])
        Ac = np.interp(sub.ravel(), fk, mA).reshape(cands.size, ks.size)
        cv = np.empty(cands.size)
        tv = np.empty(cands.size)
        for j in range(cands.size):
            cv[j], tv[j] = cand_eval(cands[j], mA, Ac[j], pk_f, pa)
        best = int(np.argmax(cv))

        ff = np.clip(cands[best] + fine, FMIN, FMAX)
        Af = harm_map(ff, mA)
        Sf = (Af * w).sum(axis=1)
        f0s[i] = float(ff[int(np.argmax(Sf))])
        rmss[i] = float(np.sqrt(np.mean(ys[i] ** 2)))
        tonals[i] = tv[best]

        if progress is not None and (i % max(1, n_frames // 20) == 0):
            progress((i + 1) / float(n_frames))

    times = (np.arange(n_frames) * step + n_frame / 2.0) / SR
    return times, f0s, rmss, tonals


def cents(f):
    f = np.asarray(f, dtype=np.float64)
    return 1200.0 * np.log2(np.maximum(f, 1e-9) / RNF)


def circ_diff_c(c1, c2):
    return np.abs(((c1 - c2 + 600.0) % 1200.0) - 600.0)


def circ_mean_c(vals, weights=None):
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return None
    if weights is None:
        weights = np.ones_like(v)
    ang = 2.0 * np.pi * v / 1200.0
    s = np.sum(weights * np.exp(1j * ang))
    if abs(s) < 1e-12:
        return np.average(v, weights=weights) % 1200.0
    return (np.angle(s) * 1200.0 / (2.0 * np.pi)) % 1200.0


def refine_freq(yb, t0, t1, fc):
    from scipy.signal import butter, sosfiltfilt, hilbert

    i0 = max(0, int(t0 * SR))
    i1 = min(len(yb), int(t1 * SR))
    if i1 - i0 < SR // 3:
        return None
    seg = yb[i0:i1].astype(np.float64)

    bw = 8.0
    lo = max(FMIN - 5.0, fc - bw) / (SR / 2.0)
    hi = min(HARM_TOP, fc + bw) / (SR / 2.0)
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


def refine_spec(yb, t0, t1, fc):
    from scipy.signal import butter, sosfiltfilt

    i0 = max(0, int(t0 * SR))
    i1 = min(len(yb), int(t1 * SR))
    if i1 - i0 < SR:
        return None
    seg = yb[i0:i1].astype(np.float64)

    bw = 12.0
    lo = max(FMIN - 5.0, fc - bw) / (SR / 2.0)
    hi = min(HARM_TOP, fc + bw) / (SR / 2.0)
    if hi <= lo:
        return None
    sos = butter(4, [lo, hi], btype="bandpass", output="sos")
    x = sosfiltfilt(sos, seg)
    if np.sqrt(np.mean(x ** 2)) < 0.02 * np.sqrt(np.mean(seg ** 2)) + 1e-12:
        return None

    nfft = 1 << 19
    W = min(len(x), 15 * SR)
    hop = max(SR, W // 3)
    df = SR / float(nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / SR)
    win = np.hanning(W)
    j_lo = int(np.searchsorted(freqs, max(FMIN, fc - 6.0)))
    j_hi = int(np.searchsorted(freqs, min(SR / 2.0 - 1.0, fc + 6.0)))

    fs, ws = [], []
    for j in range(0, max(1, len(x) - W + 1), hop):
        m = np.abs(np.fft.rfft(x[j:j + W] * win, n=nfft))
        k = j_lo + int(np.argmax(m[j_lo:j_hi]))
        if k <= 0 or k >= m.size - 1:
            continue
        a, b, c = m[k - 1], m[k], m[k + 1]
        den = a - 2.0 * b + c
        delta = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        fs.append(freqs[k] + float(np.clip(delta, -0.5, 0.5)) * df)
        ws.append(float(b))
    if not fs:
        return None
    fs = np.array(fs)
    ws = np.array(ws)
    if np.max(ws) <= 0:
        return None
    f = float(np.sum(fs * ws) / np.sum(ws))
    disp = float(np.std(fs))
    return {"f": f, "disp": disp}


def build_runs(frames_ok, max_gap=2):
    runs = []
    cur = []
    last = None
    for i in frames_ok:
        if cur and last is not None and i - last > max_gap + 1:
            runs.append(cur)
            cur = []
        cur.append(i)
        last = i
    if cur:
        runs.append(cur)
    return runs


def _fundamental_peak(M, fk, f0):
    tol0 = 0.04 * f0 + 2.0
    i1 = int(np.searchsorted(fk, f0 - tol0))
    i2 = int(np.searchsorted(fk, f0 + tol0))
    if i2 - i1 < 3:
        return False
    seg = M[i1:i2]
    pk = int(np.argmax(seg)) + i1
    if pk <= 20 or pk >= len(M) - 20:
        return False
    if not (M[pk] > M[pk - 1] and M[pk] >= M[pk + 1]):
        return False
    loc = float(np.median(M[max(0, pk - 40):pk + 41]))
    if loc <= 1e-12:
        return False
    return float(M[pk]) / max(M.max(), 1e-12) >= 0.025 and float(M[pk]) / loc >= 3.0


def common_fundamental_fix(yb, segments):
    good = [s for s in segments
            if np.isfinite(s.get("f", float("nan"))) and float(s["f"]) >= FMIN
            and float(s.get("resid", 9.0)) < 1.0]
    if not good:
        return
    anchor = max(good, key=lambda s: float(s["dur_w"]))
    nfft = NFFT
    freqs = np.fft.rfftfreq(nfft, d=1.0 / SR)
    keep = (freqs >= FMIN * 0.75) & (freqs <= HARM_TOP)
    fk = freqs[keep]
    Y = yb.astype(np.float64)
    lf = min(3 * SR, len(Y))
    step = max(lf, (len(Y) - lf) // 8)
    win = np.hanning(lf)
    mags = []
    for j in range(0, max(1, len(Y) - lf + 1), step):
        mags.append(np.abs(np.fft.rfft(Y[j:j + lf] * win, nfft))[keep])
        if len(mags) >= 8:
            break
    if not mags:
        return
    stack = np.stack(mags)
    M = np.median(stack, axis=0)
    Mmax = float(M.max())
    wmax = np.maximum(stack.max(axis=0), 1e-12)
    Mn_f = np.max(stack / wmax[None, :], axis=0)
    if Mmax <= 1e-9:
        return
    best = None
    for div in (8.0, 6.0, 4.0, 3.0, 2.0):
        f0 = float(anchor["f"]) / div
        if not (55.0 <= f0 <= FMAX):
            continue
        tot = expl = 0.0
        for s in good:
            r = float(s["f"]) / f0
            k = int(round(r))
            tot += float(s["dur_w"])
            if 1 <= k <= 8 and abs(1200.0 * np.log2(max(r / k, 1e-9))) <= 40.0:
                expl += float(s["dur_w"])
        if tot <= 0 or expl / tot < 0.75:
            continue
        if not _fundamental_peak(M, fk, f0):
            continue
        cnt = 0
        for k in (3, 4, 5, 6):
            kf = f0 * k
            tolk = 0.04 * kf + 2.0
            kk = (fk >= kf - tolk) & (fk <= kf + tolk)
            if kk.any() and float(Mn_f[kk].max()) >= 0.05:
                cnt += 1
        single = len(good) == 1
        if single and (cnt < 2 or float(anchor.get("resid", 9.0)) >= 0.5):
            continue
        best = f0
        break
    if best is None:
        return
    f0 = best
    for s in segments:
        f_s = float(s.get("f", float("nan")))
        clean = np.isfinite(f_s) and f_s >= 0.5 * f0 and f_s >= FMIN and float(s.get("resid", 9.0)) < 1.0
        if clean:
            r = f_s / f0
            k = int(round(r))
            if 1 <= k <= 8 and abs(1200.0 * np.log2(max(r / k, 1e-9))) <= 40.0:
                s["f"] = round(f0, 3)
                continue
        s["f"] = round(f0, 3)
        s["resid"] = max(float(s.get("resid", 9.0)), 1.0)
        s["trust_ok"] = False


def segment_octave_fix(yb, seg):
    from scipy.ndimage import maximum_filter1d

    f_ref = seg["f"]
    cands = []
    for m in (1.0 / 3.0, 0.5, 1.0, 2.0, 3.0):
        v = f_ref * m
        if FMIN <= v <= FMAX:
            cands.append(round(v, 2))
    if len(cands) <= 1:
        return seg

    i0 = max(0, int(seg["t0"] * SR))
    i1 = min(len(yb), int(seg["t1"] * SR))
    if i1 - i0 < SR:
        return seg

    nfft = NFFT
    freqs = np.fft.rfftfreq(nfft, d=1.0 / SR)
    keep = freqs <= (HARM_TOP * 1.4)
    fk = freqs[keep]
    ks = np.arange(1, HARM_K + 1, dtype=float)
    w = 1.0 / ks
    w = w / w.sum()

    seg_a = yb[i0:i1].astype(np.float64)
    lf = min(int(2.0 * SR), len(seg_a))
    step_lf = max(SR // 2, lf // 2)
    mags = []
    win_lf = np.hanning(lf)
    for j in range(0, max(1, len(seg_a) - lf + 1), step_lf):
        mags.append(np.abs(np.fft.rfft(seg_a[j:j + lf] * win_lf, n=nfft))[keep])
        if len(mags) >= 40:
            break
    if not mags:
        return seg
    M = np.median(np.stack(mags), axis=0)
    mA = maximum_filter1d(M, size=9, mode="nearest")
    mmax = float(mA.max())
    if mmax <= 1e-9:
        return seg

    loc = np.where((M[1:-1] > M[:-2]) & (M[1:-1] >= M[2:]))[0] + 1
    band = (fk[loc] >= FMIN * 0.75) & (fk[loc] <= HARM_TOP)
    loc = loc[band]
    if loc.size == 0:
        return seg
    if loc.size > 50:
        order = np.argsort(M[loc])[::-1][:50]
        loc = np.sort(loc[order])
    pk_f, pk_m = fk[loc], M[loc]
    pa = pk_m / (pk_m.max() + 1e-12)

    def score(c):
        kf = np.clip(ks * c, 0.0, fk[-1])
        A = np.interp(kf, fk, mA)
        harm = float((A * w).sum())
        kfr = ks * c
        kfr = kfr[kfr <= HARM_TOP]
        if kfr.size == 0 or pk_f.size == 0:
            return harm
        d = np.abs(pk_f[:, None] - kfr[None, :]).min(axis=1)
        tol = 0.04 * pk_f + 2.0
        miss = d > tol
        pen = float(np.sum(pa[miss] * np.clip(d[miss] / (0.12 * pk_f[miss] + 2.0), 0.0, 1.0)))
        return harm - W_PENALTY * pen

    scores = [score(c) for c in cands]
    best = int(np.argmax(scores))
    best_c = cands[best]

    if abs(best_c - f_ref) > 0.1 * f_ref:
        ref = refine_freq(yb, seg["t0"], seg["t1"], best_c)
        if ref is not None and ref["resid"] < 2.0:
            seg["f"] = ref["f"]
            seg["resid"] = ref["resid"]
            seg["trust_ok"] = bool(ref["resid"] < 0.5)
    return seg


def missing_fundamental_fix(yb, seg):
    from scipy.signal import find_peaks

    f = float(seg.get("f", 0.0))
    if not (FMIN * 2.0 <= f <= FMAX):
        return seg
    i0 = max(0, int(float(seg["t0"]) * SR))
    i1 = min(len(yb), int(float(seg["t1"]) * SR))
    if i1 - i0 < 2 * SR:
        return seg

    nfft = NFFT
    freqs = np.fft.rfftfreq(nfft, d=1.0 / SR)
    keep = (freqs >= FMIN * 0.75) & (freqs <= HARM_TOP)
    fk = freqs[keep]

    seg_a = yb[i0:i1].astype(np.float64)
    lf = min(int(3.0 * SR), len(seg_a))
    step_lf = max(SR, (len(seg_a) - lf) // 5) if len(seg_a) > lf else SR
    mags = []
    win_lf = np.hanning(lf)
    for j in range(0, max(1, len(seg_a) - lf + 1), step_lf):
        mags.append(np.abs(np.fft.rfft(seg_a[j:j + lf] * win_lf, n=nfft))[keep])
        if len(mags) >= 6:
            break
    if not mags:
        return seg
    M = np.median(np.stack(mags), axis=0)
    loc = np.where((M[1:-1] > M[:-2]) & (M[1:-1] >= M[2:]))[0] + 1
    if loc.size == 0:
        return seg
    if loc.size > 40:
        order = np.argsort(M[loc])[::-1][:40]
        loc = np.sort(loc[order])
    pk_f = fk[loc]
    pa = M[loc] / (M[loc].max() + 1e-12)
    strong = pa >= 0.15
    if strong.sum() < 2:
        return seg
    sf, sa = pk_f[strong], pa[strong]

    def exmask(f0, pf):
        kf = np.arange(1, HARM_K + 1, dtype=float) * f0
        kf = kf[kf <= HARM_TOP]
        d = np.abs(pf[:, None] - kf[None, :]).min(axis=1)
        return d <= (0.04 * pf + 2.0)

    def series_present(f0):
        for k in range(2, 7):
            kf = f0 * k
            if kf > HARM_TOP:
                return False
            tolk = 0.04 * kf + 2.0
            kk = (fk >= kf - tolk) & (fk <= kf + tolk)
            bar = 0.012 if k == 6 else 0.10
            if not (kk.any() and float(M[kk].max()) / (M.max() + 1e-12) >= bar):
                return False
        return True

    cur_m = exmask(f, sf)
    for m in (0.5, 1.0 / 3.0):
        fs_ = f * m
        if fs_ < FMIN:
            continue
        sub_m = exmask(fs_, sf)
        if float(sa[sub_m].sum()) < 0.75 * float(sa.sum()):
            continue
        gained = sa[~cur_m & sub_m]
        if gained.size == 0 or float(gained.max()) < 0.20:
            continue
        if not series_present(fs_):
            continue
        implied = []
        for pf in sf[sub_m]:
            n0 = max(1, int(round(pf / fs_)))
            implied.append(pf / n0)
        fx = float(np.median(implied))
        if abs(fx - fs_) <= max(0.3, 0.02 * fs_):
            seg["f"] = round(fs_, 3)
        break
    return seg


def refine_segments(yb, y, segments, times, f0s, rmss, tonals, valid, log):
    for seg in segments:
        t0 = max(0.0, times[seg["s0"]] - HOP_SEC / 2.0)
        t1 = times[seg["s1"] - 1] + HOP_SEC / 2.0
        seg["t0"], seg["t1"] = t0, t1

        frames_ok = []
        for i in range(seg["s0"], seg["s1"]):
            if valid[i] and circ_diff_c(cents(f0s[i]), seg["pc"]) <= CENTS_TOL:
                frames_ok.append(i)

        runs = build_runs(frames_ok)
        f_spec = None
        if frames_ok:
            fc_med = float(np.median(f0s[frames_ok]))
            ref_s = refine_spec(yb, t0 + 0.1, t1 - 0.1, fc_med)
            if ref_s is not None:
                f_spec, disp = ref_s["f"], ref_s["disp"]

        cand = []
        for fr in runs:
            if len(fr) < 3:
                continue
            fc_run = float(np.median(f0s[fr]))
            ra = max(0.0, times[fr[0]] - 0.2)
            rb = times[fr[-1]] + 0.2
            ref = refine_freq(yb, ra, rb, fc_run)
            if ref is None:
                continue
            cand.append({"bin": float(np.floor(cents(f0s[fr[0]]) / 1200.0)),
                         "f": ref["f"], "resid": ref["resid"],
                         "w": float(np.sum(rmss[fr])) / (1.0 + (ref["resid"] / 0.15) ** 4)})

        cand = [c for c in cand if c["resid"] < 5.0]
        vm = valid[seg["s0"]:seg["s1"]]

        if cand:
            best = min(cand, key=lambda c: c["resid"])
            tol_f = max(0.05, 0.002 * abs(best["f"]))
            cand = [c for c in cand if abs(c["f"] - best["f"]) <= tol_f]

        if cand:
            bins = np.array([c["bin"] for c in cand])
            ws = np.array([c["w"] for c in cand])
            vals = np.unique(bins)
            bws = np.array([ws[bins == v].sum() for v in vals])
            best_bin = vals[int(np.argmax(bws))]
            sel = [c for c in cand if c["bin"] == best_bin]
            wsum = float(np.sum([c["w"] for c in sel]))
            f_phase = float(np.sum([c["f"] * c["w"] for c in sel]) / wsum)
            r_phase = min(float(np.sum([c["resid"] * c["w"] for c in sel]) / wsum),
                          float(np.max([c["resid"] for c in sel])))
            seg["consistency"] = wsum / max(1.0, float(np.sum(rmss[seg["s0"]:seg["s1"]][vm]))) if vm.any() else 0.0
            if f_spec is not None and abs(f_phase - f_spec) > max(0.02, 0.004 * f_spec):
                seg["f"] = f_spec
                seg["resid"] = disp
                seg["trust_ok"] = bool(disp < 0.05)
            else:
                seg["f"] = f_phase
                seg["resid"] = r_phase
                seg["trust_ok"] = bool(r_phase < 0.5 or (f_spec is not None and disp < 0.05))
        else:
            if f_spec is not None:
                seg["f"] = f_spec
                seg["resid"] = disp
                seg["trust_ok"] = bool(disp < 0.05)
            else:
                f_pc = RNF * 2.0 ** (seg["pc"] / 1200.0)
                bin_med = np.floor(cents(f_pc) / 1200.0)
                seg["f"] = float(f_pc * (2.0 ** bin_med))
                seg["resid"] = 99.0
                seg["consistency"] = 0.0
                seg["trust_ok"] = False
        n_valid = int(vm.sum())
        seg["voiced_frac"] = (float(len(frames_ok)) / n_valid if n_valid else 0.0) if seg["s1"] > seg["s0"] else 0.0
        seg["tonal_mean"] = float(tonals[seg["s0"]:seg["s1"]][vm].mean()) if vm.any() else 0.0
        seg["dur_w"] = float(seg["s1"] - seg["s0"])

    for seg in segments:
        segment_octave_fix(yb, seg)
        missing_fundamental_fix(yb, seg)
    common_fundamental_fix(yb, segments)

    return segments


def ovoid_short_segments(segments):
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i, s in enumerate(segments):
            if (s["s1"] - s["s0"]) * HOP_SEC < MIN_SEG_SEC and len(segments) > 1:
                if i == 0:
                    j = 1
                elif i == len(segments) - 1:
                    j = i - 1
                else:
                    j = i - 1 if (segments[i - 1]["s1"] - segments[i - 1]["s0"]) >= (segments[i + 1]["s1"] - segments[i + 1]["s0"]) else i + 1
                n = segments[j]
                if circ_diff_c(s["pc"], n["pc"]) <= 90.0:
                    w = np.array([n["dur_w"], s["dur_w"]])
                    n["pc"] = circ_mean_c([n["pc"], s["pc"]], w)
                n["s0"] = min(n["s0"], s["s0"])
                n["s1"] = max(n["s1"], s["s1"])
                n["dur_w"] = float(n["s1"] - n["s0"])
                del segments[i]
                changed = True
                break
    return segments


def merge_same_pc(segments):
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i in range(len(segments) - 1):
            a, b = segments[i], segments[i + 1]
            if circ_diff_c(a["pc"], b["pc"]) <= MERGE_TOL:
                w = np.array([a["dur_w"], b["dur_w"]])
                a["pc"] = circ_mean_c([a["pc"], b["pc"]], w)
                a["s1"] = b["s1"]
                a["dur_w"] = float(a["s1"] - a["s0"])
                del segments[i + 1]
                changed = True
                break
    return segments


def merge_equal_segments(segments, tol=3.0):
    out = []
    for s in segments:
        if out:
            a = out[-1]
            fa, fs = a.get("f", float("nan")), s.get("f", float("nan"))
            if np.isfinite(fa) and np.isfinite(fs) and abs(cents(fa) - cents(fs)) <= tol \
                    and "s0" in a and "s1" in a and "s1" in s and "s0" in s:
                a["s1"] = s["s1"]
                a["t1"] = s.get("t1", a.get("t1"))
                a["dur_w"] = float(a["s1"] - a["s0"])
                total = a.get("dur_w", 0.0) + s.get("dur_w", 0.0)
                if total > 0:
                    a["voiced_frac"] = (a.get("voiced_frac", 0.0) * a["dur_w"] + s.get("voiced_frac", 0.0) * s["dur_w"]) / total
                a["resid"] = min(a.get("resid", 9.0), s.get("resid", 9.0))
                a["trust_ok"] = bool(a.get("trust_ok", False) or s.get("trust_ok", False))
                continue
        out.append(s)
    return out


_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def register_votes(r0, r1, f0s, pcs, valid, root_pc, tol=25.0):
    votes = {}
    for i in range(max(0, r0), min(r1, len(f0s))):
        if not valid[i] or np.isnan(f0s[i]):
            continue
        if circ_diff_c(float(pcs[i]), float(root_pc)) > tol:
            continue
        b = int(np.floor(cents(f0s[i]) / 1200.0))
        votes[b] = votes.get(b, 0) + 1
    return votes


def compose_single(segments, members, root, votes=None):
    fs = np.array([float(segments[i].get("f", float("nan"))) for i in members])
    ws = np.array([segments[i]["dur_w"] for i in members])
    g0 = min(segments[i]["s0"] for i in members)
    g1 = max(segments[i]["s1"] for i in members)
    if np.isnan(fs).any() or not np.isfinite(fs).all() or float(fs.max()) <= 0.0:
        return {"pc": root, "s0": g0, "s1": g1, "dur_w": float(g1 - g0)}
    fmed = float(np.median(fs))
    bins = np.round(np.log2(fs / fmed))
    vals = np.unique(bins)
    bws = np.array([float(ws[bins == v].sum()) for v in vals])
    w_share = bws / bws.sum()
    if votes:
        vtot = float(sum(votes.values()))
        if vtot > 0:
            vb = np.array([float(votes.get(int(v), 0)) for v in vals]) / vtot
            w_share = w_share + vb
    best_bin = vals[int(np.argmax(w_share))]
    sel = bins == best_bin
    f_final = float(np.sum(fs[sel] * ws[sel]) / float(ws[sel].sum()))
    return {"pc": root, "f": f_final,
            "s0": int(g0), "s1": int(g1), "dur_w": float(g1 - g0),
            "resid": float(np.average([float(segments[i].get("resid", 9.0)) for i in members], weights=ws)),
            "consistency": float(np.average([float(segments[i].get("consistency", 0.0)) for i in members], weights=ws)),
            "voiced_frac": float(np.average([float(segments[i].get("voiced_frac", 0.0)) for i in members], weights=ws)),
            "tonal_mean": float(np.average([float(segments[i].get("tonal_mean", 0.0)) for i in members], weights=ws)),
            "trust_ok": bool(np.average([1.0 if segments[i].get("trust_ok") else 0.0 for i in members], weights=ws) >= 0.5),
            "t0": float(min(segments[i]["t0"] for i in members)),
            "t1": float(max(segments[i]["t1"] for i in members))}


def _chroma_dict(det_c, fb, df):
    A = np.zeros((fb.size, 12))
    for pp in range(12):
        c = pp * 100.0 + det_c
        col = np.zeros(fb.size)
        for o in range(1, 7):
            f0 = RNF * (2.0 ** (c / 1200.0)) * (2.0 ** o)
            if f0 < 40.0 or f0 > 2000.0:
                continue
            h = 1
            while h <= 8 and f0 * h <= 2000.0:
                col += (1.0 / h) * np.exp(-0.5 * ((fb - f0 * h) / (2.2 * df)) ** 2)
                h += 1
        nrm = float(np.linalg.norm(col))
        A[:, pp] = col / nrm if nrm > 0 else col
    return A


def nnls_chroma_track(yb, log):
    from scipy.optimize import nnls
    flen = int(CHROMA_FRAME_SEC * SR)
    nfft = 32768
    freqs = np.fft.rfftfreq(nfft, d=1.0 / SR)
    band = (freqs >= 40.0) & (freqs <= 2000.0)
    fb = freqs[band]
    df = float(freqs[1] - freqs[0])
    n = len(yb)
    positions = list(range(0, max(1, n - flen + 1), int(CHROMA_STEP_SEC * SR)))
    if not positions:
        positions = [0]
    win = np.hanning(flen)
    mags = []
    for p in positions:
        chunk = yb[p:p + flen].astype(np.float64)
        w = np.zeros(flen, dtype=np.float64)
        w[:len(chunk)] = chunk * win[:len(chunk)]
        mags.append(np.abs(np.fft.rfft(w, nfft))[band])
    mags = np.array(mags, dtype=np.float64)
    mm = float(mags.max())
    if mm <= 0:
        return np.array([]), np.zeros((12, 0)), 0.0
    mags /= mm

    def solve(det, idx):
        A = _chroma_dict(det, fb, df)
        C = np.zeros((12, len(idx)))
        for j, i in enumerate(idx):
            try:
                c, _ = nnls(A, mags[i], maxiter=200)
            except Exception:
                c = np.zeros(12)
            C[:, j] = c
        return C

    sub = list(range(0, len(positions), 4))[:16]
    best_det, best_sc = 0.0, -1.0
    for det in range(-50, 51, 10):
        C = solve(float(det), sub)
        v = C.mean(axis=1)
        if float(v.sum()) <= 1e-12:
            continue
        sc = float(v.max()) / max(1e-12, float(v.mean()))
        if sc > best_sc:
            best_sc, best_det = sc, float(det)
    C = solve(best_det, list(range(len(positions))))
    ch_times = np.array([p / float(SR) for p in positions])
    return ch_times, C, best_det


def key_track_windows(C, ch_times, det):
    w = max(2, int(round(KEY_W_SEC / CHROMA_STEP_SEC)))
    hop = max(1, int(round(KEY_HOP_SEC / CHROMA_STEP_SEC)))
    pr_maj = _KK_MAJOR / np.linalg.norm(_KK_MAJOR)
    pr_min = _KK_MINOR / np.linalg.norm(_KK_MINOR)
    wins = []
    for wi in range(0, max(1, C.shape[1] - w + 1), hop):
        v = C[:, wi:wi + w].mean(axis=1)
        if float(v.sum()) <= 1e-9:
            continue
        vn = v / (np.linalg.norm(v) + 1e-12)
        best = None
        for r in range(12):
            for pr in (pr_maj, pr_min):
                cc = float(np.dot(vn, np.roll(pr, r)))
                if best is None or cc > best[0]:
                    best = (cc, r)
        if best is not None and best[0] >= KEY_CORR_MIN:
            wins.append({"t0": float(ch_times[wi]),
                         "t1": float(ch_times[min(C.shape[1] - 1, wi + w - 1)] + CHROMA_STEP_SEC),
                         "pc": (best[1] * 100.0 + det) % 1200.0})
    runs = []
    for k in wins:
        last_pc = runs[-1]["pcs"][-1] if runs else None
        if runs and circ_diff_c(k["pc"], last_pc) <= KEY_ROOT_TOL and k["t0"] - runs[-1]["t1"] <= CHROMA_STEP_SEC * 1.5:
            runs[-1]["t1"] = k["t1"]
            runs[-1]["pcs"].append(k["pc"])
        else:
            runs.append({"t0": k["t0"], "t1": k["t1"], "pcs": [k["pc"]]})
    for r in runs:
        r["pc"] = float(np.median(r["pcs"]))
    return runs


def detect_key_blocks(segments, groups, durs, root, ch_times, C, det):
    runs = key_track_windows(C, ch_times, det)
    data_idx = [j for j in range(C.shape[1]) if float(C[:, j].sum()) > 1e-9]
    cover_end = float(ch_times[data_idx[-1]]) if data_idx else 0.0
    runs = [r for r in runs if r["t1"] >= cover_end - 1.0]
    total = float(durs.sum())
    cands = []
    for run in runs:
        if (run["t1"] - run["t0"]) < MIN_MOD_SEC:
            continue
        if circ_diff_c(run["pc"], root) <= 25.0:
            continue
        gi, bestm = None, -1.0
        for g in range(len(groups)):
            if circ_diff_c(float(segments[groups[g][0]]["pc"]), run["pc"]) <= KEY_ROOT_TOL:
                m = float(durs[groups[g]].sum())
                if m > bestm:
                    bestm, gi = m, g
        if gi is None or bestm / total < MOD_MASS:
            continue
        ov = [i for i in groups[gi] if segments[i]["t0"] < run["t1"] and segments[i]["t1"] > run["t0"]]
        if not ov:
            continue
        s0f = int(min(segments[i]["s0"] for i in ov))
        s1f = int(max(segments[i]["s1"] for i in ov))
        if (s1f - s0f) * HOP_SEC < MIN_MOD_SEC:
            continue
        cands.append({"s0": s0f, "s1": s1f,
                      "pc": float(np.median([segments[i]["pc"] for i in ov])),
                      "mass": bestm})
    cands.sort(key=lambda b: (b["s0"], -b["mass"]))
    out = []
    for b in cands:
        if out and b["s0"] < out[-1]["s1"]:
            if b["mass"] > out[-1]["mass"]:
                out[-1] = b
            continue
        out.append(b)
    return out


def canonize_groups(segments, yb, y, times, f0s, rmss, tonals, valid, log):
    if len(segments) <= 1:
        return segments, True
    durs = np.array([s["dur_w"] for s in segments])
    groups = []
    for i, s in enumerate(segments):
        for g in groups:
            if circ_diff_c(s["pc"], segments[g[0]]["pc"]) <= 25.0:
                g.append(i)
                break
        else:
            groups.append([i])
    totals = [float(durs[g].sum()) for g in groups]
    best_i = int(np.argmax(totals))
    best = groups[best_i]
    sm = float(durs.sum())
    root = float(segments[best[0]]["pc"])
    n_fr = len(f0s)
    pcs_arr = np.mod(cents(f0s), 1200.0)

    blocks = []
    try:
        log("  ...скользящая проверка локальных тональностей (хрома)...")
        ch_times, C, det = nnls_chroma_track(yb, log)
        if C.shape[1] > 0:
            blocks = detect_key_blocks(segments, groups, durs, root, ch_times, C, det)
    except Exception:
        blocks = []

    if blocks:
        log("  ...устойчивые сдвиги тона: вычленение модуляций...")
        out = []
        cur = 0
        spans = []
        for b in blocks:
            spans.append((cur, b["s0"]))
            cur = b["s1"]
        spans.append((cur, n_fr))
        for (r0, r1), b in zip(spans, blocks + [None]):
            r1 = max(r1, r0)
            if r1 - r0 >= int(MIN_SEG_SEC / HOP_SEC):
                votes = register_votes(r0, r1, f0s, pcs_arr, valid, root)
                mem = [i for i in best if segments[i]["s1"] > r0 and segments[i]["s0"] < r1]
                ms = compose_single(segments, mem if mem else best, root, votes)
                ms["t0"] = float(times[r0] - HOP_SEC / 2.0)
                ms["t1"] = float(times[min(r1, n_fr - 1)] - HOP_SEC / 2.0)
                out.append(ms)
            if b is not None:
                merged = {"pc": b["pc"], "s0": int(b["s0"]), "s1": int(b["s1"]),
                          "dur_w": float(b["s1"] - b["s0"])}
                out.append(refine_segments(yb, y, [merged], times, f0s, rmss, tonals, valid, log)[0])
        if len(out) == 1:
            return out, True
        return out, False

    votes_all = register_votes(0, n_fr, f0s, pcs_arr, valid, root)

    if len(groups) >= 3 and totals[best_i] / sm >= 0.35:
        for offs in ([0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],
                     [0.0, 200.0, 300.0, 500.0, 700.0, 800.0, 1000.0]):
            all_in = all(
                any(circ_diff_c(float(segments[g[0]]["pc"]), root + o) <= 25.0 for o in offs)
                for g in groups)
            if all_in:
                return [compose_single(segments, best, root, votes_all)], True

    if totals[best_i] / sm >= 0.55:
        second = max(t for j, t in enumerate(totals) if j != best_i) if len(totals) > 1 else 0.0
        if second / sm <= 0.35:
            return [compose_single(segments, best, root, votes_all)], True
    return segments, False


def local_dominant_pc(pcs, valid, weights, half_w=16):
    n = len(pcs)
    if n == 0:
        return pcs
    n_bins = 40
    bin_w = 1200.0 / n_bins
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half_w), min(n, i + half_w + 1)
        idx = np.arange(lo, hi)[valid[lo:hi]]
        if idx.size == 0:
            continue
        v = pcs[idx]
        wv = weights[idx]
        best_hist = None
        best_bin = None
        best_mass = -1.0
        for off in (0.0, bin_w / 2.0):
            b = np.mod((v - off) / bin_w, n_bins).astype(int)
            h = np.bincount(b, weights=wv, minlength=n_bins)
            pb = int(np.argmax(h))
            m = float(h[pb])
            if m > best_mass:
                best_mass = m
                best_hist = h
                best_off = off
                best_bin = pb
        center = (best_bin + 0.5) * bin_w + best_off
        match = circ_diff_c(v, center) <= bin_w
        pcc = circ_mean_c(v[match], wv[match])
        if i > 0 and not np.isnan(out[i - 1]):
            prev_bin = int(np.mod(((out[i - 1] - best_off) / bin_w) + 0.5, n_bins))
            if float(best_hist[prev_bin]) >= 0.75 * best_mass:
                out[i] = out[i - 1]
                continue
        if pcc is not None:
            out[i] = pcc
    return out


def analyze(path, log):
    log("Загрузка файла...")
    y = load_audio(path)
    dur = len(y) / SR
    log("Трек: %.1f сек." % dur)
    log("Очистка от ударных и шума (это может занять пару минут)...")
    yb = preprocess(y)
    log("Расчет линии тона...")

    def prog(p):
        log("  ...%d%%" % int(p * 100))

    times, f0s, rmss, tonals = frame_features(yb, progress=prog)
    if np.all(np.isnan(f0s)):
        raise RuntimeError("В файле нет звука.")
    log("Поиск устойчивых участков и модуляций...")

    finite = ~np.isnan(f0s)
    rms_max = float(np.max(rmss[finite])) if np.any(finite) else 0.0
    valid = finite & (rmss >= 0.3 * rms_max) & (tonals >= TONAL_MIN)
    if valid.mean() < 0.05:
        raise RuntimeError("Не удалось выделить тональную линию в треке.")

    pcs = np.mod(cents(f0s), 1200.0)
    pcs_f = local_dominant_pc(pcs, valid & finite, rmss * np.maximum(tonals, 0.0))

    hold = max(1, int(round(HOLD_SEC / HOP_SEC)))
    segments = []
    cur_pc = pcs_f[valid][0] if valid.any() else pcs_f[0]
    cur_start = 0
    pend_pc, pend_start, pend_count = None, None, 0

    for i in range(1, len(pcs_f)):
        if not (valid[i] and circ_diff_c(pcs_f[i], cur_pc) <= CENTS_TOL):
            if pend_pc is not None and valid[i] and circ_diff_c(pcs_f[i], pend_pc) <= CENTS_TOL:
                pend_count += 1
            elif valid[i]:
                pend_pc, pend_start, pend_count = pcs_f[i], i, 1
            if pend_pc is not None and pend_count >= hold:
                segments.append({"pc": cur_pc, "s0": cur_start, "s1": pend_start, "dur_w": float(pend_start - cur_start)})
                cur_pc, cur_start = pend_pc, pend_start
                pend_pc, pend_count = None, 0
            continue
        pend_pc, pend_count = None, 0

    segments.append({"pc": cur_pc, "s0": cur_start, "s1": len(pcs_f), "dur_w": float(len(pcs_f) - cur_start)})
    if len(segments) == 1 and (segments[0]["s1"] - segments[0]["s0"]) * HOP_SEC < HOLD_SEC:
        raise RuntimeError("Не удалось определить тонику: в треке нет держащегося тона.")

    segments = merge_same_pc(segments)
    segments = ovoid_short_segments(segments)
    log("Уточнение частоты (фазовый метод)...")
    segments = refine_segments(yb, y, segments, times, f0s, rmss, tonals, valid, log)
    log("Сверка тональностей по всему произведению...")
    segments, single = canonize_groups(segments, yb, y, times, f0s, rmss, tonals, valid, log)

    if single and (len(segments) > 1 or "f" not in segments[0]):
        log("  ...единый тон по всему произведению, уточнение...")
        segments = refine_segments(yb, y, segments, times, f0s, rmss, tonals, valid, log)

    segments = merge_equal_segments(segments)

    for i, seg in enumerate(segments):
        if i == 0:
            seg["t0"] = 0.0
        else:
            seg["t0"] = segments[i - 1]["t1"]
        if i == len(segments) - 1:
            seg["t1"] = dur
        seg["t1"] = min(seg["t1"], dur)

    return {"duration": dur, "segments": segments, "single": len(segments) == 1}


def is_good(s):
    if "trust_ok" in s and not s["trust_ok"]:
        return False
    return (s.get("voiced_frac", 1.0) > 0.4)


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
        lines.append("ТОНИКА ВСЕГО ПРОИЗВЕДЕНИЯ: %s Гц" % fmt_hz(s["f"]))
        if not is_good(s):
            lines.append("(доверие к цифре низкое: тон нестабилен или бас выражен слабо)")
    else:
        lines.append("ОБНАРУЖЕНЫ МОДУЛЯЦИИ. ТОНИКИ ПО УЧАСТКАМ:")
        for i, s in enumerate(res["segments"], 1):
            trust = "высокое" if is_good(s) else "низкое"
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
