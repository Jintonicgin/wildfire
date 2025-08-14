import numpy as np

def safe_log(x):
    try:
        return np.log(np.clip(x, 1e-6, None))
    except Exception:
        return np.nan # 오류 시 np.nan 반환

def safe_exp(x):
    try:
        return np.exp(np.clip(x, -100, 100))
    except Exception:
        return np.nan # 오류 시 np.nan 반환

def safe_pow(x, power):
    try:
        x = np.clip(x, 0, None)
        return np.power(x, power)
    except Exception:
        return 0.0

def safe_value(x, fallback=-999):
    return float(x) if np.isfinite(x) and x >= 0 else fallback

def fwi_calc(T, RH, W, P, month, FFMC0=85, DMC0=6, DC0=15, consecutive_dry_days=0, total_precip_30d=0):
    try:
        # --- DMC, DC 초기값 추정 로직 ---
        # 과거 데이터가 없을 경우, 최근 날씨 패턴으로 초기값을 보정합니다.
        if consecutive_dry_days > 2:
            # 연속 건조일이 길수록 DMC와 DC는 기본값보다 높아야 합니다.
            DMC0 = max(DMC0, 2 * consecutive_dry_days)
            # 30일 강수량이 적고 건조일이 길수록 DC(가뭄)는 심해집니다.
            dc_estimate_by_precip = 5 * consecutive_dry_days / max(total_precip_30d, 1)
            DC0 = max(DC0, dc_estimate_by_precip)

        # FFMC
        m = 147.2 * (101.0 - FFMC0) / (59.5 + FFMC0)
        if P > 0.5:
            rf = P - 0.5
            mo = m + 42.5 * rf * safe_exp(-100.0 / (251.0 - m)) * (1 - safe_exp(-6.93 / rf))
            m = min(mo, 250)
        ed = 0.942 * safe_pow(RH, 0.679) + 11 * safe_exp((RH - 100.0) / 10.0) + \
             0.18 * (21.1 - T) * (1 - 1 / safe_exp(0.115 * RH))
        if m < ed:
            kl = 0.424 * (1 - safe_pow(RH / 100.0, 1.7)) + \
                 0.0694 * np.sqrt(W) * (1 - safe_pow(RH / 100.0, 8))
            kw = kl * 0.581 * safe_exp(0.0365 * T)
            m = ed - (ed - m) * safe_exp(-kw)
        else:
            kl = 0.424 * (1 - safe_pow((100 - RH) / 100.0, 1.7)) + \
                 0.0694 * np.sqrt(W) * (1 - safe_pow((100 - RH) / 100.0, 8))
            kw = kl * 0.581 * safe_exp(0.0365 * T)
            m = ed + (m - ed) * safe_exp(-kw)
        FFMC = (59.5 * (250.0 - m)) / (147.2 + m)
        FFMC = min(max(FFMC, 0), 101)

        # DMC
        # ✅ Bug Fix: Use a 12-month list to prevent IndexError
        daylength_list = [4.7, 4.7, 4.7, 6.5, 5.4, 5.4, 5.8, 6.4, 6.2, 4.7, 4.7, 4.7]
        daylength = daylength_list[month - 1] if 1 <= month <= 12 else 4.7
        rk = 1.894 * (T + 1.1) * (100.0 - RH) * daylength * 0.0001
        DMC = DMC0 + rk
        if P > 1.5:
            ra = P
            rw = 0.92 * ra - 1.27
            wmi = 20.0 + safe_exp(5.6348 - DMC0 / 43.43)
            wmr = wmi + 1000 * rw / (48.77 + rk * rw)
            DMC = 43.43 * (5.6348 - safe_log(wmr - 20.0))
        DMC = max(DMC, 0) # 계산 결과가 음수가 되지 않도록 보정

        # DC
        # ✅ Bug Fix: Use a 12-month list to prevent IndexError
        Lf_list = [6.0, 6.0, 6.0, 9.0, 8.0, 7.0, 7.0, 7.0, 8.0, 6.0, 6.0, 6.0]
        Lf = Lf_list[month - 1] if 1 <= month <= 12 else 6.0
        V = 0.36 * (T + 2.8) + Lf
        DC = DC0 + 0.5 * V
        if P > 2.8:
            rw = 0.83 * P - 1.27
            smr = 800.0 * safe_exp(-DC0 / 400.0) + 3.937 * rw
            DC = 400.0 * safe_log(800.0 / smr)
        DC = max(DC, 0) # 계산 결과가 음수가 되지 않도록 보정

        # ISI
        mo = 147.2 * (101.0 - FFMC) / (59.5 + FFMC)
        ff = 91.9 * safe_exp(-0.1386 * mo) * (1 + safe_pow(mo, 5.31) / (4.93e7))
        ISI = ff * safe_exp(0.05039 * W)

        # BUI
        if DMC <= 0.4 * DC:
            BUI = (0.8 * DMC * DC) / (DMC + 0.4 * DC)
        else:
            BUI = DMC - (1 - (0.8 * DC) / (DMC + 0.4 * DC)) * (0.92 + 0.0114 * DMC)
        BUI = max(BUI, 0)

        # FWI
        bb = 0.1 * ISI * BUI
        FWI = bb if bb <= 1 else safe_exp(2.72 * safe_pow(bb, 0.647))

        return {
            "FFMC": safe_value(FFMC),
            "DMC": safe_value(DMC),
            "DC": safe_value(DC),
            "ISI": safe_value(ISI),
            "BUI": safe_value(BUI),
            "FWI": safe_value(FWI)
        }


    except Exception as e:
        # fallback in case any error occurs
        return {
            "FFMC": -999, "DMC": -999, "DC": -999,
            "ISI": -999, "BUI": -999, "FWI": -999
        }