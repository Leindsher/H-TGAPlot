# H-TGAPlot v2.95 — Gráfico Embutido
# Análise Termogravimétrica (TGA / DrTGA)
# Autor: Carlos Henrique Amaro da Silva

import os
import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import filedialog, messagebox
try:
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# ANÁLISE MANUAL DE INTERVALO (onset / endset / midpoint / perda de massa)

def analisar_intervalo_manual(temp, tga_raw, T_ini, T_fim, sigma=6,
                              drtga_raw=None, heating_rate=10.0):
    """
    Dado um intervalo [T_ini, T_fim] definido pelo usuário, calcula:

      - onset_temp    : interseção da baseline ANTES do intervalo com a
                        tangente no ponto de inflexão do flanco de queda.
      - endset_temp   : interseção da tangente no flanco de recuperação com
                        a baseline DEPOIS do intervalo.
      - midpoint_temp : temperatura onde a perda acumulada = 50 % do evento.
      - pico_temp     : temperatura do máximo da DTG dentro do intervalo.
      - massa_onset   : massa (%) no T_onset.
      - massa_endset  : massa (%) no T_endset.
      - massa_mid     : massa (%) no T_midpoint.
      - perda_massa   : perda acumulada MONOTÔNICA entre T_onset e T_endset.

    Algoritmo de onset/endset 
    ─────────────────────────────────────────────────────────────
    Quando drtga_raw é fornecido (curva DrTGA do instrumento, em mg/s):
      1. A DrTGA é suavizada com uma janela deslizante de ~500 pts para
         revelar o pico correto da taxa de perda.
      2. O pico da DrTGA suavizada = ponto de inflexão do TGA.
         A inclinação da tangente é derivada diretamente do valor da DrTGA
         no pico, dividido pela taxa de aquecimento em °C/s: slope = D/β.
         Isso evita erros de diferenciação numérica e replica a geometria
         de tangente do software do fabricante.
      3. Onset = interseção dessa tangente com a baseline pré-evento
         (fit linear numa janela de 30°C antes de T_ini).
      4. Endset = mesmo procedimento, mas usando o mínimo local da DrTGA
         (maior taxa de queda) no flanco de recuperação (pico → T_fim),
         intersectando com a baseline pós-evento (janela de 30°C após T_fim).

    Quando drtga_raw não está disponível (fallback):
      Usa diferenciação numérica da curva TGA suavizada, como na versão
      anterior — menos preciso, mas ainda funcional.
    """
    if not _SCIPY_OK:
        return None

    from scipy.ndimage import gaussian_filter1d

    T_ini = float(T_ini)
    T_fim = float(T_fim)
    if T_fim < T_ini:
        T_ini, T_fim = T_fim, T_ini

    # normaliza para % relativo ao primeiro valor não-nulo 
    massa_ini = tga_raw[tga_raw != 0][0] if np.any(tga_raw != 0) else tga_raw[0]
    if massa_ini == 0:
        return None
    tga_pct = (tga_raw / massa_ini) * 100.0

    # remove duplicatas de temperatura
    _, idx_u = np.unique(temp, return_index=True)
    T = temp[idx_u].astype(float)
    M = tga_pct[idx_u].astype(float)

    if len(T) < 10:
        return None

    # densidade de pontos por grau Celsius
    T_span = T[-1] - T[0]
    pts_per_C = (len(T) - 1) / T_span if T_span > 0 else 1.0

    # suaviza a curva TGA (para visualização e fallback)
    M_s = gaussian_filter1d(M, sigma=sigma)

    # índices do intervalo definido pelo usuário
    mask_iv = (T >= T_ini) & (T <= T_fim)
    if mask_iv.sum() < 5:
        return None
    idx_iv = np.where(mask_iv)[0]
    i_ini  = idx_iv[0]
    i_fim  = idx_iv[-1]

    # taxa de aquecimento em °C/s 
    beta = float(heating_rate) / 60.0  # converte de °C/min para °C/s

    # PREPARAÇÃO DA DrTGA — usada para localizar pico e calcular inclinação
    # Converte para %/s normalizado: positivo = perda de massa
    if drtga_raw is not None:
        D_raw = np.array(drtga_raw)[idx_u].astype(float)
        D_pct_s = -D_raw / massa_ini * 100.0  # positivo = perda

        # Janela de suavização: ~500 pts ≈ 8 °C 
        # Escala automaticamente com a densidade de pontos do arquivo.
        # Fator 1.5: calibrado para que o pico da DrTGA suavizada corresponda        
        smooth_win = max(3, int(1.5 * 500 * pts_per_C / 63.0))
        half = smooth_win // 2
        kernel = np.ones(smooth_win) / smooth_win
        D_smooth = np.convolve(D_pct_s, kernel, mode='full')[half: half + len(D_pct_s)]
        use_instrument_dtg = True
    else:
        # Fallback: derivada numérica do TGA suavizado
        dM_neg   = -np.gradient(M_s, T)
        D_smooth = gaussian_filter1d(dM_neg, sigma=sigma) / beta  # em %/s
        use_instrument_dtg = False

    # pico da DrTGA dentro do intervalo (= inflexão do TGA na queda)
    D_ev   = D_smooth[i_ini:i_fim + 1]
    idx_rel = int(np.argmax(D_ev))
    idx_pico = i_ini + idx_rel
    T_pico   = float(T[idx_pico])

    # baseline anterior: janela fixa de 30°C antes de T_ini 
    win_B   = max(10, int(30.0 * pts_per_C))
    i_bpre_fim = max(0, i_ini - 1)
    i_bpre_ini = max(0, i_bpre_fim - win_B)
    if i_bpre_fim > i_bpre_ini + 1:
        coef_bpre = np.polyfit(T[i_bpre_ini:i_bpre_fim + 1],
                               M_s[i_bpre_ini:i_bpre_fim + 1], 1)
    else:
        coef_bpre = np.array([0.0, M_s[i_ini]])
    
    # ONSET — abordagem híbrida selecionada pela taxa de aquecimento    
    # hr <= 10 °C/min: âncora = ponto de inflexão do TGA suavizado (mínimo de
    #   dM/dT no intervalo); slope = derivada numérica nesse ponto. Mais preciso
    #   quando a DrTGA tem baixa resolução digital (0.01 mg/s) e taxa lenta.
    # hr > 10 °C/min: âncora = pico da DrTGA suavizada; slope = -D_smooth/beta.
    #   Mais preciso quando o plateau da DrTGA é mais estreito e o pico define
    #   melhor a tangente.
    # Em ambos os casos o sinal do slope é NEGATIVO (TGA cai com T crescente).
    # Baseline pré adaptativa: plana em M(T_ini) quando slope_pre < -0.05 %/°C.

    dM_dT_num = np.gradient(M_s, T)          # derivada numérica: negativa onde há perda

    _SLOPE_ATIVO = -0.05   # %/°C — threshold de "região ainda ativa"

    if heating_rate <= 10.0:
        # Âncora no ponto de inflexão do TGA (máxima taxa de queda)
        dM_ev       = dM_dT_num[i_ini:i_fim + 1]
        idx_anc_on  = i_ini + int(np.argmin(dM_ev))
        T_anc_on    = float(T[idx_anc_on])
        slope_tang_on = float(dM_dT_num[idx_anc_on])   # %/°C, negativo
    else:
        # Âncora no pico da DrTGA suavizada, slope = -D/beta
        T_anc_on    = T_pico
        idx_anc_on  = idx_pico
        slope_tang_on = -(float(D_smooth[idx_pico]) / beta)   # %/°C, negativo

    M_infl_on = float(np.interp(T_anc_on, T, M_s))

    # Baseline adaptativa: plana se a região pré ainda estiver em queda
    if coef_bpre[0] < _SLOPE_ATIVO:
        coef_bpre = np.array([0.0, float(np.interp(T_ini, T, M_s))])

    # Intersecção: baseline vs tangente
    # baseline : M = coef_bpre[0]*T + coef_bpre[1]
    # tangente  : M = slope_tang_on*(T - T_anc_on) + M_infl_on   (slope_tang_on < 0)
    # → T_onset = (M_infl_on - slope_tang_on*T_anc_on - coef_bpre[1]) / (coef_bpre[0] - slope_tang_on)
    denom_on = coef_bpre[0] - slope_tang_on
    if abs(denom_on) > 1e-9:
        T_onset = (M_infl_on - slope_tang_on * T_anc_on - coef_bpre[1]) / denom_on
    else:
        T_onset = float(T_anc_on)
    T_onset = float(np.clip(T_onset, T_ini, T_fim))
    
    # ENDSET — mesma tangente (negativa) estendida até a baseline posterior    
    # A tangente passa por (T_anc_on, M_infl_on) com slope_tang_on < 0.
    # Ela intersecta a baseline pós-evento num T > T_anc_on → endset.
    slope_tang_end = slope_tang_on           # mesmo slope (negativo)
    M_infl_end     = M_infl_on               # mesmo ponto de ancoragem
    T_infl_end     = T_anc_on

    # Baseline posterior: janela de 30°C após T_fim
    win_B_post  = max(10, int(30.0 * pts_per_C))
    i_bpost_ini = min(len(T) - 1, i_fim + 1)
    i_bpost_fim = min(len(T) - 1, i_bpost_ini + win_B_post)
    if i_bpost_fim > i_bpost_ini + 1:
        coef_bpost = np.polyfit(T[i_bpost_ini:i_bpost_fim + 1],
                                M_s[i_bpost_ini:i_bpost_fim + 1], 1)
    else:
        coef_bpost = np.array([0.0, M_s[i_fim]])

    denom_end = coef_bpost[0] - slope_tang_end
    if abs(denom_end) > 1e-9:
        T_endset_cand = (M_infl_end - slope_tang_end * T_infl_end - coef_bpost[1]) / denom_end
    else:
        T_endset_cand = float(T[i_fim])

    # Garante que o endset caia no intervalo razoável [T_pico, T_bpost_fim]
    T_endset = float(np.clip(T_endset_cand, T_pico, T[i_bpost_fim]))

    # garante T_endset >= T_pico
    T_endset = max(T_endset, T_pico)

    # Massa nos pontos de onset e endset 
    M_onset  = float(np.interp(T_onset,  T, M_s))
    M_endset = float(np.interp(T_endset, T, M_s))

    # Perda acumulada MONOTÔNICA entre T_onset e T_endset
    # Usa a mesma função do modo automático — garante resultados idênticos
    # para o mesmo intervalo.
    perda = _perda_monotonica(T, M_s, T_onset, T_endset)

    # Midpoint: T onde M_raw(T) = (M_raw(T_ini) + M_raw(T_fim)) / 2     
    # do intervalo definido pelo usuário, usando a curva TGA original (sem suavização).
    M_ini_mid  = float(np.interp(T_ini, T, M))
    M_fim_mid  = float(np.interp(T_fim, T, M))
    M_alvo_mid = (M_ini_mid + M_fim_mid) / 2.0
    mask_event = (T >= T_ini) & (T <= T_fim)
    T_event    = T[mask_event]
    M_event    = M[mask_event]   # dados brutos (não suavizados)
    T_mid      = float((T_ini + T_fim) / 2.0)  # fallback
    for k in range(1, len(M_event)):
        if M_event[k - 1] >= M_alvo_mid >= M_event[k]:
            frac  = (M_event[k - 1] - M_alvo_mid) / (M_event[k - 1] - M_event[k] + 1e-12)
            T_mid = float(T_event[k - 1] + frac * (T_event[k] - T_event[k - 1]))
            break

    M_mid = float(np.interp(T_mid, T, M_s))

    # Perda bruta: M(T_ini) - M(T_fim)
    # Subtração direta entre os limites definidos pelo usuário, sem depender
    # do onset/endset calculado. Equivalente ao "Weight Loss" do software
    # Usa a curva original normalizada, sem suavizacao,
    # para reproduzir a diferenca direta entre os pontos escolhidos.
    M_em_T_ini  = float(np.interp(T_ini, T, M))
    M_em_T_fim  = float(np.interp(T_fim, T, M))
    perda_bruta = float(M_em_T_ini - M_em_T_fim)

    return {
        "T_ini":         T_ini,
        "T_fim":         T_fim,
        "pico_temp":     T_pico,
        "onset_temp":    T_onset,
        "endset_temp":   T_endset,
        "midpoint_temp": T_mid,
        "massa_onset":   M_onset,
        "massa_endset":  M_endset,
        "massa_mid":     M_mid,
        "massa_ini_intervalo": M_em_T_ini,
        "massa_fim_intervalo": M_em_T_fim,
        "perda_massa":   perda,
        "perda_bruta":   perda_bruta,
        "massa_ini_mg":  float(massa_ini),
    }



def _perda_monotonica(T, M_s, T_ini, T_fim):
    """
    Calcula a perda de massa acumulada monotonicamente entre T_ini e T_fim.
    Só soma quando a curva cai — oscilações de subida são ignoradas.
    Garante consistência entre modo manual e automático.
    """
    mask = (T >= T_ini) & (T <= T_fim)
    M_ev = M_s[mask]
    acum = 0.0
    for k in range(1, len(M_ev)):
        delta = M_ev[k - 1] - M_ev[k]
        if delta > 0:
            acum += delta
    return float(acum)


def _mesclar_eventos_sobrepostos(eventos, T, M_s):
    """
    Mescla eventos cujos intervalos [onset, endset] se sobrepõem, incluindo
    casos em que um evento está completamente contido dentro de outro, ou em
    que o onset de um começa antes do endset do anterior.

    O algoritmo é iterativo: após cada mesclagem o intervalo resultante pode
    absorver novos vizinhos, então repete até não haver mais sobreposições.

    Para cada grupo mesclado:
      - onset_temp  : menor onset do grupo
      - endset_temp : maior endset do grupo
      - pico_temp   : pico com maior perda individual dentro do grupo
      - massa_onset / massa_endset : interpolados na curva suavizada
      - perda_massa : massa_onset - massa_endset (perda total real do intervalo)

    Parâmetros
    ----------
    eventos : lista de dicts já calculados
    T       : array de temperatura (sem duplicatas, ordenado)
    M_s     : array de massa suavizada correspondente

    Retorna
    -------
    Lista de dicts mesclados (mesmo formato dos eventos originais).
    """
    if not eventos:
        return []

    # representa cada evento como (onset, endset, lista_de_eventos_originais)
    intervalos = sorted(
        [(e["onset_temp"], e["endset_temp"], [e]) for e in eventos],
        key=lambda x: x[0]
    )

    # mesclagem iterativa: continua até nenhuma sobreposição restar
    mudou = True
    while mudou:
        mudou = False
        resultado = []
        i = 0
        while i < len(intervalos):
            on_a, end_a, grupo_a = intervalos[i]
            j = i + 1
            while j < len(intervalos):
                on_b, end_b, grupo_b = intervalos[j]
                # sobreposição: onset de B está antes do endset de A
                # (cobre também o caso em que B está totalmente dentro de A)
                if on_b <= end_a:
                    # mescla: expande o intervalo A para cobrir B também
                    end_a  = max(end_a, end_b)
                    grupo_a = grupo_a + grupo_b
                    mudou  = True
                    j += 1
                else:
                    break
            resultado.append((on_a, end_a, grupo_a))
            i = j
        intervalos = resultado

    # constrói os eventos finais a partir dos grupos mesclados
    eventos_mesclados = []
    for (T_onset_merge, T_endset_merge, grupo) in intervalos:
        if len(grupo) == 1:
            eventos_mesclados.append(grupo[0])
            continue

        # pico representativo: o de maior perda individual
        pico_repr = max(grupo, key=lambda e: e["perda_massa"])["pico_temp"]

        # massa nos extremos do intervalo mesclado (interpolada na curva suavizada)
        M_onset_merge  = float(np.interp(T_onset_merge,  T, M_s))
        M_endset_merge = float(np.interp(T_endset_merge, T, M_s))
        perda_merge    = _perda_monotonica(T, M_s, T_onset_merge, T_endset_merge)

        eventos_mesclados.append({
            "pico_temp":    pico_repr,
            "onset_temp":   float(T_onset_merge),
            "endset_temp":  float(T_endset_merge),
            "massa_onset":  M_onset_merge,
            "massa_endset": M_endset_merge,
            "perda_massa":  perda_merge,
        })

    return eventos_mesclados


def detectar_eventos_tga(temp, tga_pct, n_eventos=3, sigma=8, min_perda=1.0,
                          min_sep_temp=20.0):
    """
    Detecta eventos de decomposição na curva TGA e retorna onset, endset e
    perda de massa para cada evento.

    Eventos cujos intervalos [onset, endset] se sobrepõem são automaticamente
    mesclados em um único evento com onset mínimo, endset máximo e perda total.

    Parâmetros
    ----------
    temp       : array de temperatura (°C)
    tga_pct    : array de massa residual (%)
    n_eventos  : número máximo de eventos a detectar (antes da mesclagem)
    sigma      : suavização gaussiana para a derivada interna
    min_perda  : perda mínima (%) para considerar um evento válido
    min_sep_temp: separação mínima entre picos (°C)

    Retorna
    -------
    Lista de dicts com chaves:
        pico_temp, onset_temp, endset_temp,
        massa_onset, massa_endset, perda_massa
    """
    if not _SCIPY_OK:
        return []

    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    # garante arrays sem duplicatas e ordenados
    _, idx_uniq = np.unique(temp, return_index=True)
    T   = temp[idx_uniq].astype(float)
    M   = tga_pct[idx_uniq].astype(float)

    if len(T) < 20:
        return []

    # derivada suavizada (negativa = perda de massa)
    M_s  = gaussian_filter1d(M, sigma=sigma)
    dM   = -np.gradient(M_s, T)          # positivo onde há perda
    dM_s = gaussian_filter1d(dM, sigma=sigma * 1.5)

    # detecta picos de perda (mínima distância em índices)
    min_dist_idx = max(5, int(min_sep_temp / np.median(np.diff(T))))
    picos, props = find_peaks(dM_s, height=0, distance=min_dist_idx)
    if len(picos) == 0:
        return []

    # ordena por intensidade e pega os N maiores
    ordem = np.argsort(dM_s[picos])[::-1]
    picos_sel = picos[ordem[:n_eventos]]
    picos_sel = np.sort(picos_sel)   # volta a ordem cronológica

    eventos = []
    for pk in picos_sel:
        T_pico = T[pk]

        # ONSET: interseção da baseline antes do pico com a tangente do flanco de subida
        # Janela de baseline: region plana antes do evento
        janela_base = max(10, int(0.08 * len(T)))
        i_base_ini  = max(0, pk - 3 * janela_base)
        i_base_fim  = max(0, pk - janela_base)
        if i_base_fim - i_base_ini < 5:
            i_base_ini = max(0, pk - 20)
            i_base_fim = max(0, pk - 5)

        # tangente da baseline (região plana antes)
        if i_base_fim > i_base_ini + 2:
            coef_base = np.polyfit(T[i_base_ini:i_base_fim],
                                   M_s[i_base_ini:i_base_fim], 1)
        else:
            coef_base = np.array([0.0, M_s[max(0, pk - 5)]])

        # ponto de maior declive no flanco de subida (antes do pico)
        i_flanco_ini = max(0, pk - janela_base)
        i_flanco_fim = pk + 1
        if i_flanco_fim > i_flanco_ini + 2:
            idx_max_slope = i_flanco_ini + np.argmax(dM_s[i_flanco_ini:i_flanco_fim])
            # tangente na inflexão do flanco
            hw = max(3, janela_base // 3)
            i0 = max(0, idx_max_slope - hw)
            i1 = min(len(T) - 1, idx_max_slope + hw)
            coef_tang = np.polyfit(T[i0:i1+1], M_s[i0:i1+1], 1)
        else:
            coef_tang = coef_base

        # interseção das duas retas
        if abs(coef_tang[0] - coef_base[0]) > 1e-9:
            T_onset = (coef_base[1] - coef_tang[1]) / (coef_tang[0] - coef_base[0])
        else:
            T_onset = T[i_flanco_ini]

        # limita ao intervalo razoável
        T_onset = float(np.clip(T_onset, T[i_base_ini], T_pico))

        # ENDSET: interseção da tangente do flanco de descida com baseline depois do pico
        janela_end = max(10, int(0.08 * len(T)))
        i_end_ini  = min(len(T) - 1, pk + janela_end)
        i_end_fim  = min(len(T) - 1, pk + 3 * janela_end)

        if i_end_fim - i_end_ini < 5:
            i_end_ini = min(len(T) - 1, pk + 5)
            i_end_fim = min(len(T) - 1, pk + 20)

        if i_end_fim > i_end_ini + 2:
            coef_end_base = np.polyfit(T[i_end_ini:i_end_fim],
                                       M_s[i_end_ini:i_end_fim], 1)
        else:
            coef_end_base = np.array([0.0, M_s[min(len(M_s)-1, pk + 5)]])

        # flanco de descida
        i_end_flanco_ini = pk
        i_end_flanco_fim = min(len(T) - 1, pk + janela_end)
        if i_end_flanco_fim > i_end_flanco_ini + 2:
            idx_max_slope_end = i_end_flanco_ini + np.argmax(
                dM_s[i_end_flanco_ini:i_end_flanco_fim + 1])
            hw = max(3, janela_end // 3)
            i0 = max(0, idx_max_slope_end - hw)
            i1 = min(len(T) - 1, idx_max_slope_end + hw)
            coef_tang_end = np.polyfit(T[i0:i1+1], M_s[i0:i1+1], 1)
        else:
            coef_tang_end = coef_end_base

        if abs(coef_tang_end[0] - coef_end_base[0]) > 1e-9:
            T_endset = (coef_end_base[1] - coef_tang_end[1]) / (
                coef_tang_end[0] - coef_end_base[0])
        else:
            T_endset = T[i_end_fim]

        T_endset = float(np.clip(T_endset, T_pico, T[min(len(T)-1, i_end_fim)]))

        # massa nos pontos de onset e endset (interpolada na curva suavizada)
        M_onset  = float(np.interp(T_onset,  T, M_s))
        M_endset = float(np.interp(T_endset, T, M_s))
        perda    = _perda_monotonica(T, M_s, T_onset, T_endset)

        # NÃO filtra por min_perda aqui — eventos pequenos podem ser parte de
        # um grupo maior após mesclagem. O filtro é aplicado depois.
        eventos.append({
            "pico_temp":   float(T_pico),
            "onset_temp":  T_onset,
            "endset_temp": T_endset,
            "massa_onset": M_onset,
            "massa_endset": M_endset,
            "perda_massa": perda,
        })

    # mescla eventos com regiões sobrepostas e só então filtra por min_perda
    eventos = _mesclar_eventos_sobrepostos(eventos, T, M_s)
    eventos = [e for e in eventos if e["perda_massa"] >= min_perda]

    return eventos


# RESOURCE PATH (PyInstaller)
import sys

def resource_path(relative_path):
    """Resolve o caminho de recursos — funciona tanto em desenvolvimento
    quanto empacotado com PyInstaller (--onefile ou --onedir)."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# PALETA
BG       = "#0f1117"
SURFACE  = "#1a1d27"
CARD     = "#22263a"
ACCENT   = "#4f8ef7"
ACCENT2  = "#7c3aed"
SUCCESS  = "#22c55e"
ERROR    = "#ef4444"
TEXT     = "#e2e8f0"
TEXT_DIM = "#64748b"
BORDER   = "#2d3148"

ACCENT_TGA  = "#1f6cf1"
ACCENT_DTG  = "#2fb8df"

FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_HEAD  = ("Segoe UI", 11, "bold")
FONT_BODY  = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO  = ("Consolas", 9)

ESTILOS_LINHA = {
    "Solida":      "-",
    "Tracejada":   "--",
    "Pontilhada":  ":",
    "Traco-ponto": "-.",
}

CORES_TAB10 = [
    "#0a19a5", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#d10718", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# MODELO

class Dataset:
    _contador = 0

    def __init__(self, nome, time, temp, tga, drtga):
        self.nome      = nome
        self.time      = time
        self.temp      = temp
        self.tga       = tga
        self.drtga     = drtga
        self.linestyle = "-"
        self.linewidth = 1.6
        idx = Dataset._contador % len(CORES_TAB10)
        self.color      = CORES_TAB10[idx]
        # cor independente para a curva DrTGA (default: laranja para contraste)
        dtg_idx = (Dataset._contador + 5) % len(CORES_TAB10)
        self.color_dtg  = CORES_TAB10[dtg_idx]
        Dataset._contador += 1


class TGAModel:
    def __init__(self):
        self.datasets: list[Dataset] = []
        self.titulo        = "Analise Termogravimetrica"
        self.xlabel        = "Temperatura (graus C)"
        self.ylabel_tga    = "Massa residual (%)"
        self.ylabel_tga_mg = "Massa (mg)"
        self.ylabel_dtga   = {
            "mg_min": "Perda de massa (mg/min)",
            "mg_s":   "Perda de massa (mg/s)",
            "mg_c":   "Perda de massa (mg/\u00b0C)",
            "pct_min": "Perda de massa (%/min)",
            "pct_s":   "Perda de massa (%/s)",
            "pct_c":   "Perda de massa (%/\u00b0C)",
        }
        self.mostrar_grid   = True
        self.grid_x         = True    # linhas verticais
        self.grid_y         = True    # linhas horizontais
        self.grid_intervalo = 0.0     # 0 = automático; >0 = passo fixo em °C ou s
        # tamanhos de fonte
        self.fonte_titulo   = 11
        self.fonte_eixo     = 10
        self.fonte_ticks    = 9
        self.fonte_legenda  = 9
        self.fonte_eventos  = 7
        self.smoothing      = False
        self.smooth_window  = 11
        self.smooth_poly    = 3
        self.x_mode         = "temp"
        self.tga_y_mode     = "percent"
        self.dtga_y_mode    = "mg_min"
        # suavizacao gaussiana para a derivada calculada numericamente
        self.dtg_sigma      = 10    # sigma do filtro gaussiano (1 = minimo, 100 = maximo)
        self.heating_rate   = 10.0  # taxa de aquecimento em °C/min (padrão TGA)
        # posicao da legenda
        self.legend_loc     = "best"
        # análise de eventos (onset/endset automático)
        self.eventos_ativo      = False
        self.eventos_n_max      = 3
        self.eventos_min_perda  = 1.0   # % mínimo de perda para considerar evento
        # intervalos manuais: lista de dicts {"T_ini": float, "T_fim": float}
        self.intervalos_manuais: list[dict] = []
        # controla se o rotulo grafico do intervalo manual mostra tambem Dm onset/endset
        self.mostrar_dm_on_end_rotulo = True

    def rotulo_tga_y(self):
        return self.ylabel_tga if self.tga_y_mode == "percent" else self.ylabel_tga_mg

    def rotulo_dtga_y(self):
        return self.ylabel_dtga.get(self.dtga_y_mode, self.ylabel_dtga["mg_min"])

    def carregar(self, caminho):
        lines = None
        for enc in ("utf-16", "utf-16-le", "utf-8", "latin-1"):
            try:
                with open(caminho, encoding=enc, errors="replace") as f:
                    lines = f.readlines()
                break
            except Exception:
                continue
        if lines is None:
            raise ValueError("Nao foi possivel decodificar o arquivo.")

        lines = [l.replace("\x00", "").rstrip() for l in lines]

        data_start = None
        for i, line in enumerate(lines):
            parts = line.split()
            if parts == ["Time", "Temp", "TGA", "DrTGA"]:
                data_start = i + 2
                break
        if data_start is None:
            raise ValueError("Cabecalho 'Time Temp TGA DrTGA' nao encontrado.")

        # Extrair taxa de aquecimento do cabeçalho (formato: "20.00\t1000.0\t0")
        # Linha com dois números: primeiro é °C/min (1-100), segundo é T_hold (100-2000)
        for line in lines[:data_start]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    hr_cand  = float(parts[0])
                    hold_cand = float(parts[1])
                    if 1.0 <= hr_cand <= 100.0 and 100.0 <= hold_cand <= 2000.0:
                        self.heating_rate = hr_cand
                        break
                except ValueError:
                    continue

        time_l, temp_l, tga_l, drtga_l = [], [], [], []
        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                time_l.append(float(parts[0]))
                temp_l.append(float(parts[1]))
                tga_l.append(float(parts[2]))
                drtga_l.append(float(parts[3]))
            except ValueError:
                continue

        if not time_l:
            raise ValueError("Nenhum dado numerico encontrado apos o cabecalho.")

        nome = os.path.basename(caminho)
        for ext in (".txt", ".TXT", ".dat", ".DAT", ".tad", ".TAD"):
            nome = nome.replace(ext, "")

        self.datasets.append(Dataset(
            nome,
            np.array(time_l),
            np.array(temp_l),
            np.array(tga_l),
            np.array(drtga_l),
        ))

    def remover(self, idx):
        if 0 <= idx < len(self.datasets):
            self.datasets.pop(idx)

    def limpar(self):
        self.datasets.clear()
        Dataset._contador = 0


# WIDGETS AUXILIARES

def _lighten(hex_color, amount=30):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(min(255, r+amount),
                                         min(255, g+amount),
                                         min(255, b+amount))

def btn(parent, text, command, color=ACCENT, width=None):
    kw = dict(text=text, command=command, bg=color, fg="white",
              activebackground=_lighten(color), activeforeground="white",
              font=FONT_HEAD, relief="flat", cursor="hand2", padx=14, pady=8)
    if width:
        kw["width"] = width
    b = tk.Button(parent, **kw)
    b.bind("<Enter>", lambda e: b.config(bg=_lighten(color)))
    b.bind("<Leave>", lambda e: b.config(bg=color))
    return b

def lbl(parent, text, font=FONT_BODY, fg=TEXT, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg,
                    bg=parent["bg"], **kw)

def card(parent, **kw):
    return tk.Frame(parent, bg=CARD, highlightthickness=1,
                    highlightbackground=BORDER, **kw)

def sep(parent):
    return tk.Frame(parent, bg=BORDER, height=1)


# CANVAS EMBUTIDO

class GraficoEmbutido:
    def __init__(self, parent_frame, model: TGAModel):
        self.model       = model
        self.parent      = parent_frame
        self._modo_atual = None
        self._ultimo_relatorio = []
        self._ultimo_relatorio_manual = []

        # Estado para arrastar anotações
        # Cada entry: {"annot": Annotation, "ax": Axes, "x_data": float, "y_data": float}
        self._annots: list[dict] = []
        self._drag_annot = None
        self._drag_offset = (0.0, 0.0)
        self._cid_press   = None
        self._cid_release = None
        self._cid_motion  = None

        self.fig = Figure(figsize=(7, 4.5), dpi=96)
        self.fig.patch.set_facecolor("white")
        self.ax  = self.fig.add_subplot(111)
        self.ax2 = None   # eixo direito para modo "ambos"
        self._estilizar()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(parent_frame, bg="white")
        toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.config(bg="white")
        self.toolbar.update()

        self._mostrar_placeholder()

    def _reset_axes(self):
        """Remove todos os axes e recria ax principal limpo."""
        self._desconectar_drag()
        self._annots.clear()
        self.fig.clear()
        self.ax  = self.fig.add_subplot(111)
        self.ax2 = None

    # Drag helpers (padrão H-DMAPlot)

    def _conectar_drag(self):
        self._desconectar_drag()
        self._cid_press   = self.canvas.mpl_connect("button_press_event",   self._on_drag_press)
        self._cid_release = self.canvas.mpl_connect("button_release_event", self._on_drag_release)
        self._cid_motion  = self.canvas.mpl_connect("motion_notify_event",  self._on_drag_motion)

    def _desconectar_drag(self):
        for attr in ("_cid_press", "_cid_release", "_cid_motion"):
            cid = getattr(self, attr, None)
            if cid is not None:
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
            setattr(self, attr, None)

    def _toolbar_ativa(self) -> bool:
        return bool(getattr(self.toolbar, "mode", ""))

    @staticmethod
    def _pixel_para_dados(ax, x_pixel, y_pixel):
        return ax.transData.inverted().transform((x_pixel, y_pixel))

    def _on_drag_press(self, event):
        if event.button != 1 or self._toolbar_ativa():
            return
        try:
            renderer = self.canvas.get_renderer()
        except Exception:
            return
        for info in self._annots:
            ann  = info["annot"]
            bbox = ann.get_window_extent(renderer)
            pad  = 6
            if (bbox.x0 - pad <= event.x <= bbox.x1 + pad and
                    bbox.y0 - pad <= event.y <= bbox.y1 + pad):
                self._drag_annot = info
                ann_xy    = ann.get_position()
                click_xy  = self._pixel_para_dados(info["ax"], event.x, event.y)
                self._drag_offset = (ann_xy[0] - click_xy[0],
                                     ann_xy[1] - click_xy[1])
                return

    def _on_drag_release(self, event):
        self._drag_annot = None

    def _on_drag_motion(self, event):
        if self._drag_annot is None or self._toolbar_ativa():
            return
        if event.xdata is None or event.ydata is None:
            return
        info = self._drag_annot
        ax   = info["ax"]
        ex, ey = self._pixel_para_dados(ax, event.x, event.y)
        novo_x = ex + self._drag_offset[0]
        novo_y = ey + self._drag_offset[1]
        ann = info["annot"]
        ann.set_position((novo_x, novo_y))
        # seta permanece apontando ao ponto original na curva
        ann.xy = (info["x_data"], info["y_data"])
        self.canvas.draw_idle()

    def _estilizar(self):
        ax = self.ax
        ax.set_facecolor("white")
        ax.tick_params(colors="#333333", labelsize=9)
        ax.xaxis.label.set_color("#333333")
        ax.yaxis.label.set_color("#333333")
        ax.title.set_color("#333333")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("#aaaaaa")
            sp.set_linewidth(0.8)

    def _aplicar_grid(self, ax, ax2=None):
        """Aplica configuração de grade ao(s) eixo(s) conforme TGAModel."""
        m = self.model
        for _ax in ([ax] if ax2 is None else [ax, ax2]):
            _ax.grid(False)

        mostrar = m.grid_x or m.grid_y
        if not mostrar:
            return

        # intervalo fixo no eixo X?
        if m.grid_intervalo > 0:
            x_min, x_max = ax.get_xlim()
            inicio = math.floor(x_min / m.grid_intervalo) * m.grid_intervalo
            ticks_x = np.arange(inicio, x_max + m.grid_intervalo, m.grid_intervalo)
            ax.set_xticks(ticks_x)

        gs = {"color": "#eeeeee", "linestyle": "--", "linewidth": 0.6, "alpha": 0.85}
        if m.grid_x and m.grid_y:
            ax.grid(True, which="major", **gs)
        elif m.grid_x:
            ax.xaxis.grid(True, which="major", **gs)
            ax.yaxis.grid(False)
        elif m.grid_y:
            ax.yaxis.grid(True, which="major", **gs)
            ax.xaxis.grid(False)

    def _mostrar_placeholder(self):
        self._reset_axes()
        self._estilizar()
        self.ax.text(0.5, 0.5,
                     "Carregue um arquivo .txt\ne clique em  Plotar TGA  ou  Plotar DrTGA",
                     ha="center", va="center",
                     transform=self.ax.transAxes,
                     fontsize=12, color="#aaaaaa",
                     fontfamily="Segoe UI")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.canvas.draw_idle()

    def _eixo_x(self, ds):
        if self.model.x_mode == "temp":
            return ds.temp, self.model.xlabel
        return ds.time, "Tempo (s)"

    def _y_suavizado(self, y):
        if self.model.smoothing and _SCIPY_OK and len(y) > self.model.smooth_window:
            return savgol_filter(y, self.model.smooth_window, self.model.smooth_poly)
        return y

    def _massa_inicial(self, ds):
        if ds.tga[0] != 0:
            return ds.tga[0]
        massa_nao_zero = ds.tga[ds.tga != 0]
        return massa_nao_zero[0] if len(massa_nao_zero) else 1.0

    def _converter_tga_y(self, ds):
        tga_raw = self._y_suavizado(ds.tga)
        if self.model.tga_y_mode == "mg":
            return tga_raw
        return (tga_raw / self._massa_inicial(ds)) * 100.0

    def _calcular_dtg(self, ds):
        """
        Calcula a derivada da curva TGA numericamente usando filtro gaussiano,
        convertendo o resultado para a unidade escolhida no painel.
        """
        modo = self.model.dtga_y_mode
        por_tempo = modo in ("mg_min", "mg_s", "pct_min", "pct_s")
        x_orig = ds.time if por_tempo else ds.temp

        if not _SCIPY_OK:
            dtg = np.gradient(ds.tga, x_orig)
            dtg = np.where(np.isfinite(dtg), dtg, 0.0)
            return self._converter_dtg_unidade(dtg, ds)

        # passo 1: remove duplicatas
        _, unique_idx = np.unique(x_orig, return_index=True)
        x_u   = x_orig[unique_idx]
        tga_u = ds.tga[unique_idx].astype(float)

        sigma = max(1.0, float(self.model.dtg_sigma))

        # passo 2: suaviza o TGA com gaussiano antes de derivar
        tga_s = gaussian_filter1d(tga_u, sigma=sigma)

        # passo 3: derivada numerica
        dtg = np.gradient(tga_s, x_u)

        # passo 4: remove NaN/Inf
        mask_ok = np.isfinite(dtg)
        if not np.all(mask_ok):
            dtg = np.interp(x_u, x_u[mask_ok], dtg[mask_ok])

        # passo 5: suaviza a derivada com sigma ainda maior
        dtg_s = gaussian_filter1d(dtg, sigma=sigma * 2.0)

        # passo 6: interpola de volta para o eixo original
        dtg_final = np.interp(x_orig, x_u, dtg_s)

        return self._converter_dtg_unidade(dtg_final, ds)

    def _converter_dtg_unidade(self, dtg, ds):
        modo = self.model.dtga_y_mode
        if modo in ("mg_min", "pct_min"):
            dtg = dtg * 60.0
        if modo.startswith("pct"):
            dtg = (dtg / self._massa_inicial(ds)) * 100.0
        return dtg

    def _anotar_eventos(self, ax, ds, y_tga):
        """
        Detecta e anota onset/endset/perda de massa na curva TGA.
        As caixas são arrastáveis (padrão H-DMAPlot): clique e arraste.
        Retorna a lista de eventos encontrados.
        """
        if not self.model.eventos_ativo or not _SCIPY_OK:
            return []

        temp      = ds.temp
        massa_ini = self._massa_inicial(ds)
        tga_pct   = (ds.tga / massa_ini) * 100.0

        eventos = detectar_eventos_tga(
            temp, tga_pct,
            n_eventos=self.model.eventos_n_max,
            sigma=max(3, self.model.dtg_sigma // 2),
            min_perda=self.model.eventos_min_perda,
        )

        em_mg = (self.model.tga_y_mode == "mg")
        ymin, ymax = ax.get_ylim()
        y_range    = (ymax - ymin) or 1.0
        dy         = y_range * 0.03

        for i, ev in enumerate(eventos):
            cor   = "#e05c00"
            T_on  = ev["onset_temp"]
            T_end = ev["endset_temp"]
            M_on  = ev["massa_onset"]  if not em_mg else ev["massa_onset"]  * massa_ini / 100.0
            M_end = ev["massa_endset"] if not em_mg else ev["massa_endset"] * massa_ini / 100.0

            # linhas verticais tracejadas
            ax.axvline(T_on,  color=cor, linestyle=":", linewidth=1.0, alpha=0.7)
            ax.axvline(T_end, color=cor, linestyle=":", linewidth=1.0, alpha=0.7)

            # pontos marcadores na curva
            ax.plot(T_on,  M_on,  "o", color=cor, markersize=6, zorder=5)
            ax.plot(T_end, M_end, "s", color=cor, markersize=6, zorder=5)

            # seta dupla horizontal onset <-> endset
            y_seta = (M_on + M_end) / 2.0
            ax.annotate("", xy=(T_end, y_seta), xytext=(T_on, y_seta),
                        arrowprops=dict(arrowstyle="<->", color=cor,
                                        lw=1.2, mutation_scale=12))

            # perda de massa — caixa arrastável ancorada no meio do intervalo
            T_meio  = (T_on + T_end) / 2.0
            unidade = "mg" if em_mg else "%"
            perda_disp = ev["perda_massa"] if not em_mg else ev["perda_massa"] * massa_ini / 100.0
            ann_dm = ax.annotate(
                f"\u0394m={perda_disp:.2f} {unidade}",
                xy=(T_meio, y_seta),
                xytext=(T_meio, y_seta - dy * 2.5),
                fontsize=self.model.fonte_eventos + 0.5,
                color=cor, ha="center", va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fff8f0", ec=cor,
                          alpha=0.92, lw=0.7),
                arrowprops=dict(arrowstyle="-|>", color=cor, lw=0.8,
                                mutation_scale=7),
                annotation_clip=False, zorder=10,
            )
            self._annots.append({"annot": ann_dm, "ax": ax,
                                  "x_data": T_meio, "y_data": y_seta})

            # T_onset — caixa arrastável ancorada no ponto onset
            ann_on = ax.annotate(
                f"T_on={T_on:.1f}\u00b0C",
                xy=(T_on, M_on),
                xytext=(T_on, M_on + dy * 1.5),
                fontsize=self.model.fonte_eventos,
                color=cor, ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=cor,
                          alpha=0.88, lw=0.7),
                arrowprops=dict(arrowstyle="-|>", color=cor, lw=0.8,
                                mutation_scale=7),
                annotation_clip=False, zorder=10,
            )
            self._annots.append({"annot": ann_on, "ax": ax,
                                  "x_data": T_on, "y_data": M_on})

            # T_endset — caixa arrastável ancorada no ponto endset
            ann_end = ax.annotate(
                f"T_end={T_end:.1f}\u00b0C",
                xy=(T_end, M_end),
                xytext=(T_end, M_end + dy * 1.5),
                fontsize=self.model.fonte_eventos,
                color=cor, ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=cor,
                          alpha=0.88, lw=0.7),
                arrowprops=dict(arrowstyle="-|>", color=cor, lw=0.8,
                                mutation_scale=7),
                annotation_clip=False, zorder=10,
            )
            self._annots.append({"annot": ann_end, "ax": ax,
                                  "x_data": T_end, "y_data": M_end})

        return eventos

    def _anotar_intervalos_manuais(self, ax, ax2=None):
        """
        Para cada intervalo manual calcula e anota onset/endset/midpoint/perda.
        As caixas de texto são arrastáveis: clique e arraste qualquer caixa.
        Retorna lista de resultados.
        """
        if not self.model.intervalos_manuais or not self.model.datasets:
            return []

        resultados = []
        CORES_EV = ["#c2410c", "#0369a1", "#15803d", "#7e22ce",
                    "#b45309", "#be123c", "#0f766e"]

        ymin, ymax = ax.get_ylim()
        y_range    = (ymax - ymin) or 1.0
        dy         = y_range * 0.028

        for ds in self.model.datasets:
            massa_ini = self._massa_inicial(ds)
            em_mg     = (self.model.tga_y_mode == "mg")

            for i_iv, iv in enumerate(self.model.intervalos_manuais):
                T_ini = iv["T_ini"]
                T_fim = iv["T_fim"]
                cor   = CORES_EV[i_iv % len(CORES_EV)]

                res = analisar_intervalo_manual(
                    ds.temp, ds.tga, T_ini, T_fim,
                    sigma=max(3, self.model.dtg_sigma // 2),
                    drtga_raw=ds.drtga,
                    heating_rate=self.model.heating_rate,
                )
                if res is None:
                    continue

                fator       = (massa_ini / 100.0) if em_mg else 1.0
                M_on        = res["massa_onset"]  * fator
                M_end       = res["massa_endset"] * fator
                M_mid_massa = res["massa_mid"]    * fator
                T_on        = res["onset_temp"]
                T_end       = res["endset_temp"]
                T_mid       = res["midpoint_temp"]
                perda       = res["perda_massa"]  * fator
                perda_bruta = res["perda_bruta"]  * fator

                # região sombreada do intervalo
                ax.axvspan(T_ini, T_fim, alpha=0.07, color=cor, zorder=0)

                # linhas verticais onset / endset / mid
                ax.axvline(T_on,  color=cor, linestyle="--", linewidth=1.1, alpha=0.85)
                ax.axvline(T_end, color=cor, linestyle="--", linewidth=1.1, alpha=0.85)
                ax.axvline(T_mid, color=cor, linestyle=":",  linewidth=0.9, alpha=0.65)

                # pontos marcadores na curva 
                ax.plot(T_on,  M_on,        "o", color=cor, markersize=7,
                        zorder=6, markeredgecolor="white", markeredgewidth=0.7)
                ax.plot(T_end, M_end,        "s", color=cor, markersize=7,
                        zorder=6, markeredgecolor="white", markeredgewidth=0.7)
                ax.plot(T_mid, M_mid_massa,  "^", color=cor, markersize=6,
                        zorder=6, markeredgecolor="white", markeredgewidth=0.7)

                # seta dupla onset <-> endset (decorativa, não arrastável) 
                y_seta = M_end - dy * 2.5
                ax.annotate("", xy=(T_end, y_seta), xytext=(T_on, y_seta),
                            arrowprops=dict(arrowstyle="<->", color=cor,
                                            lw=1.2, mutation_scale=12))

                # número do intervalo (fixo, no canto) 
                ax.text(T_ini + (T_fim - T_ini) * 0.05, ymax - dy,
                        f"#{i_iv + 1}",
                        fontsize=8, color=cor, ha="left", va="top",
                        fontweight="bold", alpha=0.8)

                # helper para criar caixa arrastável com seta ao ponto âncora
                def _add_annot(texto, x_anchor, y_anchor, x_text, y_text,
                               fc="white", fs=7, va="bottom"):
                    ann = ax.annotate(
                        texto,
                        xy=(x_anchor, y_anchor),
                        xytext=(x_text,  y_text),
                        fontsize=fs,
                        color=cor, ha="center", va=va, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=cor,
                                  alpha=0.92, lw=0.7),
                        arrowprops=dict(arrowstyle="-|>", color=cor, lw=0.8,
                                        mutation_scale=7),
                        annotation_clip=False, zorder=10,
                    )
                    self._annots.append({"annot": ann, "ax": ax,
                                          "x_data": x_anchor, "y_data": y_anchor})

                # T_onset arrastável 
                _add_annot(f"T_on={T_on:.1f}\u00b0C",
                           T_on, M_on,
                           T_on, M_on + dy * 1.6)

                # T_endset arrastável 
                _add_annot(f"T_end={T_end:.1f}\u00b0C",
                           T_end, M_end,
                           T_end, M_end + dy * 1.6)

                # T_midpoint arrastável 
                _add_annot(f"T_mid={T_mid:.1f}\u00b0C",
                           T_mid, M_mid_massa,
                           T_mid, M_mid_massa + dy * 1.6,
                           fc="#f0f9ff")

                # Dm arrastável (ancorado no meio da seta)
                T_meio  = (T_on + T_end) / 2.0
                unid    = "mg" if em_mg else "%"
                texto_perda = f"\u0394m bruto={perda_bruta:.2f} {unid}"
                if self.model.mostrar_dm_on_end_rotulo:
                    texto_perda += f"\non/end={perda:.2f} {unid}"
                _add_annot(texto_perda,
                           T_meio, y_seta,
                           T_meio, y_seta - dy * 2.0,
                           fc="#fffbeb", fs=8, va="top")

                resultados.append({
                    "intervalo": i_iv + 1,
                    "amostra":   ds.nome,
                    "T_ini":     T_ini,
                    "T_fim":     T_fim,
                    **res,
                    "em_mg":     em_mg,
                    "massa_ini_mg": float(massa_ini),
                })

        return resultados

    def redesenhar(self):
        if not self.model.datasets or self._modo_atual is None:
            self._mostrar_placeholder()
            return
        if self._modo_atual == "ambos":
            self.plotar_ambos()
        else:
            self._plotar(self._modo_atual)

    def _plotar(self, modo):
        self._modo_atual = modo
        self._reset_axes()
        ax = self.ax
        self._estilizar()

        is_tga = (modo == "tga")
        if is_tga:
            ylabel = self.model.rotulo_tga_y()
        else:
            ylabel = self.model.rotulo_dtga_y()

        for ds in self.model.datasets:
            x, xlabel = self._eixo_x(ds)
            if is_tga:
                y = self._converter_tga_y(ds)
            else:
                y = self._calcular_dtg(ds)
            ax.plot(x, y,
                    linewidth=ds.linewidth,
                    linestyle=ds.linestyle,
                    color=ds.color if is_tga else ds.color_dtg,
                    label=ds.nome)

        # anotação de onset/endset (apenas no modo TGA)
        if is_tga and self.model.eventos_ativo:
            todos_eventos = []
            for ds in self.model.datasets:
                y = self._converter_tga_y(ds)
                evs = self._anotar_eventos(ax, ds, y)
                todos_eventos.extend(evs)
            self._ultimo_relatorio = todos_eventos
        else:
            self._ultimo_relatorio = []

        # anotação de intervalos manuais (TGA e ambos)
        if is_tga and self.model.intervalos_manuais:
            self._ultimo_relatorio_manual = self._anotar_intervalos_manuais(ax)
        else:
            if not hasattr(self, "_ultimo_relatorio_manual"):
                self._ultimo_relatorio_manual = []

        ax.set_xlabel(xlabel, fontsize=self.model.fonte_eixo)
        ax.set_ylabel(ylabel, fontsize=self.model.fonte_eixo)
        ax.set_title(self.model.titulo, fontsize=self.model.fonte_titulo, fontweight="bold")
        ax.tick_params(axis="both", labelsize=self.model.fonte_ticks)
        ax.legend(facecolor="white", edgecolor="#dddddd",
                  labelcolor="#222222", fontsize=self.model.fonte_legenda, framealpha=0.9,
                  loc=self.model.legend_loc)

        self._aplicar_grid(ax)

        self.fig.tight_layout()
        if self._annots:
            self._conectar_drag()
        self.canvas.draw_idle()

    def plotar_tga(self):
        if self.model.datasets:
            self._plotar("tga")

    def plotar_dtga(self):
        if self.model.datasets:
            self._plotar("dtga")

    def plotar_ambos(self):
        if not self.model.datasets:
            return
        self._modo_atual = "ambos"
        self._reset_axes()

        ax1 = self.ax              # eixo esquerdo — TGA (% ou mg)
        ax2 = ax1.twinx()          # eixo direito  — DrTGA na unidade escolhida
        self.ax2 = ax2

        for ax in (ax1, ax2):
            ax.set_facecolor("white")
            ax.tick_params(labelsize=self.model.fonte_ticks)
        for sp in ax1.spines.values():
            sp.set_edgecolor("#aaaaaa")
            sp.set_linewidth(0.8)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#aaaaaa")
            sp.set_linewidth(0.8)

        x_label     = self.model.xlabel if self.model.x_mode == "temp" else "Tempo (s)"
        y_label_dtg = self.model.rotulo_dtga_y()

        lines_all = []
        all_dtg = []
        for ds in self.model.datasets:
            x, _ = self._eixo_x(ds)

            y_tga = self._converter_tga_y(ds)
            y_dtg = self._calcular_dtg(ds)
            all_dtg.append(y_dtg)

            l1, = ax1.plot(x, y_tga,
                           linewidth=ds.linewidth, linestyle=ds.linestyle,
                           color=ds.color,     label=f"{ds.nome} — TGA")
            l2, = ax2.plot(x, y_dtg,
                           linewidth=ds.linewidth, linestyle="--",
                           color=ds.color_dtg, label=f"{ds.nome} — DrTGA")
            lines_all += [l1, l2]

        if all_dtg:
            import numpy as _np
            dtg_concat = _np.concatenate(all_dtg)
            dtg_abs    = max(abs(dtg_concat.min()), abs(dtg_concat.max()))
            margin     = (dtg_abs * 1.15) if dtg_abs > 0 else 1.0
            ax2.set_ylim(-margin, margin)

        # anotação de onset/endset no eixo TGA (esquerdo)
        if self.model.eventos_ativo:
            todos_eventos = []
            for ds in self.model.datasets:
                y = self._converter_tga_y(ds)
                evs = self._anotar_eventos(ax1, ds, y)
                todos_eventos.extend(evs)
            self._ultimo_relatorio = todos_eventos
        else:
            self._ultimo_relatorio = []

        # anotação de intervalos manuais no eixo TGA
        if self.model.intervalos_manuais:
            self._ultimo_relatorio_manual = self._anotar_intervalos_manuais(ax1)
        else:
            if not hasattr(self, "_ultimo_relatorio_manual"):
                self._ultimo_relatorio_manual = []

        ax1.set_xlabel(x_label,              fontsize=self.model.fonte_eixo, color="#333333")
        ax1.set_ylabel(self.model.rotulo_tga_y(), fontsize=self.model.fonte_eixo, color="#333333")
        ax2.set_ylabel(y_label_dtg,           fontsize=self.model.fonte_eixo, color="#555555")
        ax1.tick_params(axis="y", colors="#333333", labelsize=self.model.fonte_ticks)
        ax1.tick_params(axis="x", labelsize=self.model.fonte_ticks)
        ax2.tick_params(axis="y", colors="#555555", labelsize=self.model.fonte_ticks)

        self.ax.set_title(self.model.titulo, fontsize=self.model.fonte_titulo, fontweight="bold", color="#333333")

        labels = [l.get_label() for l in lines_all]
        ax1.legend(lines_all, labels,
                   facecolor="white", edgecolor="#dddddd",
                   labelcolor="#222222", fontsize=self.model.fonte_legenda, framealpha=0.9,
                   loc=self.model.legend_loc)

        self._aplicar_grid(ax1)

        self.fig.tight_layout()
        if self._annots:
            self._conectar_drag()
        self.canvas.draw_idle()

    def exportar_png(self, pasta, modo):
        if not self.model.datasets or self._modo_atual is None:
            return None
        sufixo = {"tga": "TGA", "dtga": "DrTGA", "ambos": "TGA_DrTGA"}.get(modo, "grafico")
        nome    = self.model.datasets[0].nome if len(self.model.datasets) == 1 else "grafico"
        arquivo = f"{nome}_{sufixo}.png"
        caminho = os.path.join(pasta, arquivo)
        self.fig.savefig(caminho, dpi=300, facecolor="white", bbox_inches="tight")
        return caminho

    # popups de edição
    def _popup(self, root, titulo_janela, texto_atual, callback):
        popup = tk.Toplevel(root)
        popup.title(titulo_janela)
        popup.configure(bg="white")
        popup.resizable(False, False)
        popup.grab_set()
        popup.attributes("-topmost", True)
        popup.update_idletasks()
        pw, ph = 380, 115
        sw, sh = popup.winfo_screenwidth(), popup.winfo_screenheight()
        popup.geometry(f"{pw}x{ph}+{(sw-pw)//2}+{(sh-ph)//2}")

        tk.Label(popup, text=titulo_janela, font=("Segoe UI", 9, "bold"),
                 fg="#555555", bg="white").pack(anchor="w", padx=14, pady=(10, 2))

        var = tk.StringVar(value=texto_atual)
        entry = tk.Entry(popup, textvariable=var, font=("Segoe UI", 11),
                         bg="white", fg="#111111", insertbackground="#111111",
                         relief="flat", highlightthickness=1,
                         highlightcolor="#4f8ef7", highlightbackground="#cccccc")
        entry.pack(fill="x", padx=14, pady=4, ipady=5)
        entry.select_range(0, "end")
        entry.focus_set()

        def _ok(event=None):
            novo = var.get().strip()
            popup.destroy()
            if novo:
                callback(novo)
            self.canvas.draw_idle()

        def _cancel(event=None):
            popup.destroy()

        bf = tk.Frame(popup, bg="white")
        bf.pack(anchor="e", padx=14, pady=(2, 10))
        tk.Button(bf, text="Cancelar", command=_cancel, bg="#f1f5f9",
                  fg="#555555", font=("Segoe UI", 9), relief="flat",
                  padx=10, pady=4, cursor="hand2").pack(side="left", padx=4)
        tk.Button(bf, text="  OK  ", command=_ok, bg="#4f8ef7",
                  fg="white", font=("Segoe UI", 9, "bold"), relief="flat",
                  padx=10, pady=4, cursor="hand2").pack(side="left")

        entry.bind("<Return>", _ok)
        entry.bind("<Escape>", _cancel)
        popup.wait_window()

    def editar_titulo(self, root):
        def _set(novo):
            self.model.titulo = novo
            self.ax.set_title(novo, fontsize=self.model.fonte_titulo, fontweight="bold")
            self.canvas.draw_idle()
        self._popup(root, "Editar titulo", self.model.titulo, _set)

    def editar_xlabel(self, root):
        def _set(novo):
            self.model.xlabel = novo
            if self._modo_atual:
                self.ax.set_xlabel(novo, fontsize=self.model.fonte_eixo)
                self.canvas.draw_idle()
        self._popup(root, "Editar rotulo - Eixo X", self.model.xlabel, _set)

    def editar_ylabel(self, root):
        if self._modo_atual == "tga":
            atual = self.model.rotulo_tga_y()
        elif self._modo_atual == "ambos":
            atual = self.model.rotulo_tga_y()
        else:
            atual = self.model.rotulo_dtga_y()

        def _set(novo):
            if self._modo_atual == "tga":
                if self.model.tga_y_mode == "percent":
                    self.model.ylabel_tga = novo
                else:
                    self.model.ylabel_tga_mg = novo
            elif self._modo_atual == "ambos":
                if self.model.tga_y_mode == "percent":
                    self.model.ylabel_tga = novo
                else:
                    self.model.ylabel_tga_mg = novo
            else:
                self.model.ylabel_dtga[self.model.dtga_y_mode] = novo
            self.redesenhar()

        self._popup(root, "Editar rotulo - Eixo Y", atual, _set)

    def editar_nome_amostra(self, root, idx):
        if idx is None or idx >= len(self.model.datasets):
            return
        ds = self.model.datasets[idx]
        def _set(novo):
            ds.nome = novo
            self.redesenhar()
        self._popup(root, "Editar nome da amostra", ds.nome, _set)


# INTERFACE PRINCIPAL

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("H-TGAPlot v2.95")
        try:
            self.root.iconbitmap(resource_path("TGAPlot.ico"))
        except Exception:
            pass  # ícone não encontrado — continua sem ele
        self.root.configure(bg=BG)
        self.root.minsize(900, 560)

        self.model = TGAModel()
        self._build()
        self.root.mainloop()

    def _build(self):
        # cabeçalho
        header = tk.Frame(self.root, bg=SURFACE, pady=14)
        header.pack(fill="x")
        tk.Label(header, text="H-TGAPlot",
                 font=FONT_TITLE, fg=ACCENT, bg=SURFACE).pack(side="left", padx=24)
        tk.Label(header, text="Analise Termogravimetrica  —  TGA / DrTGA",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=SURFACE).pack(side="left", padx=4)
        tk.Label(header, text="Autor: Carlos Henrique Amaro da Silva",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=SURFACE).pack(side="right", padx=24)

        # corpo
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        painel = tk.Frame(body, bg=BG, width=210)
        painel.pack(side="left", fill="y", padx=(16, 8), pady=16)
        painel.pack_propagate(False)
        self._build_painel(painel)

        direita = tk.Frame(body, bg=BG)
        direita.pack(side="left", fill="both", expand=True, padx=(0, 16), pady=16)
        self._build_workspace(direita)

        # barra de status
        self.status_var = tk.StringVar(value="Pronto. Carregue um arquivo .txt para comecar.")
        sb = tk.Frame(self.root, bg=SURFACE, pady=4)
        sb.pack(fill="x", side="bottom")
        tk.Label(sb, textvariable=self.status_var,
                 font=FONT_SMALL, fg=TEXT_DIM, bg=SURFACE).pack(side="left", padx=16)
        #tk.Label(sb, text="COLOCAR TEXTO AQUI PARA ALGUMA ANOTAÇAO OU INFORMAÇAO",
        #         font=FONT_SMALL, fg=TEXT_DIM, bg=SURFACE).pack(side="right", padx=16)


    def _build_painel(self, parent):
        canvas_p = tk.Canvas(parent, bg=BG, bd=0, highlightthickness=0)
        canvas_p.pack(side="left", fill="both", expand=True)

        sb = tk.Scrollbar(parent, orient="vertical", command=canvas_p.yview)
        sb.pack(side="right", fill="y")
        canvas_p.configure(yscrollcommand=sb.set)

        col = tk.Frame(canvas_p, bg=BG)
        wid = canvas_p.create_window((0, 0), window=col, anchor="nw")

        col.bind("<Configure>",
                 lambda e: canvas_p.configure(scrollregion=canvas_p.bbox("all")))
        canvas_p.bind("<Configure>",
                      lambda e: canvas_p.itemconfig(wid, width=e.width))
        col.bind_all("<MouseWheel>",
                     lambda e: canvas_p.yview_scroll(int(-1*(e.delta/120)), "units"))

        # ARQUIVOS
        lbl(col, "ARQUIVOS", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 6))
        btn(col, "Carregar .txt",       self._carregar, color=ACCENT2).pack(fill="x", pady=2)
        btn(col, "Remover selecionado", self._remover,  color="#7f1d1d").pack(fill="x", pady=2)
        btn(col, "Limpar tudo",         self._limpar,   color="#374151").pack(fill="x", pady=2)

        sep(col).pack(fill="x", pady=10)

        # VISUALIZAÇÃO
        lbl(col, "VISUALIZACAO", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 6))
        btn(col, "Plotar TGA",        self._plotar_tga,   color=ACCENT_TGA).pack(fill="x", pady=2)
        btn(col, "Plotar DrTGA",      self._plotar_dtga,  color=ACCENT_DTG).pack(fill="x", pady=2)
        btn(col, "TGA + DrTGA",       self._plotar_ambos, color="#e20a15").pack(fill="x", pady=2)

        sep(col).pack(fill="x", pady=8)

        # SUAVIZAÇÃO
        lbl(col, "SUAVIZACAO", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 6))

        smooth_card = card(col)
        smooth_card.pack(fill="x", pady=4)

        self.smooth_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            smooth_card, text="Ativar Smoothing",
            variable=self.smooth_var, command=self._on_smooth_toggle,
            bg=CARD, fg=TEXT, selectcolor=BG,
            activebackground=CARD, activeforeground=TEXT,
            font=FONT_SMALL
        ).pack(anchor="w", padx=8, pady=(6, 2))

        tk.Label(smooth_card, text="Janela (pontos):",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack(anchor="w", padx=8)

        self.smooth_win_var = tk.IntVar(value=11)
        tk.Scale(smooth_card, from_=5, to=51, resolution=2,
                 orient="horizontal", variable=self.smooth_win_var,
                 bg=CARD, fg=TEXT, troughcolor=BG,
                 highlightthickness=0, activebackground=ACCENT,
                 command=self._on_smooth_change).pack(fill="x", padx=8, pady=(0, 2))

        self.smooth_desc = tk.Label(smooth_card, text="11 pts - suavizacao leve",
                                    font=("Segoe UI", 8), fg=TEXT_DIM, bg=CARD)
        self.smooth_desc.pack(anchor="w", padx=8, pady=(0, 6))

        sep(col).pack(fill="x", pady=10)

        # SUAVIZACAO DrTGA — filtro gaussiano
        lbl(col, "SUAVIZACAO DrTGA", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))
        tk.Label(col, text="Filtro gaussiano aplicado\na derivada numerica do TGA",
                 font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG,
                 justify="left").pack(anchor="w", padx=4, pady=(0, 4))

        dtg_card = card(col)
        dtg_card.pack(fill="x", pady=4)

        tk.Label(dtg_card, text="Sigma (intensidade):",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=CARD).pack(anchor="w", padx=8, pady=(8, 2))

        self.dtg_sigma_var = tk.IntVar(value=10)
        tk.Scale(dtg_card, from_=1, to=100, resolution=1,
                 orient="horizontal", variable=self.dtg_sigma_var,
                 bg=CARD, fg=TEXT, troughcolor=BG,
                 highlightthickness=0, activebackground=ACCENT,
                 command=self._on_dtg_smooth_change).pack(fill="x", padx=8, pady=(0, 2))

        self.dtg_smooth_desc = tk.Label(dtg_card, text="sigma=10 — suavizacao media",
                                         font=("Segoe UI", 8), fg=TEXT_DIM, bg=CARD)
        self.dtg_smooth_desc.pack(anchor="w", padx=8, pady=(0, 6))

        sep(col).pack(fill="x", pady=10)

        # ANÁLISE MANUAL DE INTERVALO
        lbl(col, "ANALISE DE INTERVALO", font=("Segoe UI", 8, "bold"),
            fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))
        tk.Label(col,
                 text="Defina intervalos de temperatura\npara calcular onset, endset,\nmidpoint e perda de massa.",
                 font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG,
                 justify="left").pack(anchor="w", padx=4, pady=(0, 4))

        iv_card = card(col)
        iv_card.pack(fill="x", pady=4)

        # Entradas T_ini / T_fim
        r_iv0 = tk.Frame(iv_card, bg=CARD)
        r_iv0.pack(fill="x", padx=8, pady=(8, 2))
        tk.Label(r_iv0, text="T inicio (°C):", font=FONT_SMALL,
                 fg=TEXT_DIM, bg=CARD, width=13, anchor="w").pack(side="left")
        self.iv_tini_var = tk.StringVar(value="")
        tk.Entry(r_iv0, textvariable=self.iv_tini_var, width=7,
                 bg=BG, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=FONT_SMALL).pack(side="left", padx=4)

        r_iv1 = tk.Frame(iv_card, bg=CARD)
        r_iv1.pack(fill="x", padx=8, pady=2)
        tk.Label(r_iv1, text="T fim (°C):", font=FONT_SMALL,
                 fg=TEXT_DIM, bg=CARD, width=13, anchor="w").pack(side="left")
        self.iv_tfim_var = tk.StringVar(value="")
        tk.Entry(r_iv1, textvariable=self.iv_tfim_var, width=7,
                 bg=BG, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=FONT_SMALL).pack(side="left", padx=4)

        btn(iv_card, "+ Adicionar intervalo",
            self._adicionar_intervalo, color="#0369a1").pack(
            fill="x", padx=8, pady=(6, 2))

        # Lista de intervalos adicionados
        tk.Label(iv_card, text="Intervalos definidos:",
                 font=("Segoe UI", 8), fg=TEXT_DIM, bg=CARD).pack(
                 anchor="w", padx=8, pady=(6, 2))

        self.iv_listbox = tk.Listbox(
            iv_card, bg=BG, fg=TEXT, font=FONT_MONO,
            selectbackground=ACCENT2, selectforeground="white",
            relief="flat", bd=0, highlightthickness=0,
            height=4, activestyle="none"
        )
        self.iv_listbox.pack(fill="x", padx=8, pady=(0, 2))

        btn(iv_card, "Remover selecionado",
            self._remover_intervalo, color="#7f1d1d").pack(
            fill="x", padx=8, pady=2)
        btn(iv_card, "Limpar todos",
            self._limpar_intervalos, color="#374151").pack(
            fill="x", padx=8, pady=(2, 4))
        self.btn_toggle_on_end = btn(iv_card, "Rotulo on/end  ON",
            self._toggle_rotulo_on_end, color="#0e7490")
        self.btn_toggle_on_end.pack(fill="x", padx=8, pady=2)
        btn(iv_card, "Ver resultados",
            self._ver_relatorio_manual, color="#92400e").pack(
            fill="x", padx=8, pady=(2, 8))

        sep(col).pack(fill="x", pady=10)

        # ANÁLISE DE EVENTOS (ONSET / ENDSET AUTOMÁTICO)
        lbl(col, "ONSET / ENDSET (AUTO)", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))

        ev_card = card(col)
        ev_card.pack(fill="x", pady=4)

        self.eventos_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            ev_card, text="Detectar eventos",
            variable=self.eventos_var, command=self._on_eventos_toggle,
            bg=CARD, fg=TEXT, selectcolor=BG,
            activebackground=CARD, activeforeground=TEXT,
            font=FONT_SMALL
        ).pack(anchor="w", padx=8, pady=(6, 2))

        # Nº máximo de eventos
        r_ev1 = tk.Frame(ev_card, bg=CARD)
        r_ev1.pack(fill="x", padx=8, pady=2)
        tk.Label(r_ev1, text="Máx. eventos:", font=FONT_SMALL,
                 fg=TEXT_DIM, bg=CARD).pack(side="left")
        self.eventos_n_var = tk.IntVar(value=3)
        tk.Spinbox(r_ev1, from_=1, to=6, textvariable=self.eventos_n_var,
                   command=self._on_eventos_n_change, width=4,
                   bg=BG, fg=TEXT, insertbackground=TEXT,
                   relief="flat", font=FONT_SMALL).pack(side="left", padx=6)

        # Perda mínima
        r_ev2 = tk.Frame(ev_card, bg=CARD)
        r_ev2.pack(fill="x", padx=8, pady=2)
        tk.Label(r_ev2, text="Perda mín. (%):", font=FONT_SMALL,
                 fg=TEXT_DIM, bg=CARD).pack(side="left")
        self.eventos_perda_var = tk.DoubleVar(value=1.0)
        tk.Spinbox(r_ev2, from_=0.5, to=20.0, increment=0.5,
                   textvariable=self.eventos_perda_var,
                   command=self._on_eventos_perda_change, width=5,
                   bg=BG, fg=TEXT, insertbackground=TEXT,
                   relief="flat", font=FONT_SMALL).pack(side="left", padx=6)

        btn(ev_card, "Ver relatório de eventos",
            self._ver_relatorio, color="#92400e").pack(fill="x", padx=8, pady=(4, 8))

        sep(col).pack(fill="x", pady=10)

        # EIXO X
        lbl(col, "EIXO X", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))
        self.xmode_var = tk.StringVar(value="temp")
        for label, val in [("Temperatura (graus C)", "temp"), ("Tempo (s)", "time")]:
            tk.Radiobutton(
                col, text=label, variable=self.xmode_var, value=val,
                command=self._on_xmode,
                font=FONT_SMALL, fg=TEXT, bg=BG,
                activebackground=BG, activeforeground=ACCENT,
                selectcolor=CARD, relief="flat", bd=0,
            ).pack(anchor="w", padx=4, pady=1)

        sep(col).pack(fill="x", pady=10)

        # EIXO Y - TGA
        lbl(col, "EIXO Y - TGA", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))
        self.tga_ymode_var = tk.StringVar(value="percent")
        for label, val in [("Massa residual (%)", "percent"), ("Massa (mg)", "mg")]:
            tk.Radiobutton(
                col, text=label, variable=self.tga_ymode_var, value=val,
                command=self._on_tga_ymode,
                font=FONT_SMALL, fg=TEXT, bg=BG,
                activebackground=BG, activeforeground=ACCENT,
                selectcolor=CARD, relief="flat", bd=0,
            ).pack(anchor="w", padx=4, pady=1)

        sep(col).pack(fill="x", pady=10)

        # EIXO Y - DrTGA
        lbl(col, "EIXO Y - DrTGA", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))
        self.dtga_ymode_var = tk.StringVar(value="mg_min")
        dtga_opcoes = [
            ("mg/min", "mg_min"),
            ("mg/s", "mg_s"),
            ("mg/graus C", "mg_c"),
            ("%/min", "pct_min"),
            ("%/s", "pct_s"),
            ("%/graus C", "pct_c"),
        ]
        for label, val in dtga_opcoes:
            tk.Radiobutton(
                col, text=label, variable=self.dtga_ymode_var, value=val,
                command=self._on_dtga_ymode,
                font=FONT_SMALL, fg=TEXT, bg=BG,
                activebackground=BG, activeforeground=ACCENT,
                selectcolor=CARD, relief="flat", bd=0,
            ).pack(anchor="w", padx=4, pady=1)

        sep(col).pack(fill="x", pady=10)

        # GRADE DO GRÁFICO
        lbl(col, "GRADE DO GRAFICO", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))

        grid_card = card(col)
        grid_card.pack(fill="x", pady=4)

        # Checkboxes X e Y
        chk_row = tk.Frame(grid_card, bg=CARD)
        chk_row.pack(fill="x", padx=8, pady=(8, 2))
        self.grid_x_var = tk.BooleanVar(value=True)
        self.grid_y_var = tk.BooleanVar(value=True)
        tk.Checkbutton(chk_row, text="Linhas X", variable=self.grid_x_var,
                       bg=CARD, fg=TEXT, selectcolor=BG,
                       activebackground=CARD, font=FONT_SMALL,
                       command=self._on_grid_change).pack(side="left", padx=(0, 8))
        tk.Checkbutton(chk_row, text="Linhas Y", variable=self.grid_y_var,
                       bg=CARD, fg=TEXT, selectcolor=BG,
                       activebackground=CARD, font=FONT_SMALL,
                       command=self._on_grid_change).pack(side="left")

        # Intervalo fixo
        intv_row = tk.Frame(grid_card, bg=CARD)
        intv_row.pack(fill="x", padx=8, pady=(4, 8))
        tk.Label(intv_row, text="Intervalo:", font=FONT_SMALL,
                 fg=TEXT_DIM, bg=CARD).pack(side="left")
        INTERVALOS_GRID = ["Auto", "1", "2", "5", "10", "20", "25", "50", "100"]
        self.grid_intervalo_var = tk.StringVar(value="Auto")
        from tkinter import ttk as _ttk
        intv_combo = _ttk.Combobox(intv_row, textvariable=self.grid_intervalo_var,
                                   values=INTERVALOS_GRID, state="readonly",
                                   font=FONT_SMALL, width=7)
        intv_combo.pack(side="left", padx=6)
        intv_combo.bind("<<ComboboxSelected>>", lambda e: self._on_grid_change())

        sep(col).pack(fill="x", pady=10)

        # TAMANHO DE FONTES
        lbl(col, "TAMANHO DE FONTES", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))

        font_card = card(col)
        font_card.pack(fill="x", pady=4)

        def _font_row(parent, label, attr_name, default):
            row = tk.Frame(parent, bg=CARD)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=label, font=FONT_SMALL, fg=TEXT_DIM,
                     bg=CARD, width=10, anchor="w").pack(side="left")
            var = tk.IntVar(value=default)
            setattr(self, attr_name, var)
            lbl_val = tk.Label(row, text=str(default), font=FONT_SMALL,
                               fg=TEXT, bg=CARD, width=3)
            lbl_val.pack(side="right")
            tk.Scale(row, from_=6, to=20, resolution=1, orient="horizontal",
                     variable=var, bg=CARD, fg=TEXT, troughcolor=BG,
                     highlightthickness=0, activebackground=ACCENT,
                     command=lambda v, lv=lbl_val: [
                         lv.config(text=str(int(float(v)))),
                         self._on_fonte_change()
                     ],
                     length=80).pack(side="left", padx=4)

        _font_row(font_card, "Título:",    "fonte_titulo_var",  11)
        _font_row(font_card, "Rót.eixo:", "fonte_eixo_var",    10)
        _font_row(font_card, "Ticks:",     "fonte_ticks_var",    9)
        _font_row(font_card, "Legenda:",   "fonte_legenda_var",  9)
        _font_row(font_card, "Eventos:",   "fonte_eventos_var",  7)
        tk.Frame(font_card, bg=CARD, height=4).pack()  # padding inferior

        sep(col).pack(fill="x", pady=10)

        # POSIÇÃO DA LEGENDA
        lbl(col, "POSICAO DA LEGENDA", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))
        self.legend_loc_var = tk.StringVar(value="best")
        leg_opcoes = [
            ("Automatico",        "best"),
            ("Superior direito",  "upper right"),
            ("Superior esquerdo", "upper left"),
            ("Inferior direito",  "lower right"),
            ("Inferior esquerdo", "lower left"),
            ("Centro direito",    "center right"),
            ("Centro esquerdo",   "center left"),
            ("Superior centro",   "upper center"),
            ("Inferior centro",   "lower center"),
            ("Centro",            "center"),
        ]
        for label, val in leg_opcoes:
            tk.Radiobutton(
                col, text=label, variable=self.legend_loc_var, value=val,
                command=self._on_legend_loc,
                font=FONT_SMALL, fg=TEXT, bg=BG,
                activebackground=BG, activeforeground=ACCENT,
                selectcolor=CARD, relief="flat", bd=0,
            ).pack(anchor="w", padx=4, pady=1)

        sep(col).pack(fill="x", pady=10)

        # EDITAR GRÁFICO
        lbl(col, "EDITAR GRAFICO", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 6))
        btn(col, "Editar titulo",       self._editar_titulo, color="#374151").pack(fill="x", pady=2)
        btn(col, "Editar eixo X",       self._editar_xlabel, color="#374151").pack(fill="x", pady=2)
        btn(col, "Editar eixo Y",       self._editar_ylabel, color="#374151").pack(fill="x", pady=2)
        btn(col, "Editar nome amostra", self._editar_nome,   color="#374151").pack(fill="x", pady=2)

        sep(col).pack(fill="x", pady=10)

        # ESTILO DA AMOSTRA
        lbl(col, "ESTILO SELECIONADO", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 6))

        estilo_card = card(col)
        estilo_card.pack(fill="x")

        r1 = tk.Frame(estilo_card, bg=CARD)
        r1.pack(fill="x", padx=8, pady=(8, 2))
        tk.Label(r1, text="Linha:", font=FONT_SMALL, fg=TEXT_DIM,
                 bg=CARD, width=7, anchor="w").pack(side="left")
        self.estilo_var = tk.StringVar(value="Solida")
        em = tk.OptionMenu(r1, self.estilo_var, *ESTILOS_LINHA.keys(),
                           command=self._on_estilo_change)
        em.config(bg=BG, fg=TEXT, activebackground=ACCENT2,
                  activeforeground="white", font=FONT_SMALL,
                  relief="flat", highlightthickness=0, bd=0)
        em["menu"].config(bg=BG, fg=TEXT, activebackground=ACCENT2,
                          activeforeground="white", font=FONT_SMALL)
        em.pack(side="left", padx=4)

        r2 = tk.Frame(estilo_card, bg=CARD)
        r2.pack(fill="x", padx=8, pady=2)
        tk.Label(r2, text="Cor TGA:", font=FONT_SMALL, fg=TEXT_DIM,
                 bg=CARD, width=7, anchor="w").pack(side="left")
        self.cor_preview = tk.Label(r2, bg=CORES_TAB10[0], width=4,
                                    relief="flat", cursor="hand2")
        self.cor_preview.pack(side="left", padx=4, ipady=6)
        self.cor_preview.bind("<Button-1>", lambda e: self._escolher_cor("tga"))
        tk.Label(r2, text="clicar", font=("Segoe UI", 8),
                 fg=TEXT_DIM, bg=CARD).pack(side="left")

        r2b = tk.Frame(estilo_card, bg=CARD)
        r2b.pack(fill="x", padx=8, pady=2)
        tk.Label(r2b, text="Cor DTG:", font=FONT_SMALL, fg=TEXT_DIM,
                 bg=CARD, width=7, anchor="w").pack(side="left")
        self.cor_dtg_preview = tk.Label(r2b, bg=CORES_TAB10[5], width=4,
                                         relief="flat", cursor="hand2")
        self.cor_dtg_preview.pack(side="left", padx=4, ipady=6)
        self.cor_dtg_preview.bind("<Button-1>", lambda e: self._escolher_cor("dtg"))
        tk.Label(r2b, text="clicar", font=("Segoe UI", 8),
                 fg=TEXT_DIM, bg=CARD).pack(side="left")

        r3 = tk.Frame(estilo_card, bg=CARD)
        r3.pack(fill="x", padx=8, pady=(2, 8))
        tk.Label(r3, text="Espess.:", font=FONT_SMALL, fg=TEXT_DIM,
                 bg=CARD, width=7, anchor="w").pack(side="left")
        self.lw_var = tk.DoubleVar(value=1.6)
        tk.Scale(r3, from_=0.5, to=5.0, resolution=0.5,
                 orient="horizontal", variable=self.lw_var,
                 bg=CARD, fg=TEXT, troughcolor=BG,
                 highlightthickness=0, activebackground=ACCENT,
                 length=100, command=self._on_lw_change).pack(side="left")
        self.lw_label = tk.Label(r3, text="1.6pt", font=FONT_SMALL,
                                  fg=TEXT_DIM, bg=CARD, width=4)
        self.lw_label.pack(side="left", padx=2)
        sep(col).pack(fill="x", pady=10)

        # EXPORTAR
        lbl(col, "EXPORTAR", font=("Segoe UI", 8, "bold"), fg=TEXT_DIM).pack(anchor="w", pady=(0, 6))
        btn(col, "Exportar PNG", self._exportar_png, color="#0e7490").pack(fill="x", pady=2)

    def _build_workspace(self, parent):
        topo = tk.Frame(parent, bg=BG)
        topo.pack(fill="x", pady=(0, 8))

        list_card = card(topo)
        list_card.pack(side="left", fill="both", expand=True)

        tk.Label(list_card, text="Amostras carregadas",
                 font=FONT_HEAD, fg=TEXT, bg=CARD).pack(anchor="w", padx=12, pady=(8, 4))
        sep(list_card).pack(fill="x", padx=8)

        fl = tk.Frame(list_card, bg=CARD)
        fl.pack(fill="both", expand=True, padx=4, pady=4)

        sb_scroll = tk.Scrollbar(fl)
        sb_scroll.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            fl, bg=CARD, fg=TEXT, font=FONT_MONO,
            selectbackground=ACCENT2, selectforeground="white",
            relief="flat", bd=0, highlightthickness=0,
            yscrollcommand=sb_scroll.set, activestyle="none",
            height=4
        )
        self.listbox.pack(fill="both", expand=True)
        sb_scroll.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        info_card = card(topo)
        info_card.pack(side="left", fill="y", padx=(8, 0))

        tk.Label(info_card, text="Informacoes",
                 font=FONT_HEAD, fg=TEXT, bg=CARD).pack(anchor="w", padx=12, pady=(8, 4))
        sep(info_card).pack(fill="x", padx=8)

        self.info_var = tk.StringVar(value="Selecione uma amostra.")
        tk.Label(info_card, textvariable=self.info_var,
                 font=FONT_SMALL, fg=TEXT_DIM, bg=CARD,
                 justify="left", wraplength=260).pack(anchor="w", padx=12, pady=8)

        grafico_card = card(parent)
        grafico_card.pack(fill="both", expand=True)

        self.grafico = GraficoEmbutido(grafico_card, self.model)

    # ações 
    def _carregar(self):
        arquivos = filedialog.askopenfilenames(
            title="Selecionar arquivos TGA (.txt)",
            filetypes=[("Arquivos de texto", "*.txt"), ("Todos", "*.*")]
        )
        for arq in arquivos:
            try:
                self.model.carregar(arq)
                self.listbox.insert(tk.END, f"  {os.path.basename(arq)}")
                self._status(f"Carregado: {os.path.basename(arq)}")
            except Exception as e:
                messagebox.showerror("H-TGAPlot", f"Erro ao carregar:\n{e}")

    def _remover(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("H-TGAPlot", "Selecione uma amostra para remover.")
            return
        idx = sel[0]
        self.model.remover(idx)
        self.listbox.delete(idx)
        self.info_var.set("Selecione uma amostra.")
        self.grafico.redesenhar()
        self._status("Amostra removida.")

    def _limpar(self):
        self.model.limpar()
        self.listbox.delete(0, tk.END)
        self.info_var.set("Selecione uma amostra.")
        self.grafico._modo_atual = None
        self.grafico.redesenhar()
        self._status("Tudo limpo.")

    def _plotar_tga(self):
        if not self.model.datasets:
            messagebox.showwarning("H-TGAPlot", "Carregue ao menos um arquivo .txt primeiro.")
            return
        self.grafico.plotar_tga()
        self._status("Curva TGA plotada.")

    def _plotar_dtga(self):
        if not self.model.datasets:
            messagebox.showwarning("H-TGAPlot", "Carregue ao menos um arquivo .txt primeiro.")
            return
        self.grafico.plotar_dtga()
        self._status("Curva DrTGA plotada.")

    def _plotar_ambos(self):
        if not self.model.datasets:
            messagebox.showwarning("H-TGAPlot", "Carregue ao menos um arquivo .txt primeiro.")
            return
        self.grafico.plotar_ambos()
        self._status("TGA + DrTGA plotados.")

    def _on_grid_change(self):
        raw = self.grid_intervalo_var.get()
        self.model.grid_intervalo = 0.0 if raw == "Auto" else float(raw)
        self.model.grid_x = self.grid_x_var.get()
        self.model.grid_y = self.grid_y_var.get()
        # mantém compatibilidade com mostrar_grid usado internamente
        self.model.mostrar_grid = self.model.grid_x or self.model.grid_y
        self.grafico.redesenhar()

    def _on_fonte_change(self):
        self.model.fonte_titulo  = int(self.fonte_titulo_var.get())
        self.model.fonte_eixo    = int(self.fonte_eixo_var.get())
        self.model.fonte_ticks   = int(self.fonte_ticks_var.get())
        self.model.fonte_legenda = int(self.fonte_legenda_var.get())
        self.model.fonte_eventos = int(self.fonte_eventos_var.get())
        self.grafico.redesenhar()

    def _on_xmode(self):
        self.model.x_mode = self.xmode_var.get()
        self.grafico.redesenhar()

    def _on_tga_ymode(self):
        self.model.tga_y_mode = self.tga_ymode_var.get()
        self.grafico.redesenhar()

    def _on_dtga_ymode(self):
        self.model.dtga_y_mode = self.dtga_ymode_var.get()
        self.grafico.redesenhar()

    def _on_legend_loc(self):
        self.model.legend_loc = self.legend_loc_var.get()
        self.grafico.redesenhar()
        self._status(f"Legenda: {self.legend_loc_var.get()}")

    def _exportar_png(self):
        if not self.model.datasets or self.grafico._modo_atual is None:
            messagebox.showwarning("H-TGAPlot", "Plote uma curva antes de exportar.")
            return
        pasta = filedialog.askdirectory(title="Pasta para salvar PNG")
        if not pasta:
            return
        caminho = self.grafico.exportar_png(pasta, self.grafico._modo_atual)
        if caminho:
            messagebox.showinfo("H-TGAPlot", f"PNG salvo em:\n{caminho}")
            self._status(f"PNG exportado: {caminho}")

    def _idx_sel(self):
        sel = self.listbox.curselection()
        if not sel:
            return None
        idx = sel[0]
        return idx if idx < len(self.model.datasets) else None

    def _on_select(self, event):
        idx = self._idx_sel()
        if idx is None:
            return
        ds = self.model.datasets[idx]
        self.info_var.set(
            f"Nome:   {ds.nome}\n"
            f"Pontos: {len(ds.time)}\n"
            f"Temp:   {ds.temp.min():.1f} - {ds.temp.max():.1f} C\n"
            f"TGA:    {ds.tga.min():.3f} - {ds.tga.max():.3f} mg\n"
            f"DrTGA:  {ds.drtga.min():.4f} - {ds.drtga.max():.4f} mg/s"
        )
        nome_estilo = next((k for k, v in ESTILOS_LINHA.items()
                            if v == ds.linestyle), "Solida")
        self.estilo_var.set(nome_estilo)
        self.lw_var.set(ds.linewidth)
        self.lw_label.config(text=f"{ds.linewidth:.1f}pt")
        self.cor_preview.config(bg=ds.color)
        self.cor_dtg_preview.config(bg=ds.color_dtg)

    def _on_estilo_change(self, valor):
        idx = self._idx_sel()
        if idx is None:
            return
        self.model.datasets[idx].linestyle = ESTILOS_LINHA[valor]
        self.grafico.redesenhar()

    def _on_lw_change(self, val):
        idx = self._idx_sel()
        if idx is None:
            return
        lw = float(val)
        self.model.datasets[idx].linewidth = lw
        self.lw_label.config(text=f"{lw:.1f}pt")
        self.grafico.redesenhar()

    def _escolher_cor(self, curva="tga"):
        from tkinter.colorchooser import askcolor
        idx = self._idx_sel()
        if idx is None:
            messagebox.showwarning("H-TGAPlot", "Selecione uma amostra na lista primeiro.")
            return
        ds = self.model.datasets[idx]
        cor_atual = ds.color if curva == "tga" else ds.color_dtg
        titulo    = f"Cor TGA — {ds.nome}" if curva == "tga" else f"Cor DrTGA — {ds.nome}"
        resultado = askcolor(color=cor_atual, title=titulo, parent=self.root)
        if resultado and resultado[1]:
            if curva == "tga":
                ds.color     = resultado[1]
                self.cor_preview.config(bg=resultado[1])
            else:
                ds.color_dtg = resultado[1]
                self.cor_dtg_preview.config(bg=resultado[1])
            self.grafico.redesenhar()
            self._status(f"Cor {'TGA' if curva == 'tga' else 'DrTGA'} de '{ds.nome}' alterada.")

    def _editar_titulo(self):
        self.grafico.editar_titulo(self.root)

    def _editar_xlabel(self):
        self.grafico.editar_xlabel(self.root)

    def _editar_ylabel(self):
        self.grafico.editar_ylabel(self.root)

    def _editar_nome(self):
        idx = self._idx_sel()
        if idx is None:
            messagebox.showwarning("H-TGAPlot", "Selecione uma amostra na lista primeiro.")
            return
        self.grafico.editar_nome_amostra(self.root, idx)


    # Intervalos manuais
    def _adicionar_intervalo(self):
        if not self.model.datasets:
            messagebox.showwarning("H-TGAPlot",
                "Carregue ao menos um arquivo .txt primeiro.")
            return
        try:
            T_ini = float(self.iv_tini_var.get().replace(",", "."))
            T_fim = float(self.iv_tfim_var.get().replace(",", "."))
        except ValueError:
            messagebox.showwarning("H-TGAPlot",
                "Digite valores numericos validos para T inicio e T fim.")
            return
        if T_ini >= T_fim:
            messagebox.showwarning("H-TGAPlot",
                "T inicio deve ser menor que T fim.")
            return
        for iv in self.model.intervalos_manuais:
            if not (T_fim <= iv["T_ini"] or T_ini >= iv["T_fim"]):
                if not messagebox.askyesno("H-TGAPlot",
                        f"Este intervalo sobrepoe o intervalo "
                        f"#{self.model.intervalos_manuais.index(iv)+1} "
                        f"({iv['T_ini']:.1f}-{iv['T_fim']:.1f} C).\n"
                        "Deseja adicionar mesmo assim?"):
                    return
                break
        self.model.intervalos_manuais.append({"T_ini": T_ini, "T_fim": T_fim})
        n = len(self.model.intervalos_manuais)
        self.iv_listbox.insert(tk.END,
            f"  #{n}  {T_ini:.1f} - {T_fim:.1f} C")
        self.iv_tini_var.set("")
        self.iv_tfim_var.set("")
        if self.grafico._modo_atual:
            self.grafico.redesenhar()
        self._status(f"Intervalo #{n} adicionado: {T_ini:.1f}-{T_fim:.1f} C")

    def _remover_intervalo(self):
        sel = self.iv_listbox.curselection()
        if not sel:
            messagebox.showwarning("H-TGAPlot",
                "Selecione um intervalo na lista para remover.")
            return
        idx = sel[0]
        self.model.intervalos_manuais.pop(idx)
        self.iv_listbox.delete(0, tk.END)
        for i, iv in enumerate(self.model.intervalos_manuais):
            self.iv_listbox.insert(tk.END,
                f"  #{i+1}  {iv['T_ini']:.1f} - {iv['T_fim']:.1f} C")
        if self.grafico._modo_atual:
            self.grafico.redesenhar()
        self._status("Intervalo removido.")

    def _limpar_intervalos(self):
        self.model.intervalos_manuais.clear()
        self.iv_listbox.delete(0, tk.END)
        self.grafico._ultimo_relatorio_manual = []
        if self.grafico._modo_atual:
            self.grafico.redesenhar()
        self._status("Todos os intervalos removidos.")

    def _toggle_rotulo_on_end(self):
        self.model.mostrar_dm_on_end_rotulo = not self.model.mostrar_dm_on_end_rotulo
        ativo = self.model.mostrar_dm_on_end_rotulo
        self.btn_toggle_on_end.config(
            text="Rotulo on/end  ON" if ativo else "Rotulo on/end  OFF",
            bg="#0e7490" if ativo else "#374151"
        )
        if self.grafico._modo_atual:
            self.grafico.redesenhar()
        self._status(
            "Dm onset/endset visivel no rotulo."
            if ativo else
            "Dm onset/endset oculto do rotulo grafico."
        )

    def _ver_relatorio_manual(self):
        res_lista = getattr(self.grafico, "_ultimo_relatorio_manual", [])
        if not res_lista:
            if not self.model.datasets or not self.model.intervalos_manuais:
                messagebox.showinfo("H-TGAPlot",
                    "Nenhum resultado disponivel.\n\n"
                    "Defina ao menos um intervalo e plote a curva TGA.")
                return
            if self.grafico._modo_atual in ("tga", "ambos"):
                self.grafico.redesenhar()
                res_lista = getattr(self.grafico, "_ultimo_relatorio_manual", [])
            if not res_lista:
                messagebox.showinfo("H-TGAPlot",
                    "Plote a curva TGA (ou TGA + DrTGA) para calcular os resultados.")
                return

        popup = tk.Toplevel(self.root)
        popup.title("Resultados - Analise de Intervalo Manual")
        popup.configure(bg="white")
        popup.resizable(True, True)
        popup.grab_set()
        popup.attributes("-topmost", True)
        n_rows = len(res_lista)
        pw, ph = 980, 280 + n_rows * 34
        sw, sh = popup.winfo_screenwidth(), popup.winfo_screenheight()
        popup.geometry(f"{pw}x{min(ph,640)}+{(sw-pw)//2}+{(sh-min(ph,640))//2}")

        tk.Label(popup, text="Resultados - Analise de Intervalo Manual",
                 font=("Segoe UI", 12, "bold"), fg="#1e293b",
                 bg="white").pack(pady=(16, 2))
        tk.Label(popup,
                 text="Metodo: intersecao de tangentes  |  Midpoint: T onde perda = 50% do evento",
                 font=("Segoe UI", 9), fg="#64748b", bg="white").pack()
        tk.Label(popup,
                 text="Dm (onset->endset): perda monotonica  |  Dm bruto (T_ini->T_fim): perda total no intervalo",
                 font=("Segoe UI", 8), fg="#94a3b8", bg="white").pack()
        tk.Frame(popup, bg="#e2e8f0", height=1).pack(fill="x", padx=16, pady=10)

        frame_outer = tk.Frame(popup, bg="white")
        frame_outer.pack(fill="both", expand=True, padx=16)
        sb_h = tk.Scrollbar(frame_outer, orient="horizontal")
        sb_h.pack(side="bottom", fill="x")
        tbl = tk.Canvas(frame_outer, bg="white", xscrollcommand=sb_h.set)
        tbl.pack(side="left", fill="both", expand=True)
        sb_h.config(command=tbl.xview)
        inner = tk.Frame(tbl, bg="white")
        tbl.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: tbl.configure(scrollregion=tbl.bbox("all")))

        COLS = ["#", "Amostra", "T ini", "T fim",
                "T_onset", "T_endset", "T_midpoint", "T_pico",
                "M ini", "M fim", "M onset", "M endset",
                "Dm (onset->end)", "Dm bruto (ini->fim)"]
        WIDTHS = [30, 130, 70, 70, 82, 84, 90, 80, 90, 90, 90, 90, 112, 130]

        def _make_cell(parent, text, font, fg, bg, width, pady=3):
            """Célula de largura fixa em pixels com Label centralizado."""
            f = tk.Frame(parent, bg=bg, width=width, height=22)
            f.pack_propagate(False)
            f.pack(side="left", padx=1, pady=pady)
            tk.Label(f, text=text, font=font, fg=fg, bg=bg,
                     anchor="center").pack(fill="both", expand=True)

        hdr_f = tk.Frame(inner, bg="#f1f5f9")
        hdr_f.pack(fill="x")
        for c, w in zip(COLS, WIDTHS):
            _make_cell(hdr_f, c,
                       ("Segoe UI", 8, "bold"), "#374151", "#f1f5f9", w, pady=4)

        for i, res in enumerate(res_lista):
            row_bg = "white" if i % 2 == 0 else "#f8fafc"
            row = tk.Frame(inner, bg=row_bg)
            row.pack(fill="x")
            em_mg    = res.get("em_mg", False)
            m_ini_mg = res.get("massa_ini_mg", 1.0)
            fator    = (m_ini_mg / 100.0) if em_mg else 1.0
            unid     = "mg" if em_mg else "%"
            perda_bruta = res["perda_bruta"]
            M_ini_iv = res.get("massa_ini_intervalo")
            M_fim_iv = res.get("massa_fim_intervalo")
            vals = [
                f"#{res['intervalo']}",
                res["amostra"],
                f"{res['T_ini']:.1f}C",
                f"{res['T_fim']:.1f}C",
                f"{res['onset_temp']:.1f}C",
                f"{res['endset_temp']:.1f}C",
                f"{res['midpoint_temp']:.1f}C",
                f"{res['pico_temp']:.1f}C",
                "" if M_ini_iv is None else f"{M_ini_iv*fator:.3f} {unid}",
                "" if M_fim_iv is None else f"{M_fim_iv*fator:.3f} {unid}",
                f"{res['massa_onset']*fator:.3f} {unid}",
                f"{res['massa_endset']*fator:.3f} {unid}",
                f"{res['perda_massa']*fator:.3f} {unid}",
                f"{perda_bruta*fator:.3f} {unid}",
            ]
            for v, w in zip(vals, WIDTHS):
                _make_cell(row, v, ("Consolas", 8), "#1e293b", row_bg, w)

        tk.Frame(popup, bg="#e2e8f0", height=1).pack(fill="x", padx=16, pady=6)

        def _copiar_tsv():
            header = "\t".join(COLS)
            linhas = [header]
            for res in res_lista:
                em_mg    = res.get("em_mg", False)
                m_ini_mg = res.get("massa_ini_mg", 1.0)
                fator    = (m_ini_mg / 100.0) if em_mg else 1.0
                perda_bruta = res["perda_bruta"]
                M_ini_iv = res.get("massa_ini_intervalo")
                M_fim_iv = res.get("massa_fim_intervalo")
                linhas.append(
                    f"#{res['intervalo']}\t{res['amostra']}\t"
                    f"{res['T_ini']:.1f}\t{res['T_fim']:.1f}\t"
                    f"{res['onset_temp']:.1f}\t{res['endset_temp']:.1f}\t"
                    f"{res['midpoint_temp']:.1f}\t{res['pico_temp']:.1f}\t"
                    f"{'' if M_ini_iv is None else f'{M_ini_iv*fator:.3f}'}\t"
                    f"{'' if M_fim_iv is None else f'{M_fim_iv*fator:.3f}'}\t"
                    f"{res['massa_onset']*fator:.3f}\t"
                    f"{res['massa_endset']*fator:.3f}\t"
                    f"{res['perda_massa']*fator:.3f}\t"
                    f"{perda_bruta*fator:.3f}"
                )
            popup.clipboard_clear()
            popup.clipboard_append("\n".join(linhas))
            self._status("Tabela copiada (TSV) - pronto para colar no Excel.")

        btn_f = tk.Frame(popup, bg="white")
        btn_f.pack(pady=(0, 14))
        tk.Button(btn_f, text="Copiar como TSV (Excel)",
                  command=_copiar_tsv,
                  bg="#0e7490", fg="white", font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=12, pady=6, cursor="hand2").pack(
                  side="left", padx=6)
        tk.Button(btn_f, text="Fechar", command=popup.destroy,
                  bg="#374151", fg="white", font=("Segoe UI", 9),
                  relief="flat", padx=12, pady=6, cursor="hand2").pack(
                  side="left", padx=6)

    def _on_smooth_toggle(self):
        self.model.smoothing = self.smooth_var.get()
        if not _SCIPY_OK and self.model.smoothing:
            messagebox.showwarning("H-TGAPlot",
                "scipy nao encontrado.\nInstale com:  pip install scipy")
            self.smooth_var.set(False)
            self.model.smoothing = False
            return
        if self.model.datasets:
            self.grafico.redesenhar()
        self._status(f"Smoothing {'ativado' if self.model.smoothing else 'desativado'}.")

    def _on_smooth_change(self, val):
        w = int(float(val))
        if w % 2 == 0:
            w += 1
        self.model.smooth_window = w
        self.smooth_win_var.set(w)
        if w <= 11:
            desc = f"{w} pts - suavizacao leve"
        elif w <= 25:
            desc = f"{w} pts - suavizacao media"
        else:
            desc = f"{w} pts - suavizacao forte"
        self.smooth_desc.config(text=desc)
        if self.model.smoothing and self.model.datasets:
            self.grafico.redesenhar()

    def _on_dtg_smooth_change(self, val):
        sigma = int(float(val))
        self.model.dtg_sigma = sigma
        if sigma <= 5:
            desc = f"sigma={sigma} — suavizacao leve"
        elif sigma <= 25:
            desc = f"sigma={sigma} — suavizacao media"
        elif sigma <= 60:
            desc = f"sigma={sigma} — suavizacao forte"
        else:
            desc = f"sigma={sigma} — suavizacao maxima"
        self.dtg_smooth_desc.config(text=desc)
        if self.model.datasets and self.grafico._modo_atual in ("dtga", "ambos"):
            self.grafico.redesenhar()

    def _on_eventos_toggle(self):
        self.model.eventos_ativo = self.eventos_var.get()
        if not _SCIPY_OK and self.model.eventos_ativo:
            messagebox.showwarning("H-TGAPlot",
                "scipy nao encontrado.\nInstale com:  pip install scipy")
            self.eventos_var.set(False)
            self.model.eventos_ativo = False
            return
        if self.model.datasets and self.grafico._modo_atual:
            self.grafico.redesenhar()
        self._status(f"Onset/Endset {'ativado' if self.model.eventos_ativo else 'desativado'}.")

    def _on_eventos_n_change(self):
        try:
            self.model.eventos_n_max = int(self.eventos_n_var.get())
        except Exception:
            pass
        if self.model.eventos_ativo and self.model.datasets and self.grafico._modo_atual:
            self.grafico.redesenhar()

    def _on_eventos_perda_change(self):
        try:
            self.model.eventos_min_perda = float(self.eventos_perda_var.get())
        except Exception:
            pass
        if self.model.eventos_ativo and self.model.datasets and self.grafico._modo_atual:
            self.grafico.redesenhar()

    def _ver_relatorio(self):
        """Exibe uma janela com a tabela de eventos detectados."""
        evs = getattr(self.grafico, "_ultimo_relatorio", [])
        if not evs:
            messagebox.showinfo("H-TGAPlot",
                "Nenhum evento detectado.\n\n"
                "Certifique-se de que:\n"
                "• A opção 'Detectar eventos' está ativada\n"
                "• Uma curva TGA foi plotada\n"
                "• Há eventos com perda acima do mínimo configurado.")
            return

        popup = tk.Toplevel(self.root)
        popup.title("Relatório de Eventos TGA")
        popup.configure(bg="white")
        popup.resizable(True, True)
        popup.grab_set()
        popup.attributes("-topmost", True)

        pw, ph = 620, 300 + len(evs) * 32
        sw, sh = popup.winfo_screenwidth(), popup.winfo_screenheight()
        popup.geometry(f"{pw}x{min(ph,580)}+{(sw-pw)//2}+{(sh-min(ph,580))//2}")

        tk.Label(popup, text="Eventos de Decomposição Detectados",
                 font=("Segoe UI", 12, "bold"), fg="#1e293b",
                 bg="white").pack(pady=(16, 4))
        tk.Label(popup, text="Método: interseção de tangentes (onset/endset)",
                 font=("Segoe UI", 9), fg="#64748b", bg="white").pack()

        sep_f = tk.Frame(popup, bg="#e2e8f0", height=1)
        sep_f.pack(fill="x", padx=16, pady=10)

        # cabeçalho da tabela
        cols = ["Evento", "T_pico (°C)", "T_onset (°C)", "T_endset (°C)",
                "Massa onset", "Massa endset", "Δm (%)"]
        widths = [60, 100, 100, 110, 100, 110, 80]

        hdr = tk.Frame(popup, bg="#f1f5f9")
        hdr.pack(fill="x", padx=16)
        for c, w in zip(cols, widths):
            tk.Label(hdr, text=c, font=("Segoe UI", 9, "bold"),
                     fg="#374151", bg="#f1f5f9", width=w//8,
                     relief="flat", anchor="center").pack(side="left", padx=2, pady=4)

        em_mg = (self.model.tga_y_mode == "mg")
        for i, ev in enumerate(evs):
            row_bg = "white" if i % 2 == 0 else "#f8fafc"
            row = tk.Frame(popup, bg=row_bg)
            row.pack(fill="x", padx=16)

            massa_ini = 1.0
            if self.model.datasets:
                ds = self.model.datasets[0]
                if ds.tga[0] != 0:
                    massa_ini = ds.tga[0]
                else:
                    nz = ds.tga[ds.tga != 0]
                    if len(nz):
                        massa_ini = nz[0]

            M_on  = ev["massa_onset"]  if not em_mg else ev["massa_onset"]  * massa_ini / 100.0
            M_end = ev["massa_endset"] if not em_mg else ev["massa_endset"] * massa_ini / 100.0
            perda = ev["perda_massa"]  if not em_mg else ev["perda_massa"]  * massa_ini / 100.0
            unid  = "%" if not em_mg else "mg"

            valores = [
                f"#{i+1}",
                f"{ev['pico_temp']:.1f}",
                f"{ev['onset_temp']:.1f}",
                f"{ev['endset_temp']:.1f}",
                f"{M_on:.2f} {unid}",
                f"{M_end:.2f} {unid}",
                f"{perda:.2f} {unid}",
            ]
            for v, w in zip(valores, widths):
                tk.Label(row, text=v, font=("Consolas", 9),
                         fg="#1e293b", bg=row_bg, width=w//8,
                         anchor="center").pack(side="left", padx=2, pady=3)

        sep_f2 = tk.Frame(popup, bg="#e2e8f0", height=1)
        sep_f2.pack(fill="x", padx=16, pady=8)

        def _copiar():
            linhas = ["Evento\tT_pico(°C)\tT_onset(°C)\tT_endset(°C)\tMassa_onset\tMassa_endset\tDelta_m"]
            for i, ev in enumerate(evs):
                linhas.append(
                    f"#{i+1}\t{ev['pico_temp']:.1f}\t{ev['onset_temp']:.1f}\t"
                    f"{ev['endset_temp']:.1f}\t{ev['massa_onset']:.2f}\t"
                    f"{ev['massa_endset']:.2f}\t{ev['perda_massa']:.2f}"
                )
            popup.clipboard_clear()
            popup.clipboard_append("\n".join(linhas))
            self._status("Tabela copiada para a área de transferência.")

        btn_f = tk.Frame(popup, bg="white")
        btn_f.pack(pady=(0, 14))
        tk.Button(btn_f, text="Copiar como TSV", command=_copiar,
                  bg="#0e7490", fg="white", font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=12, pady=6, cursor="hand2").pack(side="left", padx=6)
        tk.Button(btn_f, text="Fechar", command=popup.destroy,
                  bg="#374151", fg="white", font=("Segoe UI", 9),
                  relief="flat", padx=12, pady=6, cursor="hand2").pack(side="left", padx=6)

    def _status(self, msg):
        self.status_var.set(msg)


if __name__ == "__main__":
    App()