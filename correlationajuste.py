import pandas as pd
import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp1d

import tkinter as tk
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import style

style.use("dark_background")


# =========================================================
# FUNÇÃO PRINCIPAL
# =========================================================
def analyze_csvs(
        device_path,
        powerlab_path,

        device_time_col,
        device_flow_col,
        device_volume_col,

        powerlab_time_col,
        powerlab_flow_col,
        powerlab_volume_col
):

    # =====================================================
    # LEITURA
    # =====================================================
    device = pd.read_csv(
        device_path,
        sep=';',
        encoding='utf-8-sig'
    )

    powerlab = pd.read_csv(
        powerlab_path,
        sep=','
    )

    device.columns = device.columns.str.strip()
    powerlab.columns = powerlab.columns.str.strip()

    # =====================================================
    # TEMPO
    # =====================================================
    device["tempo_s"] = (
        pd.to_numeric(device[device_time_col], errors='coerce')
        / 1_000_000
    )

    powerlab["tempo_s"] = pd.to_numeric(
        powerlab[powerlab_time_col],
        errors='coerce'
    )

    # =====================================================
    # FLUXO
    # =====================================================
    device["fluxo"] = pd.to_numeric(
        device[device_flow_col],
        errors='coerce'
    )

    powerlab["fluxo"] = pd.to_numeric(
        powerlab[powerlab_flow_col],
        errors='coerce'
    )

    # =====================================================
    # VOLUME (LITROS)
    # =====================================================
    device["volume_l"] = pd.to_numeric(
        device[device_volume_col],
        errors='coerce'
    )

    powerlab["volume_l"] = pd.to_numeric(
        powerlab[powerlab_volume_col],
        errors='coerce'
    )

    # =====================================================
    # REMOVER NaN
    # =====================================================
    device = device.dropna()
    powerlab = powerlab.dropna()

    # =====================================================
    # BASE COMUM
    # =====================================================
    fs = 100
    dt = 1 / fs

    t_min = max(
        device["tempo_s"].min(),
        powerlab["tempo_s"].min()
    )

    t_max = min(
        device["tempo_s"].max(),
        powerlab["tempo_s"].max()
    )

    tempo_comum = np.arange(
        t_min,
        t_max,
        dt
    )

    # =====================================================
    # INTERPOLAÇÃO
    # =====================================================
    f_device = interp1d(
        device["tempo_s"],
        device["fluxo"],
        kind='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    f_powerlab = interp1d(
        powerlab["tempo_s"],
        powerlab["fluxo"],
        kind='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    s_device = f_device(tempo_comum)
    s_powerlab = f_powerlab(tempo_comum)

    # =====================================================
    # REMOVER NaN
    # =====================================================
    mask = (
            ~np.isnan(s_device)
            &
            ~np.isnan(s_powerlab)
    )

    tempo_comum = tempo_comum[mask]

    s_device = s_device[mask]
    s_powerlab = s_powerlab[mask]

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================
    s1 = (
                 s_device - np.mean(s_device)
         ) / np.std(s_device)

    s2 = (
                 s_powerlab - np.mean(s_powerlab)
         ) / np.std(s_powerlab)

    # =====================================================
    # CROSS CORRELATION
    # =====================================================
    corr = correlate(
        s1,
        s2,
        mode='full',
        method='fft'
    )

    lags = np.arange(
        -len(s1) + 1,
        len(s1)
    )

    i = np.argmax(corr)

    if 0 < i < len(corr) - 1:

        y1 = corr[i - 1]
        y2 = corr[i]
        y3 = corr[i + 1]

        denom = (y1 - 2 * y2 + y3)

        delta = (
            0.5 * (y1 - y3) / denom
            if denom != 0 else 0
        )

    else:
        delta = 0

    lag_refinado = lags[i] + delta

    delay = lag_refinado * dt

    # =====================================================
    # CORRIGIR DELAY
    # =====================================================
    powerlab["tempo_corrigido"] = (
            powerlab["tempo_s"] + delay
    )

    # =====================================================
    # INTERPOLAR NOVAMENTE
    # =====================================================
    f_powerlab_sync = interp1d(
        powerlab["tempo_corrigido"],
        powerlab["fluxo"],
        kind='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    s_powerlab_sync = f_powerlab_sync(
        tempo_comum
    )

    s_device_sync = f_device(
        tempo_comum
    )

    # =====================================================
    # INTERPOLAÇÃO DO VOLUME
    # =====================================================
    f_device_volume = interp1d(
        device["tempo_s"],
        device["volume_l"],
        kind='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    f_powerlab_volume = interp1d(
        powerlab["tempo_corrigido"],
        powerlab["volume_l"],
        kind='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    v_device_sync = f_device_volume(
        tempo_comum
    )

    v_powerlab_sync = f_powerlab_volume(
        tempo_comum
    )

    # =====================================================
    # VALIDAR
    # =====================================================
    validos = (
            ~np.isnan(s_powerlab_sync)
            &
            ~np.isnan(v_device_sync)
            &
            ~np.isnan(v_powerlab_sync)
    )

    tempo_sync = tempo_comum[validos]

    s_device_sync = s_device_sync[validos]
    s_powerlab_sync = s_powerlab_sync[validos]

    v_device_sync = v_device_sync[validos]
    v_powerlab_sync = v_powerlab_sync[validos]

    # =====================================================
    # EXTRAIR MINUTO CENTRAL
    # =====================================================
    tempo_total = tempo_sync[-1] - tempo_sync[0]
    tempo_medio = tempo_sync[0] + tempo_total / 2
    
    # Minuto central: 30s antes e 30s depois do meio
    tempo_inicio_central = tempo_medio - 30
    tempo_fim_central = tempo_medio + 30
    
    mask_central = (
        (tempo_sync >= tempo_inicio_central)
        & (tempo_sync <= tempo_fim_central)
    )
    
    tempo_central = tempo_sync[mask_central]
    s_device_central = s_device_sync[mask_central]
    s_powerlab_central = s_powerlab_sync[mask_central]
    v_device_central = v_device_sync[mask_central]
    v_powerlab_central = v_powerlab_sync[mask_central]

    # =====================================================
    # CALIBRAÇÃO FLUXO (MINUTO CENTRAL)
    # =====================================================
    ganho, offset = np.polyfit(
        s_device_central,
        s_powerlab_central,
        1
    )

    s_device_calibrado = (
            ganho * s_device_central
            + offset
    )

    # =====================================================
    # CALIBRAÇÃO VOLUME (MINUTO CENTRAL)
    # =====================================================
    ganho_volume, offset_volume = np.polyfit(
        v_device_central,
        v_powerlab_central,
        1
    )

    v_device_calibrado = (
            ganho_volume * v_device_central
            + offset_volume
    )

    # =====================================================
    # BLAND ALTMAN FLUXO (MINUTO CENTRAL)
    # =====================================================
    media_fluxo = (
            s_device_calibrado
            + s_powerlab_central
    ) / 2

    diferenca_fluxo = (
            s_device_calibrado
            - s_powerlab_central
    )

    bias_fluxo = np.mean(diferenca_fluxo)

    std_fluxo = np.std(
        diferenca_fluxo,
        ddof=1
    )

    loa_superior_fluxo = bias_fluxo + 1.96 * std_fluxo
    loa_inferior_fluxo = bias_fluxo - 1.96 * std_fluxo

    # =====================================================
    # BLAND ALTMAN VOLUME (MINUTO CENTRAL)
    # =====================================================
    media_volume = (
            v_device_calibrado
            + v_powerlab_central
    ) / 2

    diferenca_volume = (
            v_device_calibrado
            - v_powerlab_central
    )

    bias_volume = np.mean(diferenca_volume)

    std_volume = np.std(
        diferenca_volume,
        ddof=1
    )

    loa_superior_volume = bias_volume + 1.96 * std_volume
    loa_inferior_volume = bias_volume - 1.96 * std_volume

    # =====================================================
    # MÉTRICAS (MINUTO CENTRAL)
    # =====================================================
    erro = (
            s_device_calibrado
            - s_powerlab_central
    )

    rmse = np.sqrt(
        np.mean(erro ** 2)
    )

    mae = np.mean(
        np.abs(erro)
    )

    corrcoef = np.corrcoef(
        s_device_calibrado,
        s_powerlab_central
    )[0, 1]

    return (
        tempo_central,

        s_device_calibrado,
        s_powerlab_central,
        
        v_device_calibrado,
        v_powerlab_central,

        media_fluxo,
        diferenca_fluxo,

        bias_fluxo,
        loa_superior_fluxo,
        loa_inferior_fluxo,

        media_volume,
        diferenca_volume,

        bias_volume,
        loa_superior_volume,
        loa_inferior_volume,

        rmse,
        mae,
        corrcoef,

        ganho,
        offset,

        ganho_volume,
        offset_volume,

        delay
    )


# =========================================================
# APP
# =========================================================
class App:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Análise de Correlação e Calibração"
        )

        self.root.geometry("1600x950")
        self.root.configure(bg="#1e1e1e")

        self.device_path = None
        self.powerlab_path = None

        # =================================================
        # CORES
        # =================================================
        BG = "#1e1e1e"
        CARD = "#2b2b2b"
        ACCENT = "#00bcd4"
        TEXT = "#f5f5f5"

        # =================================================
        # HEADER
        # =================================================
        header = tk.Frame(
            root,
            bg=BG
        )

        header.pack(
            fill=tk.X,
            pady=15
        )

        tk.Label(
            header,
            text="Análise de Correlação e Calibração",
            font=("Segoe UI", 24, "bold"),
            fg=ACCENT,
            bg=BG
        ).pack()

        self.status_label = tk.Label(
            header,
            text="Aguardando arquivos...",
            font=("Segoe UI", 10),
            fg="#aaaaaa",
            bg=BG
        )

        self.status_label.pack(
            pady=(5, 0)
        )

        # =================================================
        # MAIN
        # =================================================
        main = tk.Frame(
            root,
            bg=BG
        )

        main.pack(
            fill=tk.BOTH,
            expand=True,
            padx=15,
            pady=10
        )

        # =================================================
        # SIDEBAR
        # =================================================
        sidebar = tk.Frame(
            main,
            bg=CARD,
            width=340
        )

        sidebar.pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=(0, 12)
        )

        sidebar.pack_propagate(False)

        # =================================================
        # BOTÕES
        # =================================================
        self.select_device_btn = tk.Button(
            sidebar,
            text="Selecionar CSV ESP32",
            command=self.select_device,
            bg="#4CAF50",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief=tk.FLAT,
            pady=10,
            cursor="hand2"
        )

        self.select_device_btn.pack(
            fill=tk.X,
            padx=20,
            pady=(20, 10)
        )

        self.select_powerlab_btn = tk.Button(
            sidebar,
            text="Selecionar CSV LabChart",
            command=self.select_powerlab,
            bg="#ff9800",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief=tk.FLAT,
            pady=10,
            cursor="hand2"
        )

        self.select_powerlab_btn.pack(
            fill=tk.X,
            padx=20,
            pady=10
        )

        # =================================================
        # CAMPOS
        # =================================================
        fields = tk.Frame(
            sidebar,
            bg=CARD
        )

        fields.pack(
            fill=tk.X,
            padx=20,
            pady=20
        )

        self.last_entry = None

        self.create_entry(
            fields,
            "Tempo ESP32",
            "timestamp",
            0,
            0
        )

        self.device_time_entry = self.last_entry

        self.create_entry(
            fields,
            "Fluxo ESP32",
            "fluxo",
            0,
            1
        )

        self.device_flow_entry = self.last_entry

        self.create_entry(
            fields,
            "Volume ESP32",
            "volume",
            1,
            0
        )

        self.device_volume_entry = self.last_entry

        self.create_entry(
            fields,
            "Tempo LabChart",
            "timestamp",
            1,
            1
        )

        self.powerlab_time_entry = self.last_entry

        self.create_entry(
            fields,
            "Fluxo LabChart",
            "fluxo_lm",
            2,
            0
        )

        self.powerlab_flow_entry = self.last_entry

        self.create_entry(
            fields,
            "Volume LabChart",
            "volume",
            2,
            1
        )

        self.powerlab_volume_entry = self.last_entry

        # =================================================
        # ANALYZE BUTTON
        # =================================================
        self.analyze_btn = tk.Button(
            sidebar,
            text="GERAR ANÁLISE",
            command=self.analyze,
            state=tk.DISABLED,
            bg=ACCENT,
            fg="white",
            font=("Segoe UI", 13, "bold"),
            relief=tk.FLAT,
            pady=12,
            cursor="hand2"
        )

        self.analyze_btn.pack(
            fill=tk.X,
            padx=20,
            pady=25
        )
        
        # =================================================
        # TOGGLE FLUXO/VOLUME
        # =================================================
        self.plot_type = tk.StringVar(value="fluxo")
        
        toggle_frame = tk.Frame(
            sidebar,
            bg=CARD
        )
        
        toggle_frame.pack(
            fill=tk.X,
            padx=20,
            pady=(0, 20)
        )
        
        tk.Label(
            toggle_frame,
            text="Tipo de Plot:",
            font=("Segoe UI", 9, "bold"),
            fg="white",
            bg=CARD
        ).pack(pady=(5, 10))
        
        tk.Radiobutton(
            toggle_frame,
            text="Fluxo (L/min)",
            variable=self.plot_type,
            value="fluxo",
            command=self.update_plot,
            bg=CARD,
            fg="white",
            activebackground=CARD,
            activeforeground=ACCENT,
            font=("Segoe UI", 10),
            cursor="hand2"
        ).pack(anchor="w", pady=5)
        
        tk.Radiobutton(
            toggle_frame,
            text="Volume (L)",
            variable=self.plot_type,
            value="volume",
            command=self.update_plot,
            bg=CARD,
            fg="white",
            activebackground=CARD,
            activeforeground=ACCENT,
            font=("Segoe UI", 10),
            cursor="hand2"
        ).pack(anchor="w", pady=5)
        
        # Estado para armazenar dados
        self.current_results = None

        # =================================================
        # PLOT AREA
        # =================================================
        self.plot_frame = tk.Frame(
            main,
            bg=CARD
        )

        self.plot_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True
        )

    # =====================================================
    # ENTRY
    # =====================================================
    def create_entry(
            self,
            parent,
            label,
            default,
            row,
            col
    ):

        frame = tk.Frame(
            parent,
            bg="#2b2b2b"
        )

        frame.grid(
            row=row,
            column=col,
            padx=8,
            pady=8,
            sticky="ew"
        )

        tk.Label(
            frame,
            text=label,
            font=("Segoe UI", 9, "bold"),
            fg="white",
            bg="#2b2b2b"
        ).pack(anchor="w")

        entry = tk.Entry(
            frame,
            font=("Consolas", 10),
            bg="#3a3a3a",
            fg="white",
            relief=tk.FLAT,
            insertbackground="white"
        )

        entry.insert(0, default)

        entry.pack(
            fill=tk.X,
            ipady=5
        )

        self.last_entry = entry

    # =====================================================
    # SELECT FILES
    # =====================================================
    def select_device(self):

        self.device_path = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv")]
        )

        if self.device_path:
            self.select_device_btn.config(
                text="✓ ESP32 carregado"
            )

            self.check_ready()

    def select_powerlab(self):

        self.powerlab_path = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv")]
        )

        if self.powerlab_path:
            self.select_powerlab_btn.config(
                text="✓ LabChart carregado"
            )

            self.check_ready()

    # =====================================================
    # READY
    # =====================================================
    def check_ready(self):

        if self.device_path and self.powerlab_path:
            self.analyze_btn.config(
                state=tk.NORMAL
            )

    # =====================================================
    # ANALYZE
    # =====================================================
    def analyze(self):

        try:

            self.status_label.config(
                text="Processando sinais...",
                fg="#ffd740"
            )

            self.root.update_idletasks()

            results = analyze_csvs(

                self.device_path,
                self.powerlab_path,

                self.device_time_entry.get(),
                self.device_flow_entry.get(),
                self.device_volume_entry.get(),

                self.powerlab_time_entry.get(),
                self.powerlab_flow_entry.get(),
                self.powerlab_volume_entry.get()
            )

            (
                tempo_central,

                s_device_calibrado,
                s_powerlab_central,
                
                v_device_calibrado,
                v_powerlab_central,

                media_fluxo,
                diferenca_fluxo,
                bias_fluxo,
                loa_superior_fluxo,
                loa_inferior_fluxo,

                media_volume,
                diferenca_volume,
                bias_volume,
                loa_superior_volume,
                loa_inferior_volume,

                rmse,
                mae,
                corrcoef,

                ganho,
                offset,

                ganho_volume,
                offset_volume,

                delay

            ) = results
            
            # Armazenar dados para update_plot
            self.current_results = {
                "tempo_central": tempo_central,
                "s_device_calibrado": s_device_calibrado,
                "s_powerlab_central": s_powerlab_central,
                "v_device_calibrado": v_device_calibrado,
                "v_powerlab_central": v_powerlab_central,
                "media_fluxo": media_fluxo,
                "diferenca_fluxo": diferenca_fluxo,
                "bias_fluxo": bias_fluxo,
                "loa_superior_fluxo": loa_superior_fluxo,
                "loa_inferior_fluxo": loa_inferior_fluxo,
                "media_volume": media_volume,
                "diferenca_volume": diferenca_volume,
                "bias_volume": bias_volume,
                "loa_superior_volume": loa_superior_volume,
                "loa_inferior_volume": loa_inferior_volume,
                "rmse": rmse,
                "mae": mae,
                "corrcoef": corrcoef,
                "ganho": ganho,
                "offset": offset,
                "ganho_volume": ganho_volume,
                "offset_volume": offset_volume,
                "delay": delay
            }
            
            # Chamar draw_plot
            self.draw_plot()

        except Exception as e:

            messagebox.showerror(
                "Erro",
                str(e)
            )
    
    # =====================================================
    # UPDATE PLOT
    # =====================================================
    def update_plot(self):
        if self.current_results is not None:
            self.draw_plot()

    # =====================================================
    # DRAW PLOT
    # =====================================================
    def draw_plot(self):
        if self.current_results is None:
            return
        
        data = self.current_results
        
        # Extrair dados
        tempo_sync = data["tempo_central"]
        s_device_calibrado = data["s_device_calibrado"]
        s_powerlab_sync = data["s_powerlab_central"]
        v_device_calibrado = data["v_device_calibrado"]
        v_powerlab_sync = data["v_powerlab_central"]
        media_fluxo = data["media_fluxo"]
        diferenca_fluxo = data["diferenca_fluxo"]
        bias_fluxo = data["bias_fluxo"]
        loa_superior_fluxo = data["loa_superior_fluxo"]
        loa_inferior_fluxo = data["loa_inferior_fluxo"]
        media_volume = data["media_volume"]
        diferenca_volume = data["diferenca_volume"]
        bias_volume = data["bias_volume"]
        loa_superior_volume = data["loa_superior_volume"]
        loa_inferior_volume = data["loa_inferior_volume"]
        rmse = data["rmse"]
        mae = data["mae"]
        corrcoef = data["corrcoef"]
        ganho = data["ganho"]
        offset = data["offset"]
        ganho_volume = data["ganho_volume"]
        offset_volume = data["offset_volume"]
        delay = data["delay"]
        
        plot_type = self.plot_type.get()

        if plot_type == "fluxo":
            media = media_fluxo
            diferenca = diferenca_fluxo
            bias = bias_fluxo
            loa_superior = loa_superior_fluxo
            loa_inferior = loa_inferior_fluxo
        else:
            media = media_volume
            diferenca = diferenca_volume
            bias = bias_volume
            loa_superior = loa_superior_volume
            loa_inferior = loa_inferior_volume
        
        # =================================================
        # LIMPAR
        # =================================================
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # =================================================
        # CARDS
        # =================================================
        cards_frame = tk.Frame(
            self.plot_frame,
            bg="#2b2b2b"
        )

        cards_frame.pack(
            fill=tk.X,
            pady=(10, 5)
        )

        metric_cards = [

            ("GANHO FLUXO", f"{ganho:.6f}", "#00bcd4"),
            ("OFFSET FLUXO", f"{offset:.6f}", "#ff9800"),

            ("GANHO VOLUME", f"{ganho_volume:.6f}", "#9c27b0"),
            ("OFFSET VOLUME", f"{offset_volume:.6f}", "#e91e63"),

            ("RMSE", f"{rmse:.4f}", "#ff5252"),
            ("MAE", f"{mae:.4f}", "#ffd740"),

            ("CORRELAÇÃO", f"{corrcoef:.4f}", "#00e676"),

            ("DELAY", f"{delay * 1000:.2f} ms", "#40c4ff")
        ]

        for title, value, color in metric_cards:

            card = tk.Frame(
                cards_frame,
                bg="#252525",
                width=180,
                height=90,
                highlightthickness=1,
                highlightbackground="#3d3d3d"
            )

            card.pack(
                side=tk.LEFT,
                padx=6,
                pady=6,
                fill=tk.BOTH,
                expand=True
            )

            card.pack_propagate(False)

            tk.Label(
                card,
                text=title,
                font=("Segoe UI", 9, "bold"),
                fg="#aaaaaa",
                bg="#252525"
            ).pack(pady=(14, 4))

            tk.Label(
                card,
                text=value,
                font=("Consolas", 15, "bold"),
                fg=color,
                bg="#252525"
            ).pack()

        # =================================================
        # FIGURA
        # =================================================
        fig = Figure(
            figsize=(13, 8),
            dpi=100,
            facecolor="#2b2b2b"
        )

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        for ax in [ax1, ax2]:

            ax.set_facecolor("#252525")

            ax.tick_params(
                colors="white"
            )

            ax.grid(
                alpha=0.18,
                linestyle="--"
            )

            for spine in ax.spines.values():
                spine.set_color("#555555")

        # =================================================
        # PLOT SINAIS / VOLUME
        # =================================================
        if plot_type == "fluxo":
            
            ax1.plot(
                tempo_sync,
                s_device_calibrado,
                linewidth=2.2,
                color="#00bcd4",
                label="ESP32 calibrado"
            )

            ax1.plot(
                tempo_sync,
                s_powerlab_sync,
                linewidth=2.2,
                color="#ff9800",
                label="LabChart"
            )

            ax1.set_title(
                "Sinais sincronizados e calibrados",
                fontsize=14,
                color="white"
            )

            ax1.set_xlabel("Tempo (s)")
            ax1.set_ylabel("Fluxo (L/min)")
            
        else:  # volume
            
            ax1.plot(
                tempo_sync,
                v_device_calibrado,
                linewidth=2.2,
                color="#00bcd4",
                label="ESP32 calibrado"
            )

            ax1.plot(
                tempo_sync,
                v_powerlab_sync,
                linewidth=2.2,
                color="#ff9800",
                label="LabChart"
            )

            ax1.set_title(
                "Volume sincronizado e calibrado",
                fontsize=14,
                color="white"
            )

            ax1.set_xlabel("Tempo (s)")
            ax1.set_ylabel("Volume (L)")

        ax1.legend(
            facecolor="#333333",
            edgecolor="none"
        )

        # =================================================
        # BLAND ALTMAN
        # =================================================
        ax2.scatter(
            media,
            diferenca,
            alpha=0.55,
            s=24,
            color="#00e676"
        )

        ax2.axhline(
            bias,
            linestyle='--',
            linewidth=2,
            color="white",
            label=f"Bias = {bias:.2f}"
        )

# Ajusta o tamanho do eixo Y do Bland-Altman
        if plot_type == "fluxo":
            ax2.set_ylim(-5, 5)
        else:
            ax2.set_ylim(loa_inferior - 10, loa_superior + 10)

        ax2.axhline(
            loa_superior,
            linestyle='--',
            linewidth=2,
            color="#ff5252",
            label=f"+1.96σ = {loa_superior:.2f}"
        )

        ax2.axhline(
            loa_inferior,
            linestyle='--',
            linewidth=2,
            color="#ff5252",
            label=f"-1.96σ = {loa_inferior:.2f}"
        )

        ax2.set_title(
            "Bland-Altman",
            fontsize=14,
            color="white"
        )

        if plot_type == "fluxo":
            ax2.set_xlabel(
                "Média dos sinais (L/min)"
            )
            ax2.set_ylabel(
                "Diferença (L/min)"
            )
        else:
            ax2.set_xlabel(
                "Média do volume (L)"
            )
            ax2.set_ylabel(
                "Diferença (L)"
            )

        ax2.legend(
            facecolor="#333333",
            edgecolor="none"
        )

        fig.tight_layout(
            pad=3
        )

        # =================================================
        # CANVAS
        # =================================================
        canvas = FigureCanvasTkAgg(
            fig,
            master=self.plot_frame
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
            pady=(10, 0)
        )

        self.status_label.config(
            text="Análise concluída com sucesso",
            fg="#00e676"
        )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    root = tk.Tk()

    app = App(root)

    root.mainloop()
