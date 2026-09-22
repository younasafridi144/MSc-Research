from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Callable
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from gsi_predictor import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MEAN,
    DEFAULT_MODEL_NAME,
    DEFAULT_STD,
    GSIPredictor,
    ModelLoadInfo,
    Prediction,
    TrainingMetadata,
    load_training_metadata,
    parse_float_tuple,
)


APP_TITLE = "GSI Prediction - Swin Transformer"
MODEL_OPTIONS = (
    "swin_base_patch4_window7_224",
    "swin_base_patch4_window12_384",
)

BG = "#eef3f8"
HEADER = "#101827"
SURFACE = "#ffffff"
SURFACE_ALT = "#f7fafc"
BORDER = "#d8e0ea"
TEXT = "#111827"
MUTED = "#64748b"
TEAL = "#0f766e"
TEAL_DARK = "#115e59"
AMBER = "#f59e0b"
EARTH = "#8a6f45"
SAGE = "#87a878"


def _discover_default_model_paths() -> tuple[str, str]:
    app_dir = Path(__file__).resolve().parent
    search_roots = [app_dir, app_dir / "models", app_dir / "model_files"]

    checkpoint_candidates: list[Path] = []
    metadata_candidates: list[Path] = []

    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in {".pth", ".pt", ".ckpt", ".safetensors"}:
                checkpoint_candidates.append(path)
            elif suffix == ".json" and "metadata" in path.stem.lower():
                metadata_candidates.append(path)

    checkpoint_path = str(checkpoint_candidates[0].resolve()) if checkpoint_candidates else ""
    metadata_path = str(metadata_candidates[0].resolve()) if metadata_candidates else ""
    return checkpoint_path, metadata_path


class GSIGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1400x850")
        self.minsize(1200, 750)

        self.predictor: GSIPredictor | None = None
        self.preview_source: Image.Image | None = None
        self.preview_image: ImageTk.PhotoImage | None = None
        self.preview_overlay: Image.Image | None = None
        self.show_explanation = tk.BooleanVar(value=True)

        # Use fixed checkpoint and metadata paths supplied by the user
        self.checkpoint_path = tk.StringVar(
            value=r"C:\Users\Shahab\Desktop\MSc Thesis\thesis_Results\gsi_SwinBase_top5_outputs (1)\SwinBase_top_epoch_015_r2_0p93691_state_dict.pth"
        )
        self.metadata_path = tk.StringVar(
            value=r"C:\Users\Shahab\Desktop\MSc Thesis\thesis_Results\gsi_SwinBase_top5_outputs (1)\SwinBase_top_epoch_015_r2_0p93691_metadata.json"
        )

        # Default directory for image selection
        self.default_image_dir = r"C:\Users\Shahab\Desktop\MSc Thesis\MS Mining Engineering"
        self.image_path = tk.StringVar()
        self.model_name = tk.StringVar(value=DEFAULT_MODEL_NAME)
        self.task = tk.StringVar(value="regression")
        self.output_size = tk.IntVar(value=1)
        self.image_size = tk.IntVar(value=DEFAULT_IMAGE_SIZE)
        self.device = tk.StringVar(value="auto")
        self.mean = tk.StringVar(value=", ".join(str(value) for value in DEFAULT_MEAN))
        self.std = tk.StringVar(value=", ".join(str(value) for value in DEFAULT_STD))
        self.target_name = tk.StringVar(value="GSI")
        self.target_mean = tk.StringVar()
        self.target_std = tk.StringVar()
        self.class_names = tk.StringVar()
        self.status = tk.StringVar(value="Load your trained model to begin.")
        self.model_state = tk.StringVar(value="Model not loaded")
        self.selected_image_name = tk.StringVar(value="No image selected")
        self.result_text = tk.StringVar(value="No prediction yet")

        self._build_style()
        self._build_layout()
        self._apply_metadata_from_path(show_error=False)
        self._sync_task_fields()
        # Auto-load the configured model and metadata
        try:
            self.load_model()
        except Exception:
            # load_model handles its own errors via _operation_failed; ignore here
            pass

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        self.configure(bg="#dfe9e5")

        style.configure(".", font=("Segoe UI", 10))
        style.configure("App.TFrame", background="#f5f8f7")
        style.configure("Header.TFrame", background=HEADER)
        style.configure("Card.TFrame", background=SURFACE, bordercolor=BORDER, relief="solid", borderwidth=1)
        style.configure("Card.TLabelframe", background=SURFACE, bordercolor=BORDER, relief="solid", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background=SURFACE, foreground=TEXT, font=("Segoe UI Semibold", 10))
        style.configure("Card.TLabel", background=SURFACE, foreground=TEXT)
        style.configure("Muted.TLabel", background=SURFACE, foreground=MUTED)
        style.configure("HeaderTitle.TLabel", background=HEADER, foreground="#ffffff", font=("Segoe UI Semibold", 20))
        style.configure("HeaderSub.TLabel", background=HEADER, foreground="#cbd5e1", font=("Segoe UI", 10))
        style.configure("HeaderStatus.TLabel", background=HEADER, foreground="#dbeafe", font=("Segoe UI Semibold", 10))
        style.configure("Result.TLabel", background=SURFACE, foreground=TEAL_DARK, font=("Segoe UI Semibold", 24))
        style.configure("Status.TLabel", background="#f5f8f7", foreground="#334155", font=("Segoe UI", 9))
        style.configure("Primary.TButton", font=("Segoe UI Semibold", 11), foreground="#ffffff", background=TEAL)
        style.map("Primary.TButton", background=[("active", TEAL_DARK), ("disabled", "#94a3b8")])
        style.configure("Secondary.TButton", font=("Segoe UI Semibold", 10), foreground=TEXT, background="#e8edf3")
        style.map("Secondary.TButton", background=[("active", "#dbe4ee"), ("disabled", "#edf2f7")])
        style.configure("Browse.TButton", font=("Segoe UI Semibold", 9), padding=(10, 4))
        style.configure("Main.TNotebook", background=SURFACE, borderwidth=0)
        style.configure("Main.TNotebook.Tab", padding=(18, 8), font=("Segoe UI Semibold", 9))
        style.map("Main.TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", TEAL_DARK)])

    def _build_layout(self) -> None:
        self.background_canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.background_canvas.pack(fill="both", expand=True)
        self.background_canvas.bind("<Configure>", self._draw_background)

        shell = ttk.Frame(self.background_canvas, padding=18, style="App.TFrame")
        self.shell_window = self.background_canvas.create_window(22, 22, anchor="nw", window=shell)
        self.background_canvas.bind("<Configure>", self._resize_shell, add="+")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        self._build_header(shell)

        content = ttk.Frame(shell, style="App.TFrame")
        content.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        content.columnconfigure(0, weight=0, minsize=390)
        content.columnconfigure(1, weight=1, minsize=700)
        content.rowconfigure(0, weight=1)

        left = ttk.Frame(content, style="App.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        right = ttk.Frame(content, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._build_result_card(left)
        self._build_settings_tabs(left)
        self._build_preview(right)

        footer = ttk.Frame(shell, style="App.TFrame")
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status, style="Status.TLabel").grid(row=0, column=0, sticky="w")

    def _resize_shell(self, event: tk.Event) -> None:
        self.background_canvas.itemconfigure(
            self.shell_window,
            width=max(event.width - 44, 200),
            height=max(event.height - 44, 200),
        )

    def _draw_background(self, event: tk.Event) -> None:
        canvas = self.background_canvas
        canvas.delete("background")
        width = max(event.width, 2)
        height = max(event.height, 2)

        top = self._hex_to_rgb("#dae9e2")
        middle = self._hex_to_rgb("#f2ead9")
        bottom = self._hex_to_rgb("#d7e3ef")
        for y in range(height):
            ratio = y / max(height - 1, 1)
            if ratio < 0.55:
                local = ratio / 0.55
                color = self._blend_rgb(top, middle, local)
            else:
                local = (ratio - 0.55) / 0.45
                color = self._blend_rgb(middle, bottom, local)
            canvas.create_line(0, y, width, y, fill=self._rgb_to_hex(color), tags="background")

        ridge_y = int(height * 0.74)
        for offset, color, line_width in ((0, "#8a6f45", 2), (24, "#a58859", 1), (48, "#5f7f67", 1)):
            points: list[int] = []
            for x in range(-80, width + 120, 80):
                y = ridge_y + offset + int(18 * ((x // 80) % 2)) - int(10 * (x / max(width, 1)))
                points.extend((x, y))
            canvas.create_line(*points, smooth=True, fill=color, width=line_width, tags="background")

        for index, x in enumerate(range(-120, width + 160, 140)):
            shade = "#c3d0c7" if index % 2 == 0 else "#d2c6a6"
            canvas.create_line(
                x,
                0,
                x + int(width * 0.32),
                height,
                fill=shade,
                width=1,
                tags="background",
            )

        canvas.create_rectangle(0, 0, width, height, outline="", fill="", tags="background")
        canvas.tag_lower("background")

    def _build_header(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, padding=(20, 16), style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="GSI Predictor", style="HeaderTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="GSI inference for field images",
            style="HeaderSub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        actions = ttk.Frame(header, style="Header.TFrame")
        actions.grid(row=0, column=1, rowspan=2, sticky="e")
        self.predict_button = ttk.Button(
            actions,
            text="Predict GSI",
            command=self.predict,
            style="Primary.TButton",
            width=16,
        )
        self.predict_button.grid(row=0, column=0, ipady=4)
        ttk.Label(actions, textvariable=self.model_state, style="HeaderStatus.TLabel").grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="e",
            pady=(8, 0),
        )

    def _build_result_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, padding=16, style="Card.TFrame")
        card.grid(row=0, column=0, sticky="ew")
        card.columnconfigure(0, weight=1)

        ttk.Label(card, text="Prediction", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(card, textvariable=self.result_text, wraplength=330, style="Result.TLabel").grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(4, 10),
        )
        self.detail_box = tk.Text(
            card,
            height=4,
            wrap="word",
            bg=SURFACE_ALT,
            fg=TEXT,
            relief="flat",
            padx=10,
            pady=8,
            font=("Segoe UI", 9),
        )
        self.detail_box.grid(row=2, column=0, sticky="ew")
        self.detail_box.insert(
            "1.0",
            "Model diagnostics and prediction notes will appear here.",
        )
        self.detail_box.configure(state="disabled")

    def _build_settings_tabs(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent, style="Main.TNotebook")
        notebook.grid(row=1, column=0, sticky="nsew", pady=(14, 0))

        files_tab = ttk.Frame(notebook, padding=14, style="Card.TFrame")
        model_tab = ttk.Frame(notebook, padding=14, style="Card.TFrame")
        scaling_tab = ttk.Frame(notebook, padding=14, style="Card.TFrame")
        notebook.add(files_tab, text="Files")
        notebook.add(model_tab, text="Model")
        notebook.add(scaling_tab, text="Scaling")

        self._build_files_tab(files_tab)
        self._build_model_tab(model_tab)
        self._build_scaling_tab(scaling_tab)

    def _build_files_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        # Model and metadata are fixed; only allow choosing the image
        self._path_picker(parent, "Image", self.image_path, self._choose_image, 0)
        ttk.Label(parent, textvariable=self.selected_image_name, style="Muted.TLabel").grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(8, 0),
        )
        ttk.Checkbutton(
            parent,
            text="Show Transformer Relevancy",
            variable=self.show_explanation,
            command=self._render_preview,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_model_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="Architecture", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=7)
        ttk.Combobox(parent, textvariable=self.model_name, values=MODEL_OPTIONS).grid(
            row=0,
            column=1,
            sticky="ew",
            pady=7,
        )

        ttk.Label(parent, text="Task", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=7)
        task_box = ttk.Combobox(
            parent,
            textvariable=self.task,
            state="readonly",
            values=("regression", "classification"),
        )
        task_box.grid(row=1, column=1, sticky="ew", pady=7)
        task_box.bind("<<ComboboxSelected>>", lambda _event: self._sync_task_fields())

        ttk.Label(parent, text="Outputs", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=7)
        ttk.Spinbox(parent, from_=1, to=1000, textvariable=self.output_size, width=10).grid(
            row=2,
            column=1,
            sticky="w",
        )

        # Explainability options removed

        ttk.Label(parent, text="Image size", style="Card.TLabel").grid(row=4, column=0, sticky="w", pady=7)
        ttk.Spinbox(parent, from_=64, to=1024, increment=32, textvariable=self.image_size, width=10).grid(
            row=4,
            column=1,
            sticky="w",
            pady=7,
        )

        ttk.Label(parent, text="Device", style="Card.TLabel").grid(row=5, column=0, sticky="w", pady=7)
        ttk.Combobox(parent, textvariable=self.device, state="readonly", values=("auto", "cpu", "cuda")).grid(
            row=5,
            column=1,
            sticky="ew",
            pady=7,
        )

    def _build_scaling_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="Mean", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=self.mean).grid(row=0, column=1, sticky="ew", pady=6)

        ttk.Label(parent, text="Std", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=self.std).grid(row=1, column=1, sticky="ew", pady=6)

        ttk.Label(parent, text="Target", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=self.target_name).grid(row=2, column=1, sticky="ew", pady=6)

        ttk.Label(parent, text="Target mean", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=self.target_mean).grid(row=3, column=1, sticky="ew", pady=6)

        ttk.Label(parent, text="Target std", style="Card.TLabel").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=self.target_std).grid(row=4, column=1, sticky="ew", pady=6)

        ttk.Label(parent, text="Classes", style="Card.TLabel").grid(row=5, column=0, sticky="w", pady=6)
        self.class_entry = ttk.Entry(parent, textvariable=self.class_names)
        self.class_entry.grid(row=5, column=1, sticky="ew", pady=6)

    def _build_preview(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, padding=10, style="Card.TFrame")
        card.grid(row=0, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        top = ttk.Frame(card, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Image Preview", style="Card.TLabel", font=("Segoe UI Semibold", 11)).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(top, textvariable=self.selected_image_name, style="Muted.TLabel").grid(
            row=0,
            column=1,
            sticky="e",
            padx=(0, 10),
        )
        self.download_button = ttk.Button(
            top,
            text="Download",
            command=self._download_preview,
            style="Browse.TButton",
        )
        self.download_button.grid(row=0, column=2, sticky="e")

        self.preview_canvas = tk.Canvas(card, bg="#e8eef5", highlightthickness=0)
        self.preview_canvas.grid(row=1, column=0, sticky="nsew")
        self.preview_canvas.bind("<Configure>", lambda _event: self._render_preview())
        self._render_preview()

    def _path_picker(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        command: Callable[[], None],
        row: int,
    ) -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row + 1, column=0, sticky="ew", pady=(3, 12))
        ttk.Button(parent, text="Browse", command=command, style="Browse.TButton").grid(
            row=row + 1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(3, 12),
        )

    def _sync_task_fields(self) -> None:
        is_classification = self.task.get() == "classification"
        self.class_entry.configure(state="normal" if is_classification else "disabled")
        if not is_classification and self.output_size.get() != 1:
            self.output_size.set(1)

    def _choose_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Select trained model",
            filetypes=(
                ("PyTorch checkpoints", "*.pth *.pt *.ckpt *.bin *.safetensors"),
                ("All files", "*.*"),
            ),
        )
        if path:
            self.checkpoint_path.set(path)

    def _choose_metadata(self) -> None:
        path = filedialog.askopenfilename(
            title="Select training metadata",
            filetypes=(
                ("JSON metadata", "*.json"),
                ("All files", "*.*"),
            ),
        )
        if path:
            self.metadata_path.set(path)
            self._apply_metadata_from_path(show_error=True)

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            initialdir=getattr(self, "default_image_dir", None),
            filetypes=(
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ),
        )
        if path:
            self.image_path.set(path)
            self.selected_image_name.set(Path(path).name)
            self.preview_overlay = None
            self._show_preview(path)

    def _show_preview(self, path: str) -> None:
        try:
            self.preview_source = Image.open(path).convert("RGB")
            self._render_preview()
        except Exception as exc:
            self._show_error("Image preview failed", exc)

    def _render_preview(self) -> None:
        if not hasattr(self, "preview_canvas"):
            return
        canvas = self.preview_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 2)
        height = max(canvas.winfo_height(), 2)

        display_image = self.preview_overlay if self.show_explanation.get() and self.preview_overlay is not None else self.preview_source
        if display_image is None:
            canvas.create_rectangle(50, 50, width - 50, height - 50, outline=BORDER, width=1)
            canvas.create_text(
                width // 2,
                height // 2,
                text="Choose an image from the Files tab",
                fill=MUTED,
                font=("Segoe UI Semibold", 13),
            )
            return

        image = display_image.copy()
        image.thumbnail((max(width - 4, 1), max(height - 4, 1)), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(image)
        canvas.create_image(width // 2, height // 2, image=self.preview_image, anchor="center")
        x0 = (width - image.width) // 2
        y0 = (height - image.height) // 2
        canvas.create_rectangle(x0, y0, x0 + image.width, y0 + image.height, outline="#cbd5e1", width=1)

    def _download_preview(self) -> None:
        display_image = self.preview_overlay if self.show_explanation.get() and self.preview_overlay is not None else self.preview_source
        if display_image is None:
            messagebox.showwarning(APP_TITLE, "No image to download.")
            return
        output_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=(
                ("PNG images", "*.png"),
                ("JPEG images", "*.jpg"),
                ("All files", "*.*"),
            ),
        )
        if output_path:
            try:
                display_image.save(output_path, dpi=(600, 600))
                messagebox.showinfo(APP_TITLE, f"Image saved to:\n{output_path} (600 DPI)")
            except Exception as exc:
                messagebox.showerror(APP_TITLE, f"Failed to save image:\n{exc}")


    def load_model(self) -> None:
        checkpoint = self.checkpoint_path.get().strip()
        if not checkpoint:
            messagebox.showwarning(APP_TITLE, "Choose a trained model first.")
            return
        self._apply_metadata_from_path(show_error=False)
        model_name = self.model_name.get().strip() or DEFAULT_MODEL_NAME
        try:
            output_size = int(self.output_size.get())
            image_size = int(self.image_size.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning(APP_TITLE, "Output size and image size must be whole numbers.")
            return
        if output_size < 1 or image_size < 1:
            messagebox.showwarning(APP_TITLE, "Output size and image size must be positive.")
            return
        device = self.device.get()
        mean_text = self.mean.get()
        std_text = self.std.get()
        try:
            y_mean = self._optional_float(self.target_mean.get())
            y_std = self._optional_float(self.target_std.get())
        except ValueError:
            messagebox.showwarning(APP_TITLE, "Target mean and target std must be numbers when provided.")
            return
        if (y_mean is None) != (y_std is None):
            messagebox.showwarning(APP_TITLE, "Provide both target mean and target std, or leave both empty.")
            return

        self._set_busy(True, "Loading model...")
        self.model_state.set("Loading model...")

        def worker() -> None:
            try:
                mean = parse_float_tuple(mean_text, DEFAULT_MEAN)
                std = parse_float_tuple(std_text, DEFAULT_STD)
                predictor = GSIPredictor(
                    model_name=model_name,
                    output_size=output_size,
                    image_size=image_size,
                    mean=mean,
                    std=std,
                    y_mean=y_mean,
                    y_std=y_std,
                    device=device,
                )
                info = predictor.load_checkpoint(checkpoint)
                self.after(0, lambda: self._model_loaded(predictor, info))
            except Exception as exc:
                trace = traceback.format_exc()
                self.after(0, lambda: self._operation_failed("Could not load model", exc, trace))

        threading.Thread(target=worker, daemon=True).start()

    def _model_loaded(self, predictor: GSIPredictor, info: ModelLoadInfo) -> None:
        self.predictor = predictor
        self.model_state.set(f"Loaded on {info.device}")
        self._set_busy(False, f"Model loaded on {info.device}.")
        details = [
            f"Device: {info.device}",
            f"Missing keys: {len(info.missing_keys)}",
            f"Unexpected keys: {len(info.unexpected_keys)}",
            f"Skipped shape mismatches: {len(info.skipped_keys)}",
        ]
        if self.target_mean.get().strip() and self.target_std.get().strip():
            details.extend(
                [
                    "",
                    f"Target scaling: {self.target_name.get()} = raw output * {self.target_std.get()} + {self.target_mean.get()}",
                ]
            )
        if info.skipped_keys:
            details.append("")
            details.append("Skipped keys:")
            details.extend(f"- {key}" for key in info.skipped_keys[:20])
        self._set_details("\n".join(details))

    def predict(self) -> None:
        if self.predictor is None:
            messagebox.showwarning(APP_TITLE, "Load the model before predicting.")
            return
        image = self.image_path.get().strip()
        if not image:
            messagebox.showwarning(APP_TITLE, "Choose an image first.")
            return
        if not Path(image).exists():
            messagebox.showwarning(APP_TITLE, "The selected image path does not exist.")
            return
        task = self.task.get()
        class_names = [name.strip() for name in self.class_names.get().split(",") if name.strip()]
        use_explanation = self.show_explanation.get()
        self._set_busy(True, "Running prediction..." if not use_explanation else "Running Transformer Relevancy...")

        def worker() -> None:
            try:
                if use_explanation:
                    prediction, overlay = self.predictor.predict_image_with_heatmap(
                        image,
                        task=task,
                        class_names=class_names,
                    )
                    self.after(0, lambda: self._prediction_ready(prediction, overlay))
                else:
                    prediction = self.predictor.predict_image(image, task=task, class_names=class_names)
                    self.after(0, lambda: self._prediction_ready(prediction, None))
            except Exception as exc:
                trace = traceback.format_exc()
                self.after(0, lambda: self._operation_failed("Prediction failed", exc, trace))

        threading.Thread(target=worker, daemon=True).start()

    def _prediction_ready(self, prediction: Prediction, overlay: Image.Image | None = None) -> None:
        self.preview_overlay = overlay
        self._render_preview()
        self._set_busy(False, "Prediction complete.")
        if prediction.value is not None:
            target = self.target_name.get().strip() or "GSI"
            self.result_text.set(f"{target}: {prediction.value:.2f}")
            detail_lines = [f"{target} prediction complete."]
            # No explainability overlays in this GUI configuration
            self._set_details("\n".join(detail_lines))
            return

        self.result_text.set(f"Class: {prediction.label}")
        lines = ["Probabilities:"]
        lines.extend(f"{label}: {probability:.4%}" for label, probability in prediction.probabilities[:10])
        lines.append("")
        lines.append("Raw logits:")
        lines.append(", ".join(f"{value:.6f}" for value in prediction.raw_outputs))
        self._set_details("\n".join(lines))

    def _set_busy(self, busy: bool, status: str) -> None:
        self.status.set(status)
        state = "disabled" if busy else "normal"
        # Load button removed; only update predict button state
        self.predict_button.configure(state=state)

    def _operation_failed(self, title: str, exc: Exception, trace: str) -> None:
        self.model_state.set("Model not loaded" if self.predictor is None else self.model_state.get())
        self._set_busy(False, title)
        self._set_details(trace)
        messagebox.showerror(APP_TITLE, f"{title}:\n{exc}")

    def _show_error(self, title: str, exc: Exception) -> None:
        self.status.set(title)
        messagebox.showerror(APP_TITLE, f"{title}:\n{exc}")

    def _apply_metadata_from_path(self, show_error: bool) -> None:
        path = self.metadata_path.get().strip()
        if not path or not Path(path).exists():
            return
        try:
            metadata = load_training_metadata(path)
        except Exception as exc:
            if show_error:
                self._show_error("Metadata load failed", exc)
            return
        self._apply_metadata(metadata)

    def _apply_metadata(self, metadata: TrainingMetadata) -> None:
        if metadata.model_name:
            self.model_name.set(metadata.model_name)
        if metadata.image_size:
            self.image_size.set(metadata.image_size)
        self.task.set("regression")
        self.output_size.set(1)
        self.target_name.set(metadata.target_name)
        if metadata.y_mean is not None:
            self.target_mean.set(str(metadata.y_mean))
        if metadata.y_std is not None:
            self.target_std.set(str(metadata.y_std))
        self.status.set(f"Metadata loaded for {metadata.target_name}.")

    def _optional_float(self, value: str) -> float | None:
        value = value.strip()
        if not value:
            return None
        return float(value)

    def _set_details(self, text: str) -> None:
        self.detail_box.configure(state="normal")
        self.detail_box.delete("1.0", "end")
        self.detail_box.insert("1.0", text)
        self.detail_box.configure(state="disabled")

    def _hex_to_rgb(self, value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)

    def _rgb_to_hex(self, value: tuple[int, int, int]) -> str:
        return f"#{value[0]:02x}{value[1]:02x}{value[2]:02x}"

    def _blend_rgb(
        self,
        start: tuple[int, int, int],
        end: tuple[int, int, int],
        ratio: float,
    ) -> tuple[int, int, int]:
        ratio = max(0.0, min(1.0, ratio))
        return tuple(int(start[index] + (end[index] - start[index]) * ratio) for index in range(3))


if __name__ == "__main__":
    app = GSIGui()
    app.mainloop()
