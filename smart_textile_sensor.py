# IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import seaborn as sns
from PIL import Image         

# Fabric image overlay for realistic heatmap visualization
FABRIC_IMAGE_PATH = "fabric_texture.jpg"

# GLOBAL CONFIGURATION
GRID_ROWS       = 20            # sensors along fabric LENGTH  (rows)
GRID_COLS       = 20            # sensors along fabric WIDTH   (columns)
TIME_STEPS      = 10            # number of time snapshots
OUTPUT_DIR      = "textile_heatmaps"   # folder for all saved files

# Transparency of the heatmap overlay on top of fabric
HEATMAP_ALPHA   = 0.55

# Scenario options:  "body_heat"  |  "pressure"  |  "uniform"
SCENARIO        = "body_heat"


# FABRIC IMAGE LOADER
def load_fabric_image(path):
   
    if path is None:
        # if no image provided, plain background is displayed
        print("[Fabric Image] No image path set. Running with plain background.")
        print("  -> To use your own fabric photo, set FABRIC_IMAGE_PATH")
        print("     at the top of this file.\n")
        return None

    if not os.path.isfile(path):
        print(f"[Fabric Image] WARNING: File not found -> '{path}'")
        print("  -> Check that the path is correct and the file exists.")
        print("  -> Continuing with plain background.\n")
        return None

    print(f"[Fabric Image] Loading '{path}' ...")
    img = Image.open(path).convert("RGBA")   # RGBA so we can control alpha

    # Resize to 500x500 px
    display_size = 500
    img_resized  = img.resize((display_size, display_size), Image.LANCZOS)

    # Convert to float array in [0, 1]
    img_array = np.asarray(img_resized).astype(np.float32) / 255.0
    print(f"  -> Loaded OK  (original size: {img.size[0]}x{img.size[1]} px)\n")
    return img_array


# DATA GENERATION
def generate_base_pattern(scenario):   # scenario : str  "body_heat" | "pressure" | "uniform"

    rows, cols = np.mgrid[0:GRID_ROWS, 0:GRID_COLS]

    if scenario == "body_heat":
        # Gaussian warm-core centred at the torso region (like a sensor vest)
        cx, cy = GRID_ROWS * 0.45, GRID_COLS * 0.5
        sigma  = 6.0             # Controls how wide the heat spreads
        base   = np.exp(-((rows - cx)**2 + (cols - cy)**2) / (2 * sigma**2))    # e^(-distance² / spread)
        base   = 30 + 7 * base           # scale to 30 - 37 degrees Celsius

    elif scenario == "pressure":
        # Two localised pressure blobs  (e.g. heel + ball-of-foot)
        spot1  = np.exp(-((rows - 16)**2 + (cols - 10)**2) / (2 * 3.0**2))     # for heel
        spot2  = np.exp(-((rows -  4)**2 + (cols - 10)**2) / (2 * 4.0**2))     # for ball-of-foot
        base   = spot1 + 0.6 * spot2
        base   = 200 * base / base.max()  # scale to 0 - 200 kPa

    else:  # "uniform"
        rng  = np.random.default_rng(seed=0)       # generates same random values every run
        base = rng.uniform(20, 40, size=(GRID_ROWS, GRID_COLS))

    return base


def add_noise(base, noise_scale=0.5, seed=0):
    """Add small Gaussian noise to simulate real sensor imperfections."""
    rng   = np.random.default_rng(seed=seed)
    noise = rng.normal(loc=0, scale=noise_scale, size=base.shape)
    return base + noise


# SIMULATION
def simulate_sensor_data(scenario="body_heat"):

    print(f"[Simulation] Scenario  = '{scenario}'")
    print(f"Grid size = {GRID_ROWS}x{GRID_COLS}")
    print(f"Steps     = {TIME_STEPS}\n")

    base      = generate_base_pattern(scenario)
    grids     = []
    flat_rows = []

    for t in range(TIME_STEPS):
        # Drift the pattern slightly each step to simulate movement
        drift_x = int(t * 0.3)
        drift_y = int(t * 0.2)
        shifted = np.roll(base, shift=(drift_x, drift_y), axis=(0, 1))
        noisy   = add_noise(shifted, noise_scale=0.4, seed=t)

        grids.append(noisy)
        flat_rows.append(noisy.flatten())

        print(f"  Step {t+1:>2}/{TIME_STEPS} | "
              f"mean={noisy.mean():.2f}  "
              f"min={noisy.min():.2f}  "
              f"max={noisy.max():.2f}")

    # Pandas DataFrame  (rows = time steps, cols = sensors)
    col_names = [f"Sensor_r{r:02d}_c{c:02d}"
                 for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    df = pd.DataFrame(flat_rows, columns=col_names)
    df.index.name = "TimeStep"

    if scenario == "body_heat":
        label      = "Temperature (deg C)"
        vmin, vmax = 29, 38
    elif scenario == "pressure":
        label      = "Pressure (kPa)"
        vmin, vmax = 0, 220
    else:
        label      = "Sensor Value"
        vmin, vmax = float(df.values.min()), float(df.values.max())

    return {"grids": grids, "df": df, "label": label, "vmin": vmin, "vmax": vmax}


# VISUALIZATION HELPERS
def _draw_fabric_background(ax, fabric_img):
  
    if fabric_img is not None:
        ax.imshow(fabric_img,
                  extent=[0, GRID_COLS, GRID_ROWS, 0],
                  aspect="auto",
                  zorder=0)           # zorder=0 (drawn first, behind heatmap)
    else:
        ax.set_facecolor("#d9d4cc")


def _sensor_heatmap_rgba(grid, cmap_name, vmin, vmax, alpha):
  
    norm          = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap          = plt.get_cmap(cmap_name)
    rgba          = cmap(norm(grid)).copy()   # shape (20, 20, 4)
    rgba[..., 3]  = alpha                     # set alpha channel uniformly
    return rgba


# MAIN VISUALIZATION FUNCTIONS
def plot_single_heatmap(grid, time_step, label, vmin, vmax, fabric_img, save=True):
    
    fig, ax = plt.subplots(figsize=(7, 6))

    # Layer 0: fabric background
    _draw_fabric_background(ax, fabric_img)

    # Layer 1: semi-transparent sensor heatmap
    overlay = _sensor_heatmap_rgba(grid, "plasma", vmin, vmax, HEATMAP_ALPHA)
    ax.imshow(overlay,
              extent=[0, GRID_COLS, GRID_ROWS, 0],
              aspect="auto",
              zorder=1)           # sits on top of background

    # Color need ScalarMappable to understand which number assigns for what number
    sm = plt.cm.ScalarMappable(
            cmap=plt.get_cmap("plasma"),
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85)
    cbar.set_label(label, fontsize=10)

    # Layer 2: small white dots showing sensor positions in the fabric
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            ax.plot(c + 0.5, r + 0.5,
                    marker="o", markersize=2.2,
                    color="white", alpha=0.45, zorder=2)

    # Labels
    ax.set_title(f"Smart Textile Sensor Map  -  Time Step {time_step + 1}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Fabric Width Sensors",  fontsize=10)
    ax.set_ylabel("Fabric Length Sensors", fontsize=10)

    # Show every 5th sensor index on the ticks
    tick_pos = np.arange(0, GRID_COLS, 5)
    ax.set_xticks(tick_pos + 0.5);  ax.set_xticklabels(tick_pos)
    ax.set_yticks(tick_pos + 0.5);  ax.set_yticklabels(tick_pos)

    ax.set_xlim(0, GRID_COLS)
    ax.set_ylim(GRID_ROWS, 0)    # y=0 at top (matches image orientation)

    plt.tight_layout()

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = os.path.join(OUTPUT_DIR, f"heatmap_step_{time_step+1:02d}.png")
        fig.savefig(filename, dpi=120)
        print(f"  Saved -> {filename}")

    plt.close(fig)


def plot_all_steps_grid(grids, label, vmin, vmax, fabric_img):
    
    n        = len(grids)
    cols_sub = 5
    rows_sub = int(np.ceil(n / cols_sub))

    fig, axes = plt.subplots(rows_sub, cols_sub,
                              figsize=(cols_sub * 3.2, rows_sub * 3))
    axes_flat = axes.flatten()

    for i, (grid, ax) in enumerate(zip(grids, axes_flat)):
        _draw_fabric_background(ax, fabric_img)

        overlay = _sensor_heatmap_rgba(grid, "plasma", vmin, vmax, HEATMAP_ALPHA)
        ax.imshow(overlay,
                  extent=[0, GRID_COLS, GRID_ROWS, 0],
                  aspect="auto", zorder=1)

        ax.set_title(f"t = {i+1}", fontsize=9)
        ax.set_xlabel("Width",  fontsize=7)
        ax.set_ylabel("Length", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, GRID_COLS); ax.set_ylim(GRID_ROWS, 0)

    # One shared colorbar on the right side
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("plasma"),
                                norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes_flat[:n], shrink=0.6, label=label)

    # Hide unused panels
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Smart Textile Sensor Map - All Time Steps",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "overview_all_steps.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n  Overview saved -> {path}")
    plt.close(fig)


def create_animation(grids, label, vmin, vmax, fabric_img):
    
    fig, ax = plt.subplots(figsize=(6, 5.5))

    # Static background drawn once
    _draw_fabric_background(ax, fabric_img)

    # Initial heatmap overlay
    first_overlay = _sensor_heatmap_rgba(grids[0], "plasma", vmin, vmax, HEATMAP_ALPHA)
    im = ax.imshow(first_overlay,
                   extent=[0, GRID_COLS, GRID_ROWS, 0],
                   aspect="auto", zorder=1)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("plasma"),
                                norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.85, label=label)

    ax.set_xlabel("Fabric Width Sensors",  fontsize=10)
    ax.set_ylabel("Fabric Length Sensors", fontsize=10)
    ax.set_xlim(0, GRID_COLS); ax.set_ylim(GRID_ROWS, 0)
    title_obj = ax.set_title("Smart Textile Sensor Map  -  Time Step 1",
                              fontsize=12, fontweight="bold")

    def update_frame(frame_idx):
        """Update only the heatmap overlay each frame; background stays fixed."""
        new_overlay = _sensor_heatmap_rgba(
            grids[frame_idx], "plasma", vmin, vmax, HEATMAP_ALPHA)
        im.set_data(new_overlay)
        title_obj.set_text(
            f"Smart Textile Sensor Map  -  Time Step {frame_idx + 1}")
        return im, title_obj

    ani = animation.FuncAnimation(fig, update_frame,
                                   frames=len(grids),
                                   interval=600, blit=False, repeat=True)

    gif_path = os.path.join(OUTPUT_DIR, "sensor_animation.gif")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ani.save(gif_path, writer="pillow", fps=2)
    print(f"  Animation saved -> {gif_path}")
    plt.close(fig)


# PANDAS REPORT
def print_dataframe_summary(df):
    """Print statistics and save the sensor data as a CSV file."""
    print("\n" + "=" * 60)
    print("  PANDAS DATAFRAME SUMMARY")
    print("=" * 60)
    print(f"  Shape  : {df.shape}  (rows=time steps, cols=sensors)")
    print(f"  Columns: {df.columns[0]} ... {df.columns[-1]}\n")

    stats = df.agg(["mean", "std", "min", "max"], axis=1)
    stats.columns = ["Mean", "Std Dev", "Min", "Max"]
    print(stats.to_string())

    csv_path = os.path.join(OUTPUT_DIR, "sensor_data.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(csv_path)
    print(f"\n  Full data saved -> {csv_path}")


# MAIN
def main():
    print("=" * 60)
    print("  SMART TEXTILE SENSOR DATA VISUALISATION")
    print("=" * 60 + "\n")

    # Load fabric image (returns None if FABRIC_IMAGE_PATH is not set)
    fabric_img = load_fabric_image(FABRIC_IMAGE_PATH)

    # Run simulation
    result = simulate_sensor_data(scenario=SCENARIO)
    grids  = result["grids"]
    df     = result["df"]
    label  = result["label"]
    vmin   = result["vmin"]
    vmax   = result["vmax"]

    # DataFrame summary and CSV export
    print_dataframe_summary(df)

    # Individual heatmap PNGs (one per time step)
    print("\n[Saving individual heatmaps ...]")
    for t, grid in enumerate(grids):
        plot_single_heatmap(grid, t, label, vmin, vmax, fabric_img, save=True)

    # Overview multi-panel figure
    print("\n[Saving overview figure ...]")
    plot_all_steps_grid(grids, label, vmin, vmax, fabric_img)

    # Animated GIF
    print("\n[Creating animation ...]")
    create_animation(grids, label, vmin, vmax, fabric_img)

    print("\n" + "=" * 60)
    print(f"  All outputs saved to -> '{OUTPUT_DIR}/'")
    print("=" * 60)


if __name__ == "__main__":
    main()