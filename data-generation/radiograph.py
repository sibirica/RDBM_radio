import numpy as np
import os
import sys
import xml.etree.ElementTree as ET
from scipy.ndimage import gaussian_filter, map_coordinates
from gvxrPython3 import gvxr
import trimesh
from scipy.ndimage import binary_fill_holes, label, distance_transform_edt
import numpy as np

ACTIVE_GVXR_WINDOWS = set()

def _create_gvxr_window(window_id: int) -> None:
    """Create a GVXR renderer context.

    Without DISPLAY, use EGL (headless). With DISPLAY, use OPENGL.
    Override with GVXR_RENDERER=EGL|OPENGL.
    """
    forced = os.environ.get("GVXR_RENDERER", "").strip().upper()
    if forced in ("EGL", "OPENGL"):
        renderer = forced
    elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        renderer = "OPENGL"
    else:
        renderer = "EGL"
    gvxr.createWindow(window_id, 0, renderer)
    ACTIVE_GVXR_WINDOWS.add(window_id)

def _destroy_gvxr_window_if_needed(window_id: int):
    if window_id not in ACTIVE_GVXR_WINDOWS:
        return
    try:
        gvxr.destroyWindow(window_id)
    except Exception:
        pass
    ACTIVE_GVXR_WINDOWS.discard(window_id)

# ==============================================================
# DATA CLASSES
# ==============================================================

class SourceCord:
    """X-ray source position."""
    def __init__(self, x=None, y=None, z=None, units="cm"):
        self.x = x; self.y = y; self.z = z; self.units = units
    def is_set(self): return None not in (self.x, self.y, self.z)
    def set(self, x, y, z, units=None):
        self.x = x; self.y = y; self.z = z
        if units: self.units = units
    def get(self): return (self.x, self.y, self.z)


class SourceVals:
    """X-ray source spectrum and beam parameters."""
    def __init__(self):
        self.spectrum_type   = "mono"
        self.mono_energy     = None
        self.energy_units    = "keV"
        self.photons_per_ray = None
        self.poly_energies   = None
        self.poly_weights    = None
        self.rho             = None
    def is_set(self):
        if self.spectrum_type == "mono":
            return None not in (self.mono_energy, self.photons_per_ray, self.rho)
        return None not in (self.poly_energies, self.poly_weights,
                            self.photons_per_ray, self.rho)


class Detector:
    """Detector geometry and response parameters."""
    def __init__(self):
        self.nx             = 4096        # 2048
        self.ny             = 4096        # 2048
        self.width_cm       = 20.0    # physical width (cm) — MUST be set
        self.height_cm      = 20.0    # physical height (cm)
        self.units          = "cm"
        self.bit_depth      = 16
        self.full_well      = 65535
        self.gain           = 1.0
        self.readout_noise  = 5.0
        self.dark_current   = 0.1
        self.exposure_time  = 1.0
        self.blur_sigma_px  = 1.0
        self.fpn_sigma      = 0.01
        self.fpn_seed       = 42
        self.distortion_k1  = 0.0
    def is_set(self):
        return None not in (self.nx, self.ny, self.width_cm)
    def pixel_size_x(self): return self.width_cm  / float(self.nx)
    def pixel_size_y(self): return self.height_cm / float(self.ny)


class Location:
    """Detector position."""
    def __init__(self, x=None, y=None, z=None, units="cm"):
        self.x = x; self.y = y; self.z = z; self.units = units
    def is_set(self): return None not in (self.x, self.y, self.z)
    def set(self, x, y, z, units=None):
        self.x = x; self.y = y; self.z = z
        if units: self.units = units
    def get(self): return (self.x, self.y, self.z)


class FocalSpot:
    """Focal spot parameters."""
    def __init__(self):
        self.enabled     = False
        self.n_samples   = 1
        self.sigma_x_cm  = 0.0
        self.sigma_y_cm  = 0.0
        self.sigma_z_cm  = 0.0
        self.seed        = None


class NoiseConfig:
    """Noise model parameters."""
    def __init__(self):
        self.poisson_enabled    = True
        self.photon_flux        = 20000.0
        self.readout_sigma_adu  = 5.0
        self.readout_seed       = None
        self.dark_current_adu   = 0.1
        self.exposure_time_s    = 1.0
        self.fpn_enabled        = True
        self.fpn_sigma          = 0.01
        self.fpn_seed           = 42
        self.scatter_enabled    = True
        self.scatter_fraction   = 0.05
        self.scatter_blur_sigma = 20.0
        self.scatter_seed       = None
        self.poisson_seed       = None

class ScintillatorConfig:
    def __init__(self):
        self.enabled                = False
        self.material               = "CsI"
        self.thickness_cm           = 0.05
        self.mu_cm_inv              = 8.0
        self.light_yield            = 1.0
        self.optical_blur_sigma_px  = 0.0
        self.saturation_enabled     = False
        self.saturation_level       = 65535.0

class SimConfig:
    """Top-level simulation configuration."""
    def __init__(self):
        self.beam_type         = "cone"
        self.object_location   = None
        self.rotation          = None
        self.material_compound = "SS316"
        self.material_density  = 7.93
        self.stl_root          = "."
        self.stl_files         = []
        self.stl_units         = "mm"   # units of the STL geometry
        self.output_dir        = "radiographs"
        self.save_png          = True
        self.save_npy          = True
        self.save_flatfield    = True
        self.save_darkfield    = True
        self.n_projections     = 1
        self.rotation_step_deg = 0.0
        self.mixture           = []
        self.mixture_density   = 0.0
        self.object_jitter_enabled = False
        self.object_jitter_dist = "normal"
        self.object_jitter_x_cm = 0.0
        self.object_jitter_y_cm = 0.0
        self.object_jitter_z_cm = 0.0
        self.object_jitter_seed = None
        self.rotation_jitter_enabled    = False
        self.rotation_jitter_dist  = "normal"
        self.rotation_jitter_pitch_deg = 0.0
        self.rotation_jitter_roll_deg  = 0.0
        self.rotation_jitter_yaw_deg = 0.0
        self.rotation_jitter_seed = None
        self.gt_zoom_enabled    = True
        self.gt_zoom_pixels     = 2048
        self.gt_zoom_margin_cm  = 0.3
        self.gt_zoom_min_size_cm   = 2.0
        self.accumulate_by_attenuation = False


# ==============================================================
# XML LOADER
# ==============================================================

def load_xml(xml_path: str,
             source: SourceCord,
             vals: SourceVals,
             detector: Detector,
             det_loc: Location,
             focal: FocalSpot,
             noise: NoiseConfig,
             scint: ScintillatorConfig,
             sim: SimConfig):
    """Parse simulation XML into configuration objects."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "SimulationConfig":
        raise ValueError("Root element must be <SimulationConfig>")

    # --- Source ---
    src = root.find("Source")
    if src is not None:
        loc = src.find("Location")
        if loc is not None and not source.is_set():
            source.set(float(loc.get("x")), float(loc.get("y")),
                       float(loc.get("z")), loc.get("units", "cm"))

        beam = src.find("Beam")
        if beam is not None:
            sim.beam_type = beam.get("type", "cone").strip().lower()

        fs = src.find("FocalSpot")
        if fs is None:
            fs = src.find("FocalSpotWiggle")  # legacy support
        if fs is not None:
            focal.enabled    = fs.get("enabled", "false").lower() == "true"
            focal.n_samples  = int(fs.findtext("n_samples", "1"))
            focal.sigma_x_cm = float(fs.findtext("sigma_x_cm",
                                     fs.findtext("std_dev_cm", "0.0")))
            focal.sigma_y_cm = float(fs.findtext("sigma_y_cm",
                                     fs.findtext("std_dev_cm", "0.0")))
            focal.sigma_z_cm = float(fs.findtext("sigma_z_cm", "0.0"))
            seed_el = fs.find("seed")
            focal.seed = int(seed_el.text) if seed_el is not None else None

        spec = src.find("Spectrum")
        if spec is not None and not vals.is_set():
            stype = spec.get("type", "mono").lower()
            vals.spectrum_type = stype
            vals.energy_units  = spec.get("units", "keV")
            if stype == "mono":
                mono = spec.find("Mono")
                if mono is not None:
                    vals.mono_energy     = float(mono.get("energy"))
                    vals.photons_per_ray = int(mono.get("photons_per_ray",
                                                         "25000"))
            elif stype == "poly":
                energies, weights = [], []
                for line in spec.findall("Line"):
                    energies.append(float(line.get("energy")))
                    weights.append(float(line.get("weight")))
                vals.poly_energies   = np.array(energies, dtype=np.float32)
                vals.poly_weights    = np.array(weights,  dtype=np.float32)
                vals.poly_weights   /= vals.poly_weights.sum()
                vals.photons_per_ray = int(spec.get("photons_per_ray", "25000"))

    # --- Object ---
    obj = root.find("Object")
    if obj is not None:
        loc = obj.find("Location")
        if loc is not None:
            sim.object_location = (float(loc.get("x")),
                                   float(loc.get("y")),
                                   float(loc.get("z")))
        rot = obj.find("Rotation")
        if rot is not None:
            sim.rotation = (float(rot.get("pitch", 0)),
                            float(rot.get("roll",  0)),
                            float(rot.get("yaw",   0)))
        mats = obj.find("Materials")
        if mats is not None:
            m = mats.find("Material")
            if m is not None:
                sim.material_compound = m.get("compound",
                                              m.get("name", "W"))
                sim.material_density  = float(m.get("density", "19.3"))
                vals.rho = sim.material_density
            else:
                mixture = mats.find("Mixture")
                sim.mixture_density = mixture.get("whole_density")
                for mix in mixture.findall("Mix"):
                    name = mix.get("name")
                    compound_mix = mix.get("compound")
                    percentage = float(mix.get("percentage"))
                    sim.mixture.append({"name":name,"compound":compound_mix,"percentage":percentage})
        jitter = obj.find("Jitter")
        if jitter is not None:
            sim.object_jitter_enabled = (
                jitter.get("enabled", "false").strip().lower() == "true"
            )
            sim.object_jitter_dist = jitter.get("dist", "normal").strip().lower()
            sim.object_jitter_x_cm = float(jitter.get("x_cm", "0.0"))
            sim.object_jitter_y_cm = float(jitter.get("y_cm", "0.0"))
            sim.object_jitter_z_cm = float(jitter.get("z_cm", "0.0"))
            seed_attr = jitter.get("seed")
            sim.object_jitter_seed = int(seed_attr) if seed_attr is not None else None
        rot_jitter = obj.find("RotationJitter")
        if rot_jitter is not None:
            sim.rotation_jitter_enabled     = rot_jitter.get("enabled", "false").strip().lower() == "true"
            sim.rotation_jitter_dist        = rot_jitter.get("dist", "normal").strip().lower()
            sim.rotation_jitter_pitch_deg   = float(rot_jitter.get("pitch_deg", "0.0"))
            sim.rotation_jitter_roll_deg   = float(rot_jitter.get("roll_deg", "0.0"))
            sim.rotation_jitter_yaw_deg   = float(rot_jitter.get("yaw_deg", "0.0"))
            seed_attr = rot_jitter.get("seed")
            sim.rotation_jitter_seed = int(seed_attr) if seed_attr is not None else None

        geom = obj.find("Geometry")
        if geom is not None:
            stl = geom.find("STL")
            if stl is not None:
                sim.stl_root  = stl.get("root_path", ".")
                # stl_units: units of the STL mesh geometry
                # gmsh outputs in mm by default
                sim.stl_units = stl.get("units", "mm")
                sim.stl_files = [f.text.strip() for f in stl.findall("File")]
                sim.accumulate_by_attenuation = (
                    stl.get("accumulate_by_attenuation", "false").strip().lower() == "true"
                )

    # --- Detector ---
    det = root.find("Detector")
    if det is not None:
        loc = det.find("Location")
        if loc is not None and not det_loc.is_set():
            det_loc.set(float(loc.get("x")), float(loc.get("y")),
                        float(loc.get("z")), loc.get("units", "cm"))
        dim = det.find("Dimensions")
        if dim is not None:
            detector.nx       = int(dim.get("nx", str(detector.nx)))
            detector.ny       = int(dim.get("ny", str(detector.ny)))
            # Support both width_cm (new) and rad_mag (legacy)
            if dim.get("width_cm") is not None:
                detector.width_cm  = float(dim.get("width_cm"))
                detector.height_cm = float(dim.get("height_cm",
                                                     dim.get("width_cm")))
            elif dim.get("rad_mag") is not None:
                # Legacy: rad_mag was half-width in cm
                rad = float(dim.get("rad_mag"))
                detector.width_cm  = 2.0 * rad
                detector.height_cm = 2.0 * rad
            detector.units = dim.get("units", "cm")

        resp = det.find("Response")
        if resp is not None:
            detector.bit_depth     = int(float(resp.get("bit_depth",    "16")))
            detector.full_well     = int(float(resp.get("full_well",    "65535")))
            detector.gain          = float(resp.get("gain",             "1.0"))
            detector.readout_noise = float(resp.get("readout_noise_e",  "5.0"))
            detector.dark_current  = float(resp.get("dark_current_adu", "0.1"))
            detector.exposure_time = float(resp.get("exposure_time_s",  "1.0"))
            detector.blur_sigma_px = float(resp.get("blur_sigma_px",    "1.0"))
            detector.fpn_sigma     = float(resp.get("fpn_sigma",        "0.01"))
            detector.fpn_seed      = int(float(resp.get("fpn_seed",     "42")))
            detector.distortion_k1 = float(resp.get("distortion_k1",   "0.0"))

    sc = root.find("Scintillator")
    if sc is not None:
        scint.enabled               = sc.get("enabled", "false").strip().lower() == "true"
        scint.material              = sc.get("material", scint.material)
        scint.thickness_cm          = float(sc.get("thickness_cm", scint.thickness_cm))
        scint.mu_cm_inv             = float(sc.get("mu_cm_inv", scint.mu_cm_inv))
        scint.light_yield           = float(sc.get("light_yield", scint.light_yield))
        scint.optical_blur_sigma_px = float(sc.get("optical_blur_sigma_px", scint.optical_blur_sigma_px))
        scint.saturation_enabled    = sc.get("saturation", "false").strip().lower() == "true"
        scint.saturation_level      = float(sc.get("saturation_level", scint.saturation_level))

    # --- Noise ---
    nz = root.find("Noise")
    if nz is not None:
        noise.poisson_enabled    = nz.get("poisson",  "true").lower() == "true"
        noise.photon_flux        = float(nz.get("photon_flux",       "20000"))
        noise.readout_sigma_adu  = float(nz.get("readout_sigma_adu", "5.0"))
        noise.scatter_enabled    = nz.get("scatter",  "true").lower() == "true"
        noise.scatter_fraction   = float(nz.get("scatter_fraction",  "0.05"))
        noise.scatter_blur_sigma = float(nz.get("scatter_blur_sigma","20.0"))
        noise.fpn_enabled        = nz.get("fpn",      "true").lower() == "true"
        noise.fpn_sigma          = float(nz.get("fpn_sigma",         "0.01"))
        noise.fpn_seed           = int(float(nz.get("fpn_seed",      "42")))
        s = nz.get("poisson_seed")
        noise.poisson_seed = int(s) if s else None
        r = nz.get("readout_seed")
        noise.readout_seed = int(r) if r else None

    # --- Output ---
    out = root.find("Output")
    if out is not None:
        sim.output_dir         = out.get("folder",           "radiographs")
        sim.save_png           = out.get("png",              "true").lower() == "true"
        sim.save_npy           = out.get("npy",              "true").lower() == "true"
        sim.save_flatfield     = out.get("flatfield",        "true").lower() == "true"
        sim.save_darkfield     = out.get("darkfield",        "true").lower() == "true"
        sim.n_projections      = int(out.get("n_projections",      "1"))
        sim.rotation_step_deg  = float(out.get("rotation_step_deg","0.0"))
        sim.gt_zoom_enabled    = out.get("gt_zoom",         "true").lower()  == "true"
        sim.gt_zoom_pixels     = int(out.get("gt_zoom_pixels",      "2048"))
        sim.gt_zoom_margin_cm  = float(out.get("gt_zoom_margin_cm", "1.0"))
        sim.gt_zoom_min_size_cm= float(out.get("gt_zoom_min_size_cm", "2.0"))


# ==============================================================
# GEOMETRY VALIDATOR
# ==============================================================

def validate_geometry(source: SourceCord,
                       det_loc: Location,
                       sim: SimConfig,
                       detector: Detector) -> list[str]:
    """
    Check the scene geometry for common configuration errors.
    Returns a list of warning strings (empty = no warnings).

    Checks performed:
    1. Source and detector are not at the same position.
    2. Object location is between source and detector (in Z).
    3. Detector physical size is reasonable for the geometry.
    4. STL files exist on disk.
    5. Pixel size is not absurdly small or large.
    6. Units consistency (all positions in same units).
    """
    warnings = []

    sx, sy, sz = source.get()
    dx, dy, dz = det_loc.get()

    # Check 1: source != detector
    dist_sd = np.sqrt((sx-dx)**2 + (sy-dy)**2 + (sz-dz)**2)
    if dist_sd < 1e-6:
        warnings.append(
            f"CRITICAL: Source {source.get()} == Detector {det_loc.get()}. "
            f"They must be at different positions."
        )

    # Check 2: object Z between source and detector
    if sim.object_location is not None:
        ox, oy, oz = sim.object_location
        z_min = min(sz, dz)
        z_max = max(sz, dz)
        if not (z_min <= oz <= z_max):
            warnings.append(
                f"WARNING: Object Z={oz} is outside the source-detector "
                f"Z range [{z_min}, {z_max}]. Object may not be in beam."
            )

        # Check 3: object not at same Z as detector
        if abs(oz - dz) < 1e-3:
            warnings.append(
                f"WARNING: Object Z={oz} is very close to Detector Z={dz}. "
                f"Object may be behind or on the detector plane."
            )

    # Check 4: STL files exist
    for stl_file in sim.stl_files:
        full_path = os.path.join(sim.stl_root, stl_file)
        if not os.path.exists(full_path):
            warnings.append(
                f"CRITICAL: STL file not found: {full_path}"
            )

    # Check 5: pixel size sanity
    px = detector.pixel_size_x()
    py = detector.pixel_size_y()
    if px < 1e-4 or py < 1e-4:
        warnings.append(
            f"WARNING: Pixel size ({px:.6f} x {py:.6f} {detector.units}) "
            f"is very small. Check detector width_cm and nx/ny."
        )
    if px > 10.0 or py > 10.0:
        warnings.append(
            f"WARNING: Pixel size ({px:.2f} x {py:.2f} {detector.units}) "
            f"is very large. Check detector width_cm and nx/ny."
        )

    # Check 6: detector field of view vs object size
    # Estimate object extent from STL bounding box if possible
    # (rough check: object location should be within detector FOV)
    if sim.object_location is not None:
        ox, oy, oz = sim.object_location
        # Magnification at object plane
        if abs(sz - dz) > 1e-6:
            mag = abs(sz - dz) / abs(sz - oz) if abs(sz - oz) > 1e-6 else 1.0
            fov_x = detector.width_cm  / mag
            fov_y = detector.height_cm / mag
            if abs(ox - sx) > fov_x / 2 or abs(oy - sy) > fov_y / 2:
                warnings.append(
                    f"WARNING: Object XY position ({ox}, {oy}) may be "
                    f"outside detector FOV ({fov_x:.2f} x {fov_y:.2f} cm "
                    f"at object plane, mag={mag:.2f}x)."
                )

    return warnings
def sample_object_jitter(sim: SimConfig, proj_i: int) ->tuple[float, float, float]:
    if not sim.object_jitter_enabled:
        return 0.0, 0.0, 0.0
    seed = None if sim.object_jitter_seed is None else int(sim.object_jitter_seed) + int(proj_i)
    rng = np.random.default_rng(seed)
    if sim.object_jitter_dist == "normal":
        jx = rng.normal(0.0, sim.object_jitter_x_cm)
        jy = rng.normal(0.0, sim.object_jitter_y_cm)
        jz = rng.normal(0.0, sim.object_jitter_z_cm)
    elif sim.object_jitter_dist == "uniform":
        jx = rng.uniform(-sim.object_jitter_x_cm, sim.object_jitter_x_cm)
        jy = rng.uniform(-sim.object_jitter_y_cm, sim.object_jitter_y_cm)
        jz = rng.uniform(-sim.object_jitter_z_cm, sim.object_jitter_z_cm)
    else:
        raise ValueError(f"Unknown ubject jitter distribution: {sim.object_jitter_dist}")
    return float(jx), float(jy), float(jz)

def sample_rotation_jitter(sim: SimConfig, proj_i: int) ->tuple[float, float, float]:
    if not sim.rotation_jitter_enabled:
        return 0.0, 0.0, 0.0
    seed = None if sim.rotation_jitter_seed is None else int(sim.rotation_jitter_seed) + int(proj_i)
    rng = np.random.default_rng(seed)
    if sim.rotation_jitter_dist == "normal":
        jp = rng.normal(0.0, sim.rotation_jitter_pitch_deg)
        jr = rng.normal(0.0, sim.rotation_jitter_roll_deg)
        jy = rng.normal(0.0, sim.rotation_jitter_yaw_deg)
    elif sim.rotation_jitter_dist == "uniform":
        jp = rng.uniform(-sim.rotation_jitter_pitch_deg, sim.rotation_jitter_pitch_deg)
        jr = rng.uniform(-sim.rotation_jitter_roll_deg, sim.rotation_jitter_roll_deg)
        jy = rng.uniform(-sim.rotation_jitter_yaw_deg, sim.rotation_jitter_yaw_deg)
    else:
        raise ValueError(f"Unknown ubject jitter distribution: {sim.rotation_jitter_dist}")
    return float(jp), float(jr), float(jy)
# ==============================================================
# Configurations Print
# ==============================================================
def save_config_json(source: SourceCord,
                     vals: SourceVals,
                     detector: Detector,
                     det_loc: Location,
                     focal: FocalSpot,
                     noise_cfg: NoiseConfig,
                     sim: SimConfig,
                     run_tag: str,
                     mu: float,
                     corrected_density: float):
    import json
    config = {
        "run_tag": run_tag,
        "source": {
            "x": source.x, "y": source.y, "z": source.z,
            "units": source.units,
            "beam_type": sim.beam_type,
        },
        "spectrum": {
            "type": vals.spectrum_type,
            "energy": vals.mono_energy,
            "energy_units": vals.energy_units,
            "photons_per_ray": vals.photons_per_ray,
        },
        "material": {
            "compound": sim.material_compound,
            "density_g_cm3": corrected_density,
            "linear_attenuation_cm-1":mu,
        },
        "detector": {
            "nx": detector.nx, "ny": detector.ny,
            "width_cm": detector.width_cm,
            "height_cm": detector.height_cm,
            "pixel_size_x_cm": detector.pixel_size_x(),
            "pixel_size_y_cm": detector.pixel_size_y(),
            "bit_depth": detector.bit_depth,
            "full_well": detector.full_well,
            "gain": detector.gain,
            "blur_sigma_px": detector.blur_sigma_px,
            "distortion_k1": detector.distortion_k1,
        },
        "detector_location":{
            "x": det_loc.x, "y": det_loc.y, "z": det_loc.z,
            "units": det_loc.units,
        },
        "object": {
            "stl_files": sim.stl_files,
            "stl_units": sim.stl_units,
        },
        "focal_spot": {
            "enabled": focal.enabled,
            "n_samples": focal.n_samples,
            "sigma_x_cm": focal.sigma_x_cm,
            "sigma_y_cm": focal.sigma_y_cm,
            "sigma_z_cm": focal.sigma_z_cm,
        },
        "noise": {
            "poisson_enabled": noise_cfg.poisson_enabled,
            "photon_flux": noise_cfg.photon_flux,
            "readout_sigma_adu": noise_cfg.readout_sigma_adu,
            "dark_current_adu": noise_cfg.dark_current_adu,
            "fpn_enabled": noise_cfg.fpn_enabled,
            "fpn_sigma": noise_cfg.fpn_sigma,
            "scatter_enabled": noise_cfg.scatter_enabled,
            "scatter_fraction": noise_cfg.scatter_fraction,
        },
        "output": {
            "folder": sim.output_dir,
            "save_png": sim.save_png,
            "save_npy": sim.save_npy,
            "save_flatfield": sim.save_flatfield,
            "save_darkfield": sim.save_darkfield,
        }
    }
    path = os.path.join(sim.output_dir, f"{run_tag}_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[+] Config saved {path}")
    

# ==============================================================
# DIAGNOSTIC PREFLIGHT
# ==============================================================

def diagnostic_preflight(source: SourceCord,
                           vals: SourceVals,
                           detector: Detector,
                           det_loc: Location,
                           sim: SimConfig,
                           preview_size: int = 128) -> tuple[bool, dict]:
    """
    Render a low-resolution preview and print detailed diagnostics.

    Unlike the previous preflight, this version:
    1. Prints the full scene configuration before rendering.
    2. Tries multiple detector positions/sizes if the first fails.
    3. Returns detailed info about what it found.
    4. Does NOT raise an exception — returns (ok, info) always.
    """

    print("\n" + "="*60)
    print("DIAGNOSTIC PREFLIGHT")
    print("="*60)
    print(f"  Source:   {source.get()} [{source.units}]")
    print(f"  Detector: {det_loc.get()} [{det_loc.units}]")
    print(f"  Det size: {detector.width_cm} x {detector.height_cm} cm")
    print(f"  Det res:  {detector.nx} x {detector.ny} px")
    print(f"  Px size:  {detector.pixel_size_x():.6f} x "
          f"{detector.pixel_size_y():.6f} {detector.units}")
    print(f"  Beam:     {sim.beam_type}")
    if sim.object_location:
        print(f"  Object:   {sim.object_location} [cm]")
    print(f"  STL root: {sim.stl_root}")
    for f in sim.stl_files:
        full = os.path.join(sim.stl_root, f)
        exists = os.path.exists(full)
        print(f"  STL:      {f}  [{'EXISTS' if exists else 'MISSING'}]")
    print(f"  Material: {sim.material_compound} @ "
          f"{sim.material_density} g/cm3")
    print(f"  STL units: {sim.stl_units}")

    # Compute source-detector distance
    sx, sy, sz = source.get()
    dx, dy, dz = det_loc.get()
    sdd = np.sqrt((sx-dx)**2 + (sy-dy)**2 + (sz-dz)**2)
    print(f"  SDD:      {sdd:.2f} {source.units}")

    if sim.object_location:
        ox, oy, oz = sim.object_location
        sod = np.sqrt((sx-ox)**2 + (sy-oy)**2 + (sz-oz)**2)
        odd = np.sqrt((ox-dx)**2 + (oy-dy)**2 + (oz-dz)**2)
        mag = sdd / sod if sod > 1e-6 else float('inf')
        print(f"  SOD:      {sod:.2f} {source.units}")
        print(f"  ODD:      {odd:.2f} {source.units}")
        print(f"  Magnification: {mag:.4f}x")

    print("-"*60)

    # Set preview resolution
    px_size = detector.width_cm / float(preview_size)
    gvxr.setDetectorNumberOfPixels(preview_size, preview_size)
    gvxr.setDetectorPixelSize(px_size, px_size, detector.units)
    gvxr.setDetectorPosition(*det_loc.get(), det_loc.units)

    img = np.array(gvxr.computeXRayImage(), dtype=np.float32)
    if img.ndim == 1:
        img = img.reshape((preview_size, preview_size))

    img_min  = float(img.min())
    img_max  = float(img.max())
    img_mean = float(img.mean())
    img_std  = float(img.std())

    print(f"  Preview image stats:")
    print(f"    min={img_min:.2f}  max={img_max:.2f}  "
          f"mean={img_mean:.2f}  std={img_std:.2f}")
    print(f"    shape={img.shape}  dtype={img.dtype}")

    info = {
        "img_min": img_min, "img_max": img_max,
        "img_mean": img_mean, "img_std": img_std,
        "preview_shape": img.shape,
    }

    # Flat image: object not in beam
    if img_std < 1e-3 * max(abs(img_max), 1.0):
        print("  RESULT: Image is flat — object not in beam or scene is empty.")
        print("  SUGGESTIONS:")
        print("    1. Check that STL files exist and paths are correct.")
        print("    2. Check that object Z is between source Z and detector Z.")
        print("    3. Check that STL units match the scene units.")
        print(f"       STL units='{sim.stl_units}', scene units='cm'.")
        print(f"       If STL is in mm, object spans ~{sim.material_density:.0f}x "
              f"smaller than expected.")
        print("    4. Check that object XY position is within detector FOV.")
        print("    5. Try increasing detector width_cm or moving object closer.")
        info["reason"] = "flat_image_object_not_in_beam"
        print("="*60 + "\n")
        return False, info

    # Check attenuation: object should reduce intensity somewhere
    thresh = 0.98 * img_max
    mask   = img < thresh
    area   = float(mask.mean())
    info["area_frac"] = area

    print(f"  Attenuated area fraction: {area:.4f} "
          f"(threshold={thresh:.2f})")

    if area < 0.002:
        print("  RESULT: Object present but very small in FOV.")
        print("  SUGGESTIONS:")
        print("    1. Move object closer to detector.")
        print("    2. Increase detector width_cm.")
        print("    3. Check object scale (STL units vs scene units).")
        info["reason"] = "object_too_small"
        print("="*60 + "\n")
        return False, info

    if area > 0.95:
        print("  RESULT: Object fills entire FOV — may be clipped.")
        print("  SUGGESTIONS:")
        print("    1. Increase detector width_cm.")
        print("    2. Move detector further from object.")
        info["reason"] = "object_too_large"
        print("="*60 + "\n")
        return False, info

    # Check for edge clipping
    ys, xs = np.where(mask)
    margin = max(2, int(0.02 * preview_size))
    clipped = (xs.min() <= margin or ys.min() <= margin or
               xs.max() >= preview_size - 1 - margin or
               ys.max() >= preview_size - 1 - margin)
    info["clipped"] = clipped
    info["bbox_px"] = (int(xs.min()), int(ys.min()),
                       int(xs.max()), int(ys.max()))

    if clipped:
        print(f"  RESULT: Object footprint touches detector edge "
              f"(bbox={info['bbox_px']}).")
        print("  SUGGESTIONS:")
        print("    1. Increase detector width_cm.")
        print("    2. Center object in beam (check XY position).")
        info["reason"] = "footprint_clipped"
        print("="*60 + "\n")
        return False, info

    print(f"  RESULT: OK — object visible, area={area:.3f}, "
          f"bbox={info['bbox_px']}")
    print("="*60 + "\n")
    return True, info
"""
def validate_saved_image_coverage(
    img: np.ndarray,
    name: str = "saved_image",
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.95,
    edge_margin_frac: float = 0.02,
    atten_thresh_frac: float = 0.98,
    ) -> tuple[bool, dict]:
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2:
        return False, {
            "reason": "image_not_2d",
            "name": name,
            "shape": img.shape,
        }
    if not np.all(np.isfinite(img)):
        return False, {
            "reason": "non_finite_pixels",
            "name": name,
        }
    img_min = float(img.min())
    img_max = float(img.max())
    img_mean = float(img.mean())
    img_std = float(img.std())

    if img_max <= 0:
        return False, {
            "reason": "zero_or_negative_image",
            "name": name,
            "img_min": img_min,
            "img_max": img_max,
            "img_mean": img_mean,
            "img_std": img_std,
        }
    if img_std < 1e-4 * max(abs(img_max), 1.0):
        return False, {
            "reason": "flat_image_object_not_visible",
            "name": name,
            "img_min": img_min,
            "img_max": img_max,
            "img_mean": img_mean,
            "img_std": img_std,
        }
    mask = img < (atten_thresh_frac * img_max)
    area = float(mask.mean())
    if area < min_area_frac:
        return False, {
            "reason": "object_too_small_or_not_visible",
            "name": name,
            "area_frac": area,
            "img_min": img_min,
            "img_max": img_max,
            "img_mean": img_mean,
            "img_std": img_std,
        }
    if area > max_area_frac:
        return False, {
            "reason": "object_too_large_or_image_mostly_attenuated",
            "name": name,
            "area_frac": area,
            "img_min": img_min,
            "img_max": img_max,
            "img_mean": img_mean,
            "img_std": img_std,
        }
    ys, xs = np.where(mask)
    h, w = img.shape
    margin_x = max(2, int(edge_margin_frac * w))
    margin_y = max(2, int(edge_margin_frac * h))

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    clipped = (
        x0 <= margin_x or
        y0 <= margin_y or
        x1 >= w - 1 - margin_x or
        y1 >= h - 1 - margin_y
    )

    info = {
        "reason": "ok",
        "name": name,
        "area_frac": area,
        "bbox_px": (x0, y0, x1, y1),
        "clipped": clipped,
        "img_min": img_min,
        "img_max": img_max,
        "img_mean": img_mean,
        "img_std": img_std,
    }
    if clipped:
        info["reason"] = "object_touches_saved_image_edge"
        return False, info
    return True, info

def ask_user_autocenter_object(final_info: dict) -> bool:
    print("\n" + "="*70)
    print("[FINAL COVERAGE ERROR]")
    print("The object is not valid in the final saved image.")
    print("Press 1 to move the object projection to the middle of the detector and retry")
    print("Press 2 to stop and raise the error")
    print("\n" + "="*70)
    while True:
        ans = input("Choice [1=acept move, 2=deny]").strip()
        if ans == "1":
            return True
        if ans == "2":
            return False
        print("Invalid choice, Enter 1 or 2")

def compute_object_centering_shift_cm(
    final_info: dict,
    detector: Detector,
    source: SourceCord,
    det_loc: Location,
    sim: SimConfig,
    ) -> tuple[float, float]:

    if sim.object_location is None:
        return 0.0, 0.0
    sx, sy, sz = source.get()
    dx, dy, dz = det_loc.get()
    ox, oy, oz = sim.object_location

    if "bbox_px" in final_info:
        x0 ,y0, x1, y1 = final_info["bbox_px"]

        bbox_cx_px = 0.5 * (x0 + x1)
        bbox_cy_px = 0.5 * (y0 + y1)

        img_cx_px = detector.nx / 2.0
        img_cy_px = detector.ny / 2.0

        shift_detector_x_cm = (img_cx_px - bbox_cx_px) * detector.pixel_size_x()
        shift_detector_y_cm = (img_cy_px - bbox_cy_px) * detector.pixel_size_y()

        shift_detector_y_cm = -shift_detector_y_cm

        if sim.beam_type == "cone":
            sdd = np.sqrt((sx - dx)**2 + (sy - dy)**2 + (sz - dz)**2)
            sod = np.sqrt((sx - ox)**2 + (sy - oy)**2 + (sz - oz)**2)
            mag = sdd / max(sod, 1e-12)
        else:
            mag = 1.0
        
        shift_detector_x_cm = shift_detector_x_cm / max(mag, 1e-12)
        shift_detector_y_cm = shift_detector_y_cm / max(mag, 1e-12)

        return float(shift_detector_x_cm), float(shift_detector_y_cm)
    
    if sim.beam_type == "cone":
        if abs(dz - sz) < 1e-12:
            return 0.0, 0.0
        t = (oz - sz) / (dz - sz)
        target_x = sx + t * (dx - sx)
        target_y = sy + t * (dy - sy)
        return float(target_x - ox), float(target_y - oy)
    return float(dx - ox), float(dy - oy)

def move_object_to_center_and_report(
    final_info: dict,
    detector: Detector,
    source: SourceCord,
    det_loc: Location,
    sim: SimConfig,
    ) -> bool:
    if sim.object_location is None:
        print("[AUTOCENTER] Cannot move object becuase sim.object_location is None")
        return False
    ox, oy, oz = sim.object_location
    shift_x_cm, shift_y_cm = compute_object_centering_shift_cm(final_info=final_info, detector=detector, source=source, det_loc=det_loc, sim=sim)
    if abs(shift_x_cm) < 1e-12 and abs(shift_y_cm) < 1e-12:
        print("[AUTOCENTER] Completed shift is zero. Not retrying.")
        return False
    sim.object_location = (
        float(ox + shift_x_cm),
        float(oy + shift_y_cm),
        float(oz),
    )
    print("\n[AUTOCENTER] Moving object location:")
    print(f"old location: ({ox:.6f}, {oy:.6f}, {oz:.6f}) cm")
    print(f"shift: dx={shift_x_cm:.6f} cm, dy={shift_y_cm:.6f}")
    print(f"new location: {sim.object_location} cm")
    print("[AUTOCENTER] Retrying projection ....")
    return True
"""
# ==============================================================
# GVXR SETUP HELPERS
# ==============================================================

def setup_source(source: SourceCord, vals: SourceVals,
                 sim: SimConfig, sx=None, sy=None, sz=None):
    """
    Configure gvxr source position, beam type, and spectrum.

    Parameters
    ----------
    source : SourceCord
        Source position (x, y, z) and units.
    vals : SourceVals
        Source spectrum and beam parameters.
    sim : SimConfig
        Simulation configuration (beam type, etc.).
    sx, sy, sz : float, optional
        Override source position (default: use `source` values).
    """
    x, y, z = (sx, sy, sz) if sx is not None else source.get()

    # Set beam type
    if sim.beam_type == "parallel":
        gvxr.useParallelSource()
    elif sim.beam_type == "cone":
        gvxr.usePointSource()
    else:
        raise ValueError(f"Unknown beam_type: {sim.beam_type}")

    # Set source position
    gvxr.setSourcePosition(float(x), float(y), float(z), source.units)

    # Set spectrum
    if vals.spectrum_type == "mono":
        gvxr.setMonoChromatic(
            float(vals.mono_energy),
            vals.energy_units,
            int(vals.photons_per_ray)
        )
    elif vals.spectrum_type == "poly":
        # Polychromatic: pass arrays to gvxr
        gvxr.setEnergySpectrum(
            vals.poly_energies.tolist(),
            vals.poly_weights.tolist(),
            vals.energy_units
        )
    else:
        raise ValueError(f"Unknown spectrum_type: {vals.spectrum_type}")
    
    
def setup_detector(detector: Detector, det_loc: Location):
    """
    Configure gvxr detector geometry.

    Parameters
    ----------
    detector : Detector
        Detector geometry and response parameters.
    det_loc : Location
        Detector position (x, y, z) and units.
    """
    # Correct pixel size: physical width / number of pixels
    px_size_x = detector.width_cm / float(detector.nx)
    px_size_y = detector.height_cm / float(detector.ny)

    gvxr.setDetectorNumberOfPixels(detector.nx, detector.ny)
    gvxr.setDetectorPixelSize(px_size_x, px_size_y, detector.units)
    gvxr.setDetectorPosition(*det_loc.get(), det_loc.units)
    gvxr.setDetectorUpVector(0.0, -1.0, 0.0)  # Straight up
def calc_porosity(mixes, final_density):
    density_list = {
        "W": 19.3, "FE": 7.8, "AL": 2.7, "CU": 8.9, "PB": 11.3,
        "TI": 4.5, "NI": 8.9, "AU": 19.3, "AG": 10.5, "SN": 7.3,
        "CR": 7.1, "ZN": 7.1, "MO": 10.3, "CO": 8.9, "SI": 2.3,
    }
    inverse_density = 0.0
    porosity_list = []
    for part in mixes:
        frac = float(part["percentage"])/100
        rho = density_list.get(part["compound"].strip().upper())
        if rho is None:
            print(f"[WARNING] Unknown element '{part['compound']}', skipping")
            continue
        inverse_density += (frac / rho)
        print(f"{part['compound']}, {part['percentage']}, rho={rho} g/cm3")

    rho_mix = 1.0 / inverse_density
    porosity = (1.0 - (float(final_density)/ rho_mix)) * 100
    print(f"[ATTENTION] Mixture has porosity of {porosity}%, meaning of this mixture"
          f"{porosity}% of the material is void space")
    return porosity, rho_mix
def get_linear_attenuation(compound: str, density: float, energy_keV: float) -> float:
    SS_aliases={
        "SS", "SS316", "SS304", "STAINLESS", "STAINLESS_STEEL", "STEEL"
    }
    key = compound.strip().upper()
    if key in SS_aliases:
        mu_rho_ss = 0.04237
        if density is None or density <= 0:
            density = 7.93
        mu = mu_rho_ss * density
        print(f"[INFO] Stainless-steel preset for {compound} "
              f"@ {energy_keV} keV: mu/rho={mu_rho_ss:.5f} cm2/g"
              f"rho={density:.3f} g/cm3, mu={mu:.5f} cm-1")
        return mu, density
    compound_list = {
        "W": 74, "FE": 26, "AL": 13, "CU": 29, "PB": 82,
        "TI": 22, "NI": 28, "AU": 79, "AG": 47, "SN": 50,
        "CR": 24, "ZN": 30, "MO": 42, "CO": 27, "SI": 14,
    }

    #Mass Attenuation coefficient for elements at 2250 keV
    #Mass attenuation coefficient * density = linear coefficient measured in cm^(-1)
    #mu = (nist/rho) * rho
    #gvxr.setDensity() does this automatically to calculate density
    #this function manually does this calculation then bypasses gvxr.setDensity by using gvxr.setLinearAttenuationCoefficient
    #this solves the issue because gvxr looks up a database and takes the value at that energy then multiplies it by the density, 
    #but at high energies there may not be value or table, so it returns 0 causing a crash, this solves the problem by giving
    NIST_2250KEV = {
        "W": 0.0396, "FE": 0.0425, "AL": 0.0453, "CU": 0.0418, "PB": 0.0496,
        "TI": 0.0441, "NI": 0.0421, "AU": 0.0456, "AG": 0.0413, "SN": 0.0422,
        "CR": 0.0430, "ZN": 0.0422, "MO": 0.0407, "CO": 0.0423, "SI": 0.0454,
    }

    density_list = {
        "W": 19.3, "FE": 7.8, "AL": 2.7, "CU": 8.9, "PB": 11.3,
        "TI": 4.5, "NI": 8.9, "AU": 19.3, "AG": 10.5, "SN": 7.3,
        "CR": 7.1, "ZN": 7.1, "MO": 10.3, "CO": 8.9, "SI": 2.3,
    }
    z = compound_list.get(compound.strip().upper())
    if z is None:
        print(f"[WARNING] Unknown element {compound}, using default mu=0.1 density=1 cm-1")
        return 0.1, 1
    listed_density = density_list.get(compound.strip().upper())
    if listed_density is not None:
        print(f"[WARNING] Compound {compound} is in the compound list."
              f"Using density from periodic table ({listed_density})")
        density = listed_density
    try:
        import xraylib
        mu_rho = xraylib.CS_TOTAL(z, energy_keV)
        print(f"[INFO] xraylib mu/rho for {compound} at {energy_keV} keV: {mu_rho:.6f} cm2/g")
        return mu_rho * density, density
    except Exception:
        mu_rho = NIST_2250KEV.get(compound.strip().upper())
        if mu_rho is None:
            print(f"[WARNING] No NIST data for {compound}, using default mu=0.1 density=1 cm-1")
            return 0.1, 1
        print(f"[INFO] NIST fallback mu/rho for {compound} at {energy_keV} keV: {mu_rho:.6f} cm2/g")
        return mu_rho * density, density
def get_linear_attenuation_mixture(mixes, bulk_density, energy_keV):
    compound_list = {
        "W": 74, "FE": 26, "AL": 13, "CU": 29, "PB": 82,
        "TI": 22, "NI": 28, "AU": 79, "AG": 47, "SN": 50,
        "CR": 24, "ZN": 30, "MO": 42, "CO": 27, "SI": 14,
    }
    NIST_2250KEV = {
        "W": 0.0396, "FE": 0.0425, "AL": 0.0453, "CU": 0.0418, "PB": 0.0496,
        "TI": 0.0441, "NI": 0.0421, "AU": 0.0456, "AG": 0.0413, "SN": 0.0422,
        "CR": 0.0430, "ZN": 0.0422, "MO": 0.0407, "CO": 0.0423, "SI": 0.0454,
    }
    xraylib_ok = True
    try:
        import xraylib
        mu_rho_mix = 0.0
        for part in mixes:
            key   = part["compound"].strip().upper()
            frac  = float(part["percentage"]) / 100
            z     = compound_list.get(key)
            if z is None:
                print(f"[WARNING] Unknown element {key}, skipping")
                continue
            mu_rho = xraylib.CS_TOTAL(z, energy_keV)
            mu_rho_mix += frac * mu_rho
            print(f"[xraylib] {key:3s} w={frac:.3f} mu/rho={mu_rho:.6f} cm2/g")
    except Exception as e:
        print(f"[INFO] xraylib unavailable or failed ({e}), using NIST fallback (2250 keV).")
        xraylib_ok =False
    if not xraylib_ok:
        mu_rho_mix = 0.0
        for part in mixes:
            key   = part["compound"].strip().upper()
            frac  = float(part["percentage"]) / 100
            mu_rho    = NIST_2250KEV.get(key)
            if z is None:
                print(f"[WARNING] No NIST data for {key}, skipping")
                continue
            mu_rho_mix += frac * mu_rho
            print(f"[NIST] {key:3s} w={frac:.3f} mu/rho={mu_rho:.6f} cm2/g")
    mu = mu_rho_mix * bulk_density
    return mu, not xraylib_ok
def load_objects(sim: SimConfig, vals: SourceVals,
                 node_prefix: str = "obj", rotation_jitter_deg=(0.0, 0.0, 0.0)):
    """
    Load STL files and assign material.

    Parameters
    ----------
    sim : SimConfig
        Simulation configuration (STL files, material, etc.).
    vals : SourceVals
        Source spectrum and material density.
    node_prefix : str
        Prefix for gvxr node IDs.
    """
    for i, stl_file in enumerate(sim.stl_files):
        stl_path = os.path.join(sim.stl_root, stl_file)
        node_id  = f"{node_prefix}_{i}"

        if not os.path.exists(stl_path):
            raise FileNotFoundError(f"STL not found: {stl_path}")

        # Pass STL units to gvxr so it scales the mesh to scene units
        # If STL is in mm and scene is in cm, gvxr divides by 10
        gvxr.loadMeshFile(node_id, stl_path, sim.stl_units)
        


        if sim.rotation is not None:
            pitch, roll, yaw = sim.rotation
            if abs(yaw)   > 1e-9: gvxr.rotateNode(node_id, yaw,   0, 0, 1)
            if abs(pitch) > 1e-9: gvxr.rotateNode(node_id, pitch, 1, 0, 0)
            if abs(roll)  > 1e-9: gvxr.rotateNode(node_id, roll,  0, 1, 0)
        jpitch, jroll, jyaw = rotation_jitter_deg
        if abs(jyaw)   > 1e-9: gvxr.rotateNode(node_id, jyaw,   0, 0, 1)
        if abs(jpitch) > 1e-9: gvxr.rotateNode(node_id, jpitch, 1, 0, 0)
        if abs(jroll)  > 1e-9: gvxr.rotateNode(node_id, jroll,  0, 1, 0)
        if sim.object_location is not None:
            ox, oy, oz = sim.object_location
            # Object location is in scene units (cm)
            gvxr.translateNode(node_id, ox, oy, oz, "cm")
        # Assign material
        compound = sim.material_compound.strip()
        try:
            gvxr.setElement(node_id, compound)
        except Exception:
            gvxr.setCompound(node_id, compound)

       
        energy_keV = vals.mono_energy
        if vals.energy_units.lower() == "mev":
            energy_keV = vals.mono_energy * 1000.0
        if sim.mixture != []:
            print(f"[DEBUG] sim.mixture THE VALUE IS {sim.mixture}")
            porosity, rho_mix = calc_porosity(sim.mixture, sim.mixture_density)
            bulk_density = float(sim.mixture_density)
            mu, used_fallback = get_linear_attenuation_mixture(
                sim.mixture, bulk_density, energy_keV
            )
            elements = [p["compound"].strip() for p in sim.mixture]
            percentages = [float(p["percentage"]) for p in sim.mixture]
            gvxr.setMixture(node_id, elements, percentages)
            gvxr.setDensity(node_id, bulk_density, "g/cm3")
            gvxr.setLinearAttenuationCoefficient(node_id, mu, "cm-1")
            corrected_density = bulk_density
            print(f"  Loaded: {stl_file} (units={sim.stl_units}) "
                  f"as mixture @ {bulk_density:.4f} g/cm3"
                  f"(porosity={porosity:.2f}% mu={mu:.6f} cm-1)")
        else:
            mu, corrected_density = get_linear_attenuation(
                sim.material_compound,
                sim.material_density,
                energy_keV
            )
            sim.material_density = corrected_density
            gvxr.setDensity(node_id, float(corrected_density), "g/cm3")
            print(f"[INFO] Linear attenuation for {sim.material_compound}"
                  f"@ {energy_keV} keV: {mu:.6f} cm-1")
            gvxr.setLinearAttenuationCoefficient(node_id, mu, "cm-1")

            print(f"  Loaded: {stl_file} (units={sim.stl_units}) "
                  f"as {compound} @ {sim.material_density} g/cm3")
    return mu, corrected_density
        

# ==============================================================
# IMAGE PROCESSING
# ==============================================================    
def apply_scintillator_response(img_primary: np.ndarray,
                                scint: ScintillatorConfig,
                                detector: Detector,
                                ) -> np.ndarray:

    img = np.asarray(img_primary, dtype=np.float32)
    if not scint.enabled:
        return img
    thickness = max(float(scint.thickness_cm), 0.0)
    mu = max(float(scint.mu_cm_inv), 0.0)
    absorption_efficiency = 1.0 - np.exp(-mu * thickness)
    signal = img * absorption_efficiency * float(scint.light_yield)
    if scint.optical_blur_sigma_px > 0:
        signal = gaussian_filter(signal.astype(np.float32), sigma=float(scint.optical_blur_sigma_px))
    if scint.saturation_enabled:
        sat = max(float(scint.saturation_level), 1e-12)
        signal = sat * (1.0 - np.exp(-signal / sat))
    print(
        "[SCINTILLATOR]"
        f"enabled material={scint.material}"
        f"thickness_cm={scint.thickness_cm}"
        f"mu_cm_inv={scint.mu_cm_inv}"
        f"abs_eff={absorption_efficiency:.4f}"
        f"light_yield={scint.light_yield}"
        f"optical_blur_sigma_px={scint.optical_blur_sigma_px}"
    )
    return signal.astype(np.float32)


def compute_focal_spot_image(source: SourceCord,
                              vals: SourceVals,
                              detector: Detector,
                              det_loc: Location,
                              focal: FocalSpot,
                              sim: SimConfig) -> np.ndarray:
    """
    Simulate finite focal spot size by averaging multiple projections
    with source position sampled from a 3D Gaussian distribution.

    Parameters
    ----------
    source : SourceCord
        Source position (x, y, z) and units.
    vals : SourceVals
        Source spectrum and beam parameters.
    detector : Detector
        Detector geometry and response parameters.
    det_loc : Location
        Detector position (x, y, z) and units.
    focal : FocalSpot
        Focal spot parameters (size, samples, etc.).
    sim : SimConfig
        Simulation configuration (beam type, etc.).

    Returns
    -------
    np.ndarray
        Averaged image with focal spot blur applied.
    """
    rng = np.random.default_rng(focal.seed)
    bx, by, bz = source.get()
    dx, dy, dz = det_loc.get()
    acc = np.zeros((detector.ny, detector.nx), dtype=np.float64)

    for _ in range(focal.n_samples):
        gvxr.setSourcePosition(*source.get(), source.units)
        jx = rng.normal(0.0, focal.sigma_x_cm)
        jy = rng.normal(0.0, focal.sigma_y_cm)
        jz = rng.normal(0.0, focal.sigma_z_cm)
        if sim.beam_type == "cone":
            gvxr.setSourcePosition(bx+jx, by+jy, bz+jz, source.units)
        else:
            gvxr.setDetectorPosition(dx+jx, dy+jy, dz, det_loc.units)
        acc += compute_image(detector).astype(np.float64)

    gvxr.setSourcePosition(bx, by, bz, source.units)
    gvxr.setDetectorPosition(dx, dy, dz, det_loc.units)
    return (acc / float(focal.n_samples)).astype(np.float32)

def compute_gt_zoom_detector(sim: SimConfig,
                            source: SourceCord,
                            det_loc: Location,
                            base_detector: Detector) -> Detector:
    """
    Compute a zoomed in dector defined projection of the ground truth.
    """  
    all_pts_cm = []
    unit_scale = 0.1 if sim.stl_units.lower() == "mm" else 1.0
    for stl_file in sim.stl_files:
        stl_path = os.path.join(sim.stl_root, stl_file)
        mesh = trimesh.load(stl_path, force="mesh")
        verts = np.asarray(mesh.vertices, dtype=np.float64) * unit_scale

        if sim.rotation is not None:
            pitch, roll, yaw = sim.rotation
            verts = _apply_rotation(verts, yaw, axis=(0,0,1))
            verts = _apply_rotation(verts, pitch, axis=(1,0,0))
            verts = _apply_rotation(verts, roll, axis=(0,1,0))
        if sim.object_location is not None:
            ox, oy, oz = sim.object_location
            verts = verts + np.array([ox, oy, oz])
        all_pts_cm.append(verts)
    pts = np.concatenate(all_pts_cm, axis=0)

    sx, sy, sz = source.get()
    dx, dy, dz = det_loc.get()

    if sim.beam_type == "cone":
        dz_ray = pts[:, 2] - sz
        safe = np.abs(dz_ray) > 1e-9
        t = np.where(safe, (dz - sz)/ np.where(safe, dz_ray, 1.0), 1.0)
        proj_x = sx + t * (pts[:, 0] - sx)
        proj_y = sy + t * (pts[:, 1] - sy)
    else:
        proj_x = pts[:, 0]
        proj_y = pts[:, 1]
    
    px_min, px_max = float(proj_x.min()), float(proj_x.max())
    py_min, py_max = float(proj_y.min()), float(proj_y.max())
    proj_w = px_max - px_min
    proj_h = py_max - py_min

    side_cm = max(proj_w, proj_h) + 2.0 * sim.gt_zoom_margin_cm
    side_cm = max(side_cm, sim.gt_zoom_min_size_cm)

    side_cm = min(side_cm, float(min(base_detector.width_cm, base_detector.height_cm)))

    gt_det              = Detector()
    gt_det.nx           = int(sim.gt_zoom_pixels)
    gt_det.ny           = int(sim.gt_zoom_pixels)
    gt_det.width_cm     = side_cm
    gt_det.height_cm    = side_cm
    gt_det.units        = base_detector.units
    gt_det.bit_depth    = base_detector.bit_depth
    gt_det.full_well    = base_detector.full_well
    gt_det.gain         = base_detector.gain
    gt_det.blur_sigma_px = base_detector.blur_sigma_px
    gt_det.fpn_sigma    = base_detector.fpn_sigma
    gt_det.distortion_k1 = base_detector.distortion_k1

    cx = 0.5 * (px_min + px_max)
    cy = 0.5 * (py_min + py_max)

    return gt_det, (cx, cy)

def _apply_rotation(verts: np.ndarray, angle_deg: float, axis=(0,0,1)) -> np.ndarray:
    if abs(angle_deg) < 1e-9:
        return verts
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    ax, ay, az = axis
    K = np.array([[0, -az, ay],
                  [az, 0 , -ax],
                  [-ay, ax, 0]], dtype=np.float64)
    R = np.eye(3) + s * K + (1-c) * (K@K)
    return verts @ R.T
def compute_image(detector: Detector) -> np.ndarray:
    """
    Compute one X-ray image and return as 2D float32 array.

    Parameters
    ----------
    detector : Detector
        Detector geometry and response parameters.

    Returns
    -------
    np.ndarray
        Computed X-ray image.
    """
    img = np.array(gvxr.computeXRayImage(), dtype=np.float32)
    if img.ndim == 1:
        img = img.reshape((detector.ny, detector.nx))
    return img

def apply_detector_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Apply Gaussian MTF blur to simulate scintillator/optical blur.

    Parameters
    ----------
    img : np.ndarray
        Input image (float32).
    sigma_px : float
        Blur kernel sigma in pixels.

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    if sigma_px <= 0:
        return img.astype(np.float32)
    return gaussian_filter(img.astype(np.float32), sigma=sigma_px)

def apply_geometric_distortion(img: np.ndarray, k1: float) -> np.ndarray:
    """
    Apply radial geometric distortion (barrel k1<0, pincushion k1>0).

    Parameters
    ----------
    img : np.ndarray
        Input image (float32).
    k1 : float
        Distortion coefficient (Brown-Conrady model).

    Returns
    -------
    np.ndarray
        Distorted image.
    """
    if abs(k1) < 1e-9:
        return img.astype(np.float32)

    ny, nx = img.shape
    cx, cy = nx / 2.0, ny / 2.0
    ys, xs = np.mgrid[0:ny, 0:nx]
    xn = (xs - cx) / cx
    yn = (ys - cy) / cy
    r2 = xn**2 + yn**2
    factor = 1.0 + k1 * r2
    xd_px = xn * factor * cx + cx
    yd_px = yn * factor * cy + cy
    coords = np.array([yd_px.ravel(), xd_px.ravel()])
    return map_coordinates(img.astype(np.float32), coords,
                           order=1, mode='nearest').reshape(ny, nx)

def apply_realistic_noise(img_primary: np.ndarray,
                           noise_cfg: NoiseConfig,
                           detector: Detector) -> np.ndarray:
    """
    Apply physically correct noise chain to a primary beam image.

    Parameters
    ----------
    img_primary : np.ndarray
        Raw gvxr output image (photon counts or intensity, float32).
    noise_cfg : NoiseConfig
        Noise model parameters.
    detector : Detector
        Detector response parameters.

    Returns
    -------
    np.ndarray
        Noisy image in the same units as img_primary (float32).
    """
    img = img_primary.astype(np.float64)
    img_max = float(img.max())
    if img_max <= 0:
        return img_primary.astype(np.float32)

    img_norm = img / (img_primary.max() + 1e-12)
    N_photons = img_norm * noise_cfg.photon_flux

    # Poisson quantum noise
    if noise_cfg.poisson_enabled:
        rng_p = np.random.default_rng(noise_cfg.poisson_seed)
        N_detected = rng_p.poisson(np.clip(N_photons, 0, None)).astype(np.float64)
    else:
        N_detected = N_photons.copy()

    adu = N_detected / max(detector.gain, 1e-12)
    dark_adu = detector.dark_current * detector.exposure_time
    adu += dark_adu

    rng_r = np.random.default_rng(noise_cfg.readout_seed)
    adu += rng_r.normal(0.0, noise_cfg.readout_sigma_adu, size=img.shape)

    if noise_cfg.fpn_enabled:
        rng_fpn = np.random.default_rng(noise_cfg.fpn_seed)
        adu *= 1.0 + rng_fpn.normal(0.0, noise_cfg.fpn_sigma, size=img.shape)

    adu = np.clip(adu, 0.0, float(detector.full_well))

    if noise_cfg.scatter_enabled and noise_cfg.scatter_fraction > 0:
        scatter = gaussian_filter(
            (N_photons / max(noise_cfg.photon_flux, 1e-12)).astype(np.float32),
            sigma=noise_cfg.scatter_blur_sigma
        ).astype(np.float64)
        adu += (scatter * noise_cfg.scatter_fraction
                * noise_cfg.photon_flux / max(detector.gain, 1e-12))
        adu = np.clip(adu, 0.0, float(detector.full_well))

    adu_signal = np.clip(adu - dark_adu, 0.0, float(detector.full_well))
    img_noisy = (adu_signal * detector.gain
                 / max(noise_cfg.photon_flux, 1e-12)) * img_max
    return img_noisy.astype(np.float32)


# ==============================================================
# FLAT FIELD AND DARK FIELD
# ==============================================================

def generate_flat_field(detector: Detector,
                         noise_cfg: NoiseConfig,
                         open_beam_value: float = 1.0) -> np.ndarray:
    """
    Generate a flat field image (open beam, no object).
    Includes FPN and readout noise but no Poisson (high photon count).
    Used for flat field correction:
      I_corrected = (I_raw - I_dark) / (I_flat - I_dark)
    """
    ny, nx = detector.ny, detector.nx
    flat = np.ones((ny, nx), dtype=np.float64) * open_beam_value

    # FPN on flat field
    if noise_cfg.fpn_enabled:
        rng_fpn = np.random.default_rng(noise_cfg.fpn_seed)
        fpn_map = 1.0 + rng_fpn.normal(0.0, noise_cfg.fpn_sigma,
                                        size=(ny, nx))
        flat *= fpn_map

    # Readout noise on flat field
    rng_r = np.random.default_rng(noise_cfg.readout_seed)
    flat += rng_r.normal(0.0, noise_cfg.readout_sigma_adu / noise_cfg.photon_flux,
                         size=(ny, nx))

    return np.clip(flat, 0.0, 1.0).astype(np.float32)


def generate_dark_field(detector: Detector,
                         noise_cfg: NoiseConfig) -> np.ndarray:
    """
    Generate a dark field image (no beam, shutter closed).
    Contains only dark current and readout noise.
    """
    ny, nx = detector.ny, detector.nx
    dark = np.ones((ny, nx), dtype=np.float64) * (
        detector.dark_current * detector.exposure_time
    )
    # Readout noise
    rng_r = np.random.default_rng(noise_cfg.readout_seed)
    dark += rng_r.normal(0.0, noise_cfg.readout_sigma_adu, size=(ny, nx))
    dark = np.clip(dark, 0.0, float(detector.full_well))
    # Normalize to [0, 1] relative to full well
    return (dark / float(detector.full_well)).astype(np.float32)


# ==============================================================
# MAIN PROJECTION PIPELINE
# ==============================================================
"""
def repair_internal_open_beam_artifacts(
    img: np.ndarray,
    open_beam_frac: float = 0.995,
    object_thresh_frac: float = 0.985,
    min_component_px: int = 20,
    max_component_px: int = 50000,
    protect_background: bool = True,
) -> np.ndarray:
    
    Repairs white/open-beam artifacts inside or attached to the object footprint.

    This is intended for GVXR whole-object STL renders where overlapping shells
    cause local regions to incorrectly render as open beam.

    It does NOT affect the outside background. It only repairs bright components
    that are inside the filled object footprint.


    img = img.astype(np.float32)
    out = img.copy()

    img_max = float(np.max(img))
    if img_max <= 0:
        return out

    # Open-beam-like pixels: suspicious if they are inside the object footprint.
    open_beam_mask = img >= (open_beam_frac * img_max)

    # Object attenuation mask: pixels darker than open beam.
    object_mask = img < (object_thresh_frac * img_max)

    # Fill holes so internal white artifacts become part of object footprint.
    object_footprint = binary_fill_holes(object_mask)

    if protect_background:
        suspicious = open_beam_mask & object_footprint
    else:
        suspicious = open_beam_mask

    lbl, n = label(suspicious)

    repair_mask = np.zeros_like(suspicious, dtype=bool)

    for k in range(1, n + 1):
        comp = lbl == k
        area = int(np.sum(comp))

        if area < min_component_px:
            continue
        if area > max_component_px:
            continue

        repair_mask |= comp

    if not np.any(repair_mask):
        print("[ARTIFACT REPAIR] No internal open-beam artifacts detected.")
        return out

    # Use nearest valid non-artifact pixel to fill each bad white region.
    valid = ~repair_mask

    # distance_transform_edt returns nearest valid pixel indices when run on repair_mask.
    _, indices = distance_transform_edt(repair_mask, return_indices=True)

    repaired_values = out[indices[0], indices[1]]
    out[repair_mask] = repaired_values[repair_mask]

    print(f"[ARTIFACT REPAIR] repaired pixels: {int(np.sum(repair_mask))}")

    return out.astype(np.float32)
    """
    
def make_projection(source: SourceCord,
                     vals: SourceVals,
                     detector: Detector,
                     det_loc: Location,
                     focal: FocalSpot,
                     noise_cfg: NoiseConfig,
                     scint: ScintillatorConfig,
                     sim: SimConfig,
                     window_id: int = 0,
                     add_noise: bool = True,
                     rotation_offset_deg: float = 0.0,
                     rotation_jitter_deg=(0.0, 0.0, 0.0),
                     autocenter_attempted: bool = False,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one radiograph pair (ground_truth, noisy).

    Key fix: validate_geometry() and diagnostic_preflight() are called
    before the main render. If preflight fails, detailed diagnostics
    are printed and a RuntimeError is raised with actionable suggestions.
    """
    try:
        gvxr.destroyAllWindows()
    except Exception:
        pass
    ACTIVE_GVXR_WINDOWS.clear()
    _create_gvxr_window(window_id)
    
    for fn in ("removePolygonMeshesFromSceneGraph", "emptySceneGraph"):
        if hasattr(gvxr, fn):
            try:
                getattr(gvxr, fn)()
            except Exception:
                pass
    # Setup source and detector
    setup_source(source, vals, sim)
    setup_detector(detector, det_loc)

    # Load objects
    energy_keV = vals.mono_energy
    if vals.energy_units.lower() == "mev":
        energy_keV = vals.mono_energy * 1000.0

    mu, corrected_density = get_linear_attenuation(
        sim.material_compound,
        sim.material_density,
        energy_keV
    )
    sim.material_density = corrected_density

    if not sim.accumulate_by_attenuation:
        # Normal GVXR behavior: load all STLs into one scene
        load_objects(sim, vals, node_prefix=str(window_id))
    else:
        print("[INFO] Using per-STL attenuation accumulation mode.")
        print("[INFO] Each STL is rendered separately, converted to attenuation, then summed.")

    # Additional rotation for multi-projection series
    if abs(rotation_offset_deg) > 1e-9:
        for i in range(len(sim.stl_files)):
            gvxr.rotateNode(f"{window_id}_{i}", rotation_offset_deg, 0, 1, 0)

    # --- Geometry validation (before any render) ---
    warnings = validate_geometry(source, det_loc, sim, detector)
    for w in warnings:
        print(f"[GEOMETRY] {w}")
    if any(w.startswith("CRITICAL") for w in warnings):
        raise RuntimeError("Critical geometry error. See warnings above.")

    # --- Diagnostic preflight ---
    ok, info = diagnostic_preflight(source, vals, detector, det_loc, sim)

    if not ok:
        # Restore full resolution before raising
        setup_detector(detector, det_loc)
        raise RuntimeError(
            f"Preflight failed: {info.get('reason', 'unknown')}.\n"
            f"See diagnostic output above for suggestions.\n"
            f"Full info: {info}"
        )

    # Restore full resolution after preflight
    setup_detector(detector, det_loc)

    # --- Primary image ---
    if sim.gt_zoom_enabled:
        render_det, (cx, cy) = compute_gt_zoom_detector(
            sim, source, det_loc, detector
        )
        render_loc = Location(x=cx, y=cy, z=det_loc.z, units=det_loc.units)
        setup_detector(render_det, render_loc)
    else:
        render_det = detector
        render_loc = det_loc
    if sim.accumulate_by_attenuation:
        if focal.enabled and focal.n_samples > 1:
            img_primary = compute_focal_spot_accumulated_image(
                source, vals, render_det, render_loc, focal, sim
            )
        else:
            img_primary = compute_accumulated_stl_image(
                source, vals, render_det, render_loc, sim,
                node_prefix=f"{window_id}_primary"
            )
    else:
        if focal.enabled and focal.n_samples > 1:
            img_primary = compute_focal_spot_image(
                source, vals, render_det, render_loc, focal, sim
            )
        else:
            img_primary = compute_image(render_det)
    """
    if not sim.accumulate_by_attenuation:
        img_primary = repair_internal_open_beam_artifacts(img_primary)
    """
    print(f"  Primary image: min={img_primary.min():.2f} "
          f"max={img_primary.max():.2f} mean={img_primary.mean():.2f}")
    print(
        f"[PRIMARY] [PROJ={window_id}]"
        f"max={img_primary.max():.6f}"
        f"mean{img_primary.mean():.6f}"
    )
    """
    ok_final, final_info = validate_saved_image_coverage(img_primary, name=f"primay_projection_{window_id}",)
    print("[FINAL COVERAGE]", ok_final, final_info)
    if not ok_final:
        if not autocenter_attempted:
            user_accepts = ask_user_autocenter_object(final_info)
            if user_accepts:
                moved = move_object_to_center_and_report(
                    final_info=final_info,
                    detector=render_det,
                    source=source,
                    det_loc=render_loc,
                    sim=sim,
                )
                if sim.gt_zoom_enabled and final_info.get("reason") == "object_touches_saved_image_edge":
                    old_margin = float(sim.gt_zoom_margin_cm)
                    sim.gt_zoom_margin_cm = max(old_margin * 2.0, old_margin + 2.0, 2.0)
                    moved = True
                if moved:
                    return make_projection(
                        source=source,
                        vals=vals,
                        detector=detector,
                        det_loc=det_loc,
                        focal=focal,
                        noise_cfg=noise_cfg,
                        sim=sim,
                        window_id=window_id,
                        add_noise=add_noise,
                        rotation_offset_deg=rotation_offset_deg,
                        rotation_jitter_deg=rotation_jitter_deg,
                        autocenter_attempted=True,
                    )
        raise RuntimeError(
            f"Final saved-image coverage failed: {final_info.get('reason')}. \n"
            f"This means the image that would be saved does not contain a valid object projection.\n"
            f"Full info: {final_info}"
        )
    """
    # --- Detector effects ---
    img_scint = apply_scintillator_response(img_primary, scint, render_det)
    img_blurred    = apply_detector_blur(img_scint, render_det.blur_sigma_px)
    img_distorted  = apply_geometric_distortion(img_blurred, render_det.distortion_k1)

    # --- Noise ---
    if add_noise:
        img_noisy = apply_realistic_noise(img_distorted, noise_cfg, render_det)
    else:
        img_noisy = img_distorted.copy()

    # --- Ground truth: parallel beam, no noise, no blur ---
    original_beam = sim.beam_type
    sim.beam_type = "parallel"
    setup_source(source, vals, sim)
    setup_detector(render_det, render_loc)
    if sim.accumulate_by_attenuation:
        img_truth = compute_accumulated_stl_image(
            source, vals, render_det, render_loc, sim,
            node_prefix=f"{window_id}_truth"
        )
    else:
        img_truth = compute_image(render_det)
    sim.beam_type = original_beam
    setup_source(source,vals,sim)
    setup_detector(detector, det_loc)
    """
    if not sim.accumulate_by_attenuation:
        img_truth = repair_internal_open_beam_artifacts(img_truth)
    """
    return img_truth.astype(np.float32), img_noisy.astype(np.float32), mu, corrected_density


# ==============================================================
# SAVE OUTPUTS
# ==============================================================

def quantize_to_bit_depth(image: np.ndarray, bit_depth: int, scale_max: float) -> np.ndarray:
    """Linearly map a float image into an integer grayscale array at bit_depth.

    scale_max is the float value that maps to the integer peak
    ((2**bit_depth) - 1). Truth/noisy pairs should share the same scale_max.
    """
    bit_depth = int(bit_depth)
    if bit_depth not in (8, 16):
        raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")
    peak = (1 << bit_depth) - 1
    scale = float(scale_max) if scale_max is not None else 0.0
    if scale <= 0.0:
        scale = float(np.max(image)) if np.size(image) else 1.0
    if scale <= 0.0:
        scale = 1.0
    unit = np.clip(np.asarray(image, dtype=np.float64) / scale, 0.0, 1.0)
    quantized = np.rint(unit * peak)
    return quantized.astype(np.uint8 if bit_depth <= 8 else np.uint16)


def save_outputs(img_truth: np.ndarray,
                 img_noisy: np.ndarray,
                 flat_field: np.ndarray,
                 dark_field: np.ndarray,
                 sim: SimConfig,
                 run_tag: str,
                 source: SourceCord,
                 vals: SourceVals,
                 detector: Detector,
                 det_loc: Location,
                 focal: FocalSpot,
                 noise_cfg: NoiseConfig,
                 mu,
                 corrected_density):
    """Save PNG and/or NPY outputs along with configuration json.

    PNGs are written as true grayscale at detector.bit_depth (8 or 16).
    Truth and noisy share one linear scale so pairing stays quantitative.
    NPY keeps full float precision.
    """
    os.makedirs(sim.output_dir, exist_ok=True)
    bit_depth = int(getattr(detector, "bit_depth", 16) or 16)
    # Shared linear scale for the pair (avoid per-image percentile stretch).
    scale_max = float(max(np.max(img_truth), np.max(img_noisy), 1e-12))

    if sim.save_png:
        import imageio.v3 as imageio

        gt_png = quantize_to_bit_depth(img_truth, bit_depth, scale_max)
        noisy_png = quantize_to_bit_depth(img_noisy, bit_depth, scale_max)
        imageio.imwrite(os.path.join(sim.output_dir, f"{run_tag}_ground_truth.png"), gt_png)
        imageio.imwrite(os.path.join(sim.output_dir, f"{run_tag}_noisy.png"), noisy_png)
        if sim.save_flatfield:
            # flat/dark helpers already return roughly [0, 1]
            imageio.imwrite(
                os.path.join(sim.output_dir, f"{run_tag}_flat_field.png"),
                quantize_to_bit_depth(flat_field, bit_depth, 1.0),
            )
        if sim.save_darkfield:
            imageio.imwrite(
                os.path.join(sim.output_dir, f"{run_tag}_dark_field.png"),
                quantize_to_bit_depth(dark_field, bit_depth, 1.0),
            )

    if sim.save_npy:
        np.save(os.path.join(sim.output_dir, f"{run_tag}_ground_truth.npy"),
                img_truth)
        np.save(os.path.join(sim.output_dir, f"{run_tag}_noisy.npy"),
                img_noisy)
        if sim.save_flatfield:
            np.save(os.path.join(sim.output_dir, f"{run_tag}_flat_field.npy"),
                    flat_field)
        if sim.save_darkfield:
            np.save(os.path.join(sim.output_dir, f"{run_tag}_dark_field.npy"),
                    dark_field)
    save_config_json(source, vals, detector, det_loc, focal, noise_cfg, sim, run_tag, mu, corrected_density)
    print(f"[+] Saved: {run_tag} (png bit_depth={bit_depth})")

    
# ==============================================================
# MAIN
# ==============================================================

def main():
    args      = sys.argv[1:]
    xml_paths = [a for a in args if a.lower().endswith(".xml")]

    if not xml_paths:
        print("Usage: python stl_test.py config.xml [config2.xml ...]")
        sys.exit(1)

    for xml_path in xml_paths:
        print(f"\n=== Processing: {xml_path} ===")

        source    = SourceCord()
        vals      = SourceVals()
        detector  = Detector()
        det_loc   = Location()
        focal     = FocalSpot()
        noise_cfg = NoiseConfig()
        scint     = ScintillatorConfig()
        sim       = SimConfig()

        load_xml(xml_path, source, vals, detector, det_loc,
                 focal, noise_cfg, scint, sim)

        flat_field = generate_flat_field(detector, noise_cfg)
        dark_field = generate_dark_field(detector, noise_cfg)
        base_object_location = sim.object_location
        base_focal_seed = focal.seed
        base_poisson_seed = noise_cfg.poisson_seed
        base_readout_seed = noise_cfg.readout_seed
        base_scatter_seed = noise_cfg.scatter_seed

        for proj_i in range(sim.n_projections):
            focal.seed = None if base_focal_seed is None else base_focal_seed + proj_i
            noise_cfg.poisson_seed = None if base_poisson_seed is None else base_poisson_seed + proj_i
            noise_cfg.readout_seed = None if base_readout_seed is None else base_readout_seed + proj_i
            noise_cfg.scatter_seed = None if base_scatter_seed is None else base_scatter_seed + proj_i
            rot_offset = proj_i * sim.rotation_step_deg
            run_tag = (f"{os.path.splitext(os.path.basename(xml_path))[0]}"
                       f"_proj{proj_i:04d}")
            print(f"\n--- Projection {proj_i+1}/{sim.n_projections} ---")
            if base_object_location is not None:
                jx, jy, jz = sample_object_jitter(sim, proj_i)
                sim.object_location = (
                    base_object_location[0] + jx,
                    base_object_location[1] + jy,
                    base_object_location[2] + jz,
                )
                print(f"[JITTER] object location for proj {proj_i}: {sim.object_location}")
            else:
                sim.object_location = base_object_location
            window_id = proj_i
            rot_jitter = sample_rotation_jitter(sim, proj_i)
            if sim.rotation_jitter_enabled:
                print(f"[ROT JITER] pitch={rot_jitter[0]:.4f} degrees roll={rot_jitter[1]:.4f} degrees yaw={rot_jitter[2]:.4f} degrees")
            try:
                img_truth, img_noisy, mu, corrected_density = make_projection(
                    source, vals, detector, det_loc,
                    focal, noise_cfg, scint, sim,
                    window_id=window_id,
                    add_noise=True,
                    rotation_offset_deg=rot_offset,
                    rotation_jitter_deg=rot_jitter
                )
                save_outputs(img_truth, img_noisy,
                             flat_field, dark_field, sim, run_tag,
                             source, vals, detector,
                             det_loc, focal, noise_cfg, mu, corrected_density)
            finally:
                _destroy_gvxr_window_if_needed(window_id)
        sim.object_location = base_object_location
        base_focal_seed = focal.seed
        base_poisson_seed = base_poisson_seed
        base_readout_seed = base_readout_seed
        base_scatter_seed = base_scatter_seed


if __name__ == "__main__":
    main()