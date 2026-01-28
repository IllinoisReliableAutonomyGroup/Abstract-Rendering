import os
import torch
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, Union

def _to_float_tensor(x):
    """Convert numpy / torch input to a CPU float32 tensor."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32, copy=False))
    return x.to(dtype=torch.float32).detach().cpu()

def save_abstract_record(
    save_dir,
    index,
    lower_input,
    upper_input,
    lower_img,
    upper_img,
    point=None,
    direction=None,
    radius=None,
):
    """
    Save an abstract image record.

    Required fields:
        xl, xu        : input lower/upper bounds
        lower, upper  : image lower/upper bounds (H, W, 3), float32 in [0, 1]
        lA, uA        : placeholder (None)
        lb, ub        : placeholder (None)

    Optional:
        point, direction, radius
    """
    # --- Normalize inputs to float32 CPU tensors
    lower_input = _to_float_tensor(lower_input)
    upper_input = _to_float_tensor(upper_input)
    lower_img   = _to_float_tensor(lower_img)
    upper_img   = _to_float_tensor(upper_img)
    point       = _to_float_tensor(point)
    direction   = _to_float_tensor(direction)
    radius      = _to_float_tensor(radius)

    # --- Construct record
    record = {
        "xl": lower_input,
        "xu": upper_input,
        "lower": lower_img,
        "upper": upper_img,
        "lA": None,
        "uA": None,
        "lb": None,
        "ub": None,
        "point": point,
        "direction": direction,
        "radius": radius,
    }

    # --- Save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"abstract_{index:06d}.pt")
    torch.save(record, out_path)

    return out_path

def iter_abstract_records(folder: Union[str, Path]) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    """
    Yield (path, record) for each abstract_*.pt in a folder, sorted by filename.
    """
    folder = Path(folder)
    for p in sorted(folder.glob("abstract_*.pt")):
        yield p, load_abstract_record(p)

def load_abstract_record(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a saved abstract image record (.pt) and return a dict with keys:
      lower, upper, lA, uA, lb, ub, xl, xu, point, direction, radius

    This matches the format produced by abstract_gsplat.py.
    """
    path = Path(path)
    rec = torch.load(path, map_location="cpu")

    required_keys = ["xl", "xu", "lower", "upper", "lA", "uA", "lb", "ub", "point", "direction", "radius"]
    for k in required_keys:
        if k not in rec:
            raise KeyError(f"Missing key '{k}' in abstract record {path}: {k}")

    return rec
