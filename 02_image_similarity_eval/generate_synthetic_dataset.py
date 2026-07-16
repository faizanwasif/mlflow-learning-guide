from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).resolve().parent / "synthetic_dataset"
NUM_IDENTITIES = 12
IMAGES_PER_IDENTITY = 4
IMAGE_SIZE = 128


def make_identity_signature(rng: random.Random) -> dict:
    """A per-identity 'appearance' -- stands in for a person's real features."""
    return {
        "base_color": (rng.randint(40, 220), rng.randint(40, 220), rng.randint(40, 220)),
        "shape_offset": (rng.randint(-10, 10), rng.randint(-10, 10)),
        "shape_radius": rng.randint(30, 50),
    }


def render_image(signature: dict, rng: random.Random) -> Image.Image:
    """Draw one synthetic 'photo' of an identity, with small per-photo jitter."""
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)

    jitter = lambda v, spread: v + rng.randint(-spread, spread)  # noqa: E731
    color = tuple(max(0, min(255, jitter(c, 15))) for c in signature["base_color"])

    cx = IMAGE_SIZE // 2 + signature["shape_offset"][0] + rng.randint(-4, 4)
    cy = IMAGE_SIZE // 2 + signature["shape_offset"][1] + rng.randint(-4, 4)
    r = signature["shape_radius"] + rng.randint(-3, 3)

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return img


def main() -> None:
    rng = random.Random(7)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Generating {NUM_IDENTITIES} synthetic identities "
          f"x {IMAGES_PER_IDENTITY} images each in {OUTPUT_DIR}/\n")

    for person_idx in range(1, NUM_IDENTITIES + 1):
        person_dir = OUTPUT_DIR / f"person_{person_idx:02d}"
        person_dir.mkdir(exist_ok=True)

        signature = make_identity_signature(rng)
        for image_idx in range(IMAGES_PER_IDENTITY):
            img = render_image(signature, rng)
            img.save(person_dir / f"photo_{image_idx}.png")

        print(f"  person_{person_idx:02d}: {IMAGES_PER_IDENTITY} images")

    print(f"\nDone. Run evaluate_similarity_models.py next.")


if __name__ == "__main__":
    main()
