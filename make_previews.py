"""Stills and a GIF preview from the rendered video, for the README and the portfolio.

Run after rendering:  python make_previews.py [media/neural-spirals.mp4]
Needs PyAV and Pillow (pip install av pillow).
"""

import sys
from pathlib import Path

import av
from PIL import Image

STILLS = {"poster.png": 38.0, "lift.png": 49.0, "network.png": 30.0}  # name: seconds
GIF = ("preview.gif", 16.0, 37.0, 12, 640)  # name, start, end, fps, width


def main(video: Path) -> None:
    out = video.parent
    times = sorted(STILLS.items(), key=lambda kv: kv[1])
    gif_name, start, end, fps, width = GIF
    frames, next_gif = [], start
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            t = float(frame.pts * frame.time_base)
            if times and t >= times[0][1]:
                name, _ = times.pop(0)
                frame.to_image().save(out / name, optimize=True)
                print(f"{name}: {t:.2f} s")
            if start <= t <= end and t >= next_gif:
                img = frame.to_image()
                frames.append(img.resize((width, round(img.height * width / img.width)), Image.LANCZOS))
                next_gif += 1 / fps
            if not times and t > end:
                break
    palette = frames[len(frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    gif[0].save(out / gif_name, save_all=True, append_images=gif[1:], duration=round(1000 / fps), loop=0, optimize=True)
    print(f"{gif_name}: {len(gif)} frames, {(out / gif_name).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "media/neural-spirals.mp4"))
