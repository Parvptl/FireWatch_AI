"""Drawing detections on frames and writing key frames / annotated video (OpenCV, lazy import)."""

from __future__ import annotations

from pathlib import Path

COLOURS = {"fire": (0, 69, 255), "smoke": (200, 200, 200)}  # BGR


def draw(image, detections, time_s: float | None = None):
    import cv2

    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
        colour = COLOURS.get(d.class_name, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, max(th, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    if time_s is not None:
        stamp = f"t={time_s:8.2f}s"
        cv2.putText(
            out,
            stamp,
            (8, out.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def write_jpeg(path: Path, image, max_width: int = 960) -> None:
    import cv2

    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    tmp = path.with_suffix(".tmp.jpg")
    if not cv2.imwrite(str(tmp), image, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise OSError(f"could not write {path}")
    tmp.replace(path)


class AnnotatedVideoWriter:
    """MP4 of the analysed (sampled) frames with boxes drawn. Tries H.264, falls back to MPEG-4."""

    def __init__(self, path: Path, fps: float, width: int, height: int, max_width: int = 1280):
        import cv2

        scale = min(1.0, max_width / width)
        self.size = (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2)
        self.path = path
        self.codec = None
        for fourcc in ("avc1", "mp4v"):
            w = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*fourcc), max(fps, 1.0), self.size
            )
            if w.isOpened():
                self._w, self.codec = w, fourcc
                break
            w.release()
        if self.codec is None:
            raise OSError("no usable MP4 encoder in this OpenCV build")

    def write(self, image) -> None:
        import cv2

        if (image.shape[1], image.shape[0]) != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        self._w.write(image)

    def close(self) -> None:
        self._w.release()
