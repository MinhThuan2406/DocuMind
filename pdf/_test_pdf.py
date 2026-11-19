from __future__ import annotations

import io
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image


def _pdf_escape(text: str) -> bytes:
    out = bytearray()
    for ch in text:
        code = ord(ch)
        if ch in ("\\", "(", ")"):
            out.extend(b"\\" + ch.encode("ascii"))
        elif 32 <= code <= 126:
            out.append(code)
        else:
            out.append(ord("?"))
    return bytes(out)


def _dash_prefix(dash: tuple[float, float] | None) -> str:
    if dash is None:
        return ""
    return f"[{float(dash[0]):.2f} {float(dash[1]):.2f}] 0 d "


def _paint_op(fill: bool, stroke: bool) -> str:
    if fill and stroke:
        return "B"
    if fill:
        return "f"
    if stroke:
        return "S"
    return "n"


class _TestPage:
    def __init__(self, width: float, height: float) -> None:
        self.width = float(width)
        self.height = float(height)
        self._ops: list[bytes] = []
        self._images: list[tuple[str, int, int, bytes]] = []

    def insert_text(
        self,
        pos: tuple[float, float],
        text: str,
        *,
        fontsize: float = 11.0,
        bold: bool = False,
    ) -> None:
        font = b"/F2" if bold else b"/F1"
        x, y_top = float(pos[0]), float(pos[1])
        line_step = float(fontsize) * 1.2
        for idx, line in enumerate(text.splitlines() or [""]):
            y_pdf = self.height - (y_top + idx * line_step)
            esc = _pdf_escape(line)
            self._ops.append(
                b"BT "
                + font
                + f" {fontsize:.2f} Tf 1 0 0 1 {x:.2f} {y_pdf:.2f} Tm ({esc.decode('ascii')}) Tj ET".encode("ascii")
            )

    def draw_line(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        *,
        width: float = 0.5,
        dash: tuple[float, float] | None = None,
    ) -> None:
        y1 = self.height - float(p1[1])
        y2 = self.height - float(p2[1])
        dash_str = _dash_prefix(dash)
        self._ops.append(
            f"q {width:.2f} w {dash_str}{float(p1[0]):.2f} {y1:.2f} m {float(p2[0]):.2f} {y2:.2f} l S Q".encode("ascii")
        )

    def draw_rect(
        self,
        bbox: tuple[float, float, float, float],
        *,
        width: float = 1.0,
        dash: tuple[float, float] | None = None,
        fill: bool = False,
        stroke: bool = True,
    ) -> None:
        x0, y0, x1, y1 = (float(v) for v in bbox)
        w = x1 - x0
        h = y1 - y0
        y_pdf = self.height - y1
        dash_str = _dash_prefix(dash)
        paint = _paint_op(fill, stroke)
        self._ops.append(
            f"q {width:.2f} w {dash_str}{x0:.2f} {y_pdf:.2f} {w:.2f} {h:.2f} re {paint} Q".encode("ascii")
        )

    def draw_polygon(
        self,
        points: Sequence[tuple[float, float]],
        width: float = 1.0,
        fill: bool = False,
        stroke: bool = True,
        dash: tuple[float, float] | None = None,
    ) -> None:
        if len(points) < 2:
            return
        x0, y0 = float(points[0][0]), self.height - float(points[0][1])
        segs = [f"{x0:.2f} {y0:.2f} m"]
        for px, py in points[1:]:
            segs.append(f"{float(px):.2f} {self.height - float(py):.2f} l")
        segs.append("h")
        dash_str = _dash_prefix(dash)
        paint = _paint_op(fill, stroke)
        self._ops.append(f"q {width:.2f} w {dash_str}{' '.join(segs)} {paint} Q".encode("ascii"))

    def insert_image(
        self,
        bbox: tuple[float, float, float, float],
        *,
        stream: bytes | None = None,
        filename: str | Path | None = None,
    ) -> None:
        raw = Path(filename).read_bytes() if filename is not None else (stream or b"")
        with Image.open(io.BytesIO(raw)) as im:
            rgb = im.convert("RGB")
            iw, ih = rgb.size
            compressed = zlib.compress(rgb.tobytes())
        name = f"/Im{len(self._images) + 1}"
        self._images.append((name, iw, ih, compressed))
        x0, y0, x1, y1 = (float(v) for v in bbox)
        w = x1 - x0
        h = y1 - y0
        y_pdf = self.height - y1
        self._ops.append(f"q {w:.2f} 0 0 {h:.2f} {x0:.2f} {y_pdf:.2f} cm {name} Do Q".encode("ascii"))


class TestPdfBuilder:
    def __init__(self) -> None:
        self.pages: list[_TestPage] = []

    def new_page(self, width: float = 595.0, height: float = 842.0) -> _TestPage:
        p = _TestPage(width, height)
        self.pages.append(p)
        return p

    def to_bytes(self) -> bytes:
        objs: list[bytes] = []

        def add(body: bytes) -> int:
            objs.append(body)
            return len(objs)

        catalog_id = add(b"")
        pages_id = add(b"")
        f1_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        f2_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

        page_ids: list[int] = []
        for pg in self.pages:
            xobj_entries: list[str] = []
            for name, iw, ih, comp in pg._images:
                img_hdr = (
                    f"<< /Type /XObject /Subtype /Image /Width {iw} /Height {ih} "
                    f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
                    f"/Length {len(comp)} >>\nstream\n"
                ).encode("ascii")
                img_id = add(img_hdr + comp + b"\nendstream")
                xobj_entries.append(f"{name} {img_id} 0 R")

            stream_data = b"\n".join(pg._ops)
            content_id = add(
                f"<< /Length {len(stream_data)} >>\nstream\n".encode("ascii") + stream_data + b"\nendstream"
            )
            xobj_part = f" /XObject << {' '.join(xobj_entries)} >>" if xobj_entries else ""
            res = f"<< /Font << /F1 {f1_id} 0 R /F2 {f2_id} 0 R >>{xobj_part} >>"
            page_id = add(
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {pg.width:.2f} {pg.height:.2f}] "
                f"/Resources {res} /Contents {content_id} 0 R >>".encode("ascii")
            )
            page_ids.append(page_id)

        objs[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii")
        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        objs[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

        buf = bytearray(b"%PDF-1.4\n")
        offsets: list[int] = [0]
        for idx, body in enumerate(objs, start=1):
            offsets.append(len(buf))
            buf.extend(f"{idx} 0 obj\n".encode("ascii"))
            buf.extend(body)
            buf.extend(b"\nendobj\n")

        xref_pos = len(buf)
        buf.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii"))
        for off in offsets[1:]:
            buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        buf.extend(
            f"trailer\n<< /Size {len(objs) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
                "ascii"
            )
        )
        return bytes(buf)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_bytes(self.to_bytes())
        return out
