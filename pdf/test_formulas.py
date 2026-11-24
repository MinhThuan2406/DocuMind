import io
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pypdfium2 as pdfium  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import pdf.ir as ir_mod  # noqa: E402
from pdf._test_pdf import TestPdfBuilder  # noqa: E402
from pdf.backends import get, registered, reset_cache  # noqa: E402
from pdf.backends.formula_onnx import RapidLatexOcrBackend, UnimernetOnnxBackend, _preprocess_unimernet  # noqa: E402
from pdf.backends.registry import VALID_KINDS  # noqa: E402
from pdf.formulas import ParsedFormula, extract_formulas_on_page, formula_to_ir_dict, pad_formula_crop  # noqa: E402
from pdf.models import RasterPage, SemanticRegion, TextBlock  # noqa: E402


class _MockFormulaExtractor:
    name = "mock-formula"
    source = "mock_formula"

    def __init__(self, responses):
        self._responses = list(responses)
        self.crops: list[Image.Image] = []

    def extract_formula(self, crop: Image.Image) -> tuple[str, float]:
        self.crops.append(crop.copy())
        if not self._responses:
            return ("", 0.0)
        return self._responses.pop(0)


def test_formula_registry_and_lazy_imports():
    assert "formula" in VALID_KINDS
    assert {"auto", "unimernet-onnx", "rapid-latex-ocr"} <= set(registered("formula"))

    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys, pdf, pdf.formulas; "
                "from pdf.backends import get; "
                "b_auto = get('formula', 'auto'); "
                "b_uni = get('formula', 'unimernet-onnx'); "
                "b_rap = get('formula', 'rapid-latex-ocr'); "
                "assert type(b_auto).__name__ == 'UnimernetOnnxBackend'; "
                "assert type(b_uni).__name__ == 'UnimernetOnnxBackend'; "
                "assert type(b_rap).__name__ == 'RapidLatexOcrBackend'; "
                "bad = [m for m in ('torch', 'transformers', 'docling', 'onnxruntime', "
                "'tokenizers', 'huggingface_hub', 'rapid_latex_ocr') if m in sys.modules]; "
                "assert not bad, bad"
            ),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0


def test_unimernet_onnx_preprocessing_and_kv_decode():
    wide = Image.new("RGB", (200, 50), (0, 0, 0))
    arr = _preprocess_unimernet(wide)
    assert arr.shape == (1, 3, 192, 672)
    assert arr.dtype == np.float32
    expected_white_r = (1.0 - 0.485) / 0.229
    assert abs(float(arr[0, 0, 0, 0]) - expected_white_r) < 1e-4
    expected_black_r = (0.0 - 0.485) / 0.229
    assert abs(float(arr[0, 0, 96, 336]) - expected_black_r) < 1e-4

    import onnxruntime as ort
    from tokenizers import Tokenizer

    created_sessions: list[tuple[str, list[str]]] = []
    step_calls: list[dict[str, object]] = []

    class _FakeIO:
        def __init__(self, name: str):
            self.name = name

    class _FakeSession:
        def __init__(self, path: str, providers: list[str] | None = None, **_kw):
            self.path = Path(path).name
            created_sessions.append((self.path, list(providers or [])))
            self.vocab_size = 8

        def get_inputs(self):
            if self.path == "encoder_model_quantized.onnx":
                return [_FakeIO("pixel_values")]
            if self.path == "decoder_model_quantized.onnx":
                return [_FakeIO("input_ids"), _FakeIO("encoder_hidden_states")]
            return [
                _FakeIO("input_ids"),
                _FakeIO("past_key_values.0.decoder.key"),
                _FakeIO("past_key_values.0.decoder.value"),
                _FakeIO("past_key_values.0.encoder.key"),
                _FakeIO("past_key_values.0.encoder.value"),
            ]

        def get_outputs(self):
            if self.path == "encoder_model_quantized.onnx":
                return [_FakeIO("last_hidden_state")]
            if self.path == "decoder_model_quantized.onnx":
                return [
                    _FakeIO("logits"),
                    _FakeIO("present.0.decoder.key"),
                    _FakeIO("present.0.decoder.value"),
                    _FakeIO("present.0.encoder.key"),
                    _FakeIO("present.0.encoder.value"),
                ]
            return [
                _FakeIO("logits"),
                _FakeIO("present.0.decoder.key"),
                _FakeIO("present.0.decoder.value"),
            ]

        def run(self, _out_names, feed: dict[str, np.ndarray]):
            if self.path == "encoder_model_quantized.onnx":
                assert feed["pixel_values"].shape == (1, 3, 192, 672)
                return [np.ones((1, 4, 8), dtype=np.float32)]

            def _logits_for(tok_id: int) -> np.ndarray:
                lg = np.full((1, 1, self.vocab_size), -10.0, dtype=np.float32)
                lg[0, 0, tok_id] = 10.0
                return lg

            if self.path == "decoder_model_quantized.onnx":
                assert feed["input_ids"].tolist() == [[0]]
                assert "encoder_hidden_states" in feed
                enc_k = np.full((1, 1, 4, 2), 7.0, dtype=np.float32)
                enc_v = np.full((1, 1, 4, 2), 9.0, dtype=np.float32)
                dec_k = np.full((1, 1, 1, 2), 1.0, dtype=np.float32)
                dec_v = np.full((1, 1, 1, 2), 1.0, dtype=np.float32)
                next_tok = 2 if _FakeSession.emit_empty else 3
                return [_logits_for(next_tok), dec_k, dec_v, enc_k, enc_v]

            step_calls.append({
                "input_ids": feed["input_ids"].tolist(),
                "enc_k_val": float(feed["past_key_values.0.encoder.key"][0, 0, 0, 0]),
                "dec_k_len": int(feed["past_key_values.0.decoder.key"].shape[2]),
            })
            cur = int(feed["input_ids"][0, 0])
            nxt = 4 if cur == 3 else (5 if cur == 4 else 2)
            new_len = int(feed["past_key_values.0.decoder.key"].shape[2]) + 1
            return [
                _logits_for(nxt),
                np.full((1, 1, new_len, 2), float(new_len), dtype=np.float32),
                np.full((1, 1, new_len, 2), float(new_len), dtype=np.float32),
            ]

    _FakeSession.emit_empty = False

    class _FakeTokenizer:
        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            assert skip_special_tokens is True
            vocab = {3: "x", 4: "^", 5: "2"}
            return "".join(vocab[i] for i in ids)

    orig_session = ort.InferenceSession
    orig_from_file = Tokenizer.from_file
    ort.InferenceSession = _FakeSession  # type: ignore[assignment]
    Tokenizer.from_file = staticmethod(lambda _path: _FakeTokenizer())  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            small_dir = Path(tmp) / "small"
            small_dir.mkdir()
            for fname in (
                "encoder_model_quantized.onnx",
                "decoder_model_quantized.onnx",
                "decoder_with_past_model_quantized.onnx",
                "tokenizer.json",
            ):
                (small_dir / fname).write_bytes(b"stub")

            backend = UnimernetOnnxBackend(device="auto", model_path=tmp)
            assert backend.ensure_loaded() is True
            assert backend.ensure_loaded() is False
            assert all(provs == ["CPUExecutionProvider"] for _, provs in created_sessions)

            latex, conf = backend.extract_formula(wide)
            assert latex == "x^2"
            assert conf >= 0.70
            assert step_calls == [
                {"input_ids": [[3]], "enc_k_val": 7.0, "dec_k_len": 1},
                {"input_ids": [[4]], "enc_k_val": 7.0, "dec_k_len": 2},
                {"input_ids": [[5]], "enc_k_val": 7.0, "dec_k_len": 3},
            ]

            _FakeSession.emit_empty = True
            empty_latex, empty_conf = backend.extract_formula(wide)
            assert empty_latex == "" and empty_conf == 0.0

            created_sessions.clear()
            gpu_backend = UnimernetOnnxBackend(device="cuda", model_path=tmp)
            gpu_backend.ensure_loaded()
            assert all(provs == ["CUDAExecutionProvider", "CPUExecutionProvider"] for _, provs in created_sessions)
    finally:
        ort.InferenceSession = orig_session  # type: ignore[assignment]
        Tokenizer.from_file = orig_from_file  # type: ignore[assignment]


def test_rapid_latex_ocr_optional_backend_lazy():
    reset_cache()
    orig_mod = sys.modules.get("rapid_latex_ocr")
    try:
        sys.modules["rapid_latex_ocr"] = None  # type: ignore[assignment]
        backend = RapidLatexOcrBackend()
        try:
            backend.ensure_loaded()
            raise AssertionError("expected ImportError when rapid_latex_ocr is missing")
        except ImportError:
            pass

        calls: list[dict[str, str]] = []

        class _FakeLatexOCR:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.mode = "tuple"

            def __call__(self, img):
                assert isinstance(img, Image.Image)
                if self.mode == "tuple":
                    return (r"\frac{a}{b}", 0.04)
                return r"x^2 + y^2"

        fake_mod = types.ModuleType("rapid_latex_ocr")
        fake_mod.LatexOCR = _FakeLatexOCR  # type: ignore[attr-defined]
        sys.modules["rapid_latex_ocr"] = fake_mod

        with tempfile.TemporaryDirectory() as tmp:
            wdir = Path(tmp)
            for fname in ("image_resizer.onnx", "encoder.onnx", "decoder.onnx", "tokenizer.json"):
                (wdir / fname).write_bytes(b"stub")
            injected = RapidLatexOcrBackend(model_path=wdir)
            assert injected.ensure_loaded() is True
            assert injected.ensure_loaded() is False
            assert len(calls) == 1
            assert Path(calls[0]["encoder_path"]).name == "encoder.onnx"

            crop = Image.new("RGB", (40, 20), (255, 255, 255))
            latex1, conf1 = injected.extract_formula(crop)
            assert latex1 == r"\frac{a}{b}"
            assert conf1 >= 0.70

            injected._model.mode = "str"
            latex2, conf2 = injected.extract_formula(crop)
            assert latex2 == r"x^2 + y^2"
            assert conf2 >= 0.70
    finally:
        if orig_mod is None:
            sys.modules.pop("rapid_latex_ocr", None)
        else:
            sys.modules["rapid_latex_ocr"] = orig_mod
        reset_cache()


def test_white_pad_crop_and_extract_on_pdfium_and_raster():
    small = Image.new("RGB", (50, 20), (10, 20, 30))
    p_small = pad_formula_crop(small, pad_frac=0.04)
    assert p_small.size == (50 + 16, 20 + 16)
    assert p_small.getpixel((0, 0)) == (255, 255, 255)
    assert p_small.getpixel((8, 8)) == (10, 20, 30)

    mid = Image.new("RGB", (500, 100), (0, 0, 0))
    p_mid = pad_formula_crop(mid, pad_frac=0.04)
    assert p_mid.size == (500 + 40, 100 + 40)

    large = Image.new("RGB", (2000, 100), (0, 0, 0))
    p_large = pad_formula_crop(large, pad_frac=0.04)
    assert p_large.size == (2000 + 96, 100 + 96)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "rot.pdf"
        b = TestPdfBuilder()
        pg = b.new_page(width=400, height=300)
        pg.insert_text((80, 100), "E = mc^2", fontsize=16)
        b.save(pdf_path)

        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            doc[0].set_rotation(90)
            doc.save(pdf_path)
        finally:
            doc.close()

        doc2 = pdfium.PdfDocument(str(pdf_path))
        try:
            ext = _MockFormulaExtractor([(r"E = mc^2", 0.94)])
            parsed = extract_formulas_on_page(doc2[0], [(20.0, 20.0, 220.0, 160.0)], extractor=ext, dpi=150)
            assert len(parsed) == 1
            assert parsed[0] == ParsedFormula(
                latex=r"E = mc^2",
                confidence=0.94,
                source="mock_formula",
                backend="mock-formula",
            )
            assert len(ext.crops) == 1
            assert ext.crops[0].getpixel((0, 0)) == (255, 255, 255)
        finally:
            doc2.close()

    raster_img = Image.new("RGB", (400, 200), (255, 255, 255))
    draw = ImageDraw.Draw(raster_img)
    draw.rectangle((40, 40, 160, 80), fill=(0, 0, 0))
    fake_raster = SimpleNamespace(image=raster_img, page_box=(0.0, 0.0, 200.0, 100.0))
    ext_r = _MockFormulaExtractor([(r"\alpha + \beta", 0.91)])
    parsed_r = extract_formulas_on_page(
        None,
        [(20.0, 20.0, 80.0, 40.0)],
        extractor=ext_r,
        raster_page=fake_raster,
    )
    assert len(parsed_r) == 1 and parsed_r[0].latex == r"\alpha + \beta"
    assert ext_r.crops[0].getpixel((0, 0)) == (255, 255, 255)
    assert ext_r.crops[0].getpixel((ext_r.crops[0].width // 2, ext_r.crops[0].height // 2)) == (0, 0, 0)

    ir_dict = formula_to_ir_dict("fml_0001", "elem_0002", 1, "sample.pdf", (10.1234, 20.0, 50.0, 40.0), parsed_r[0])
    assert ir_dict == {
        "id": "fml_0001",
        "element_id": "elem_0002",
        "page": 1,
        "source_file": "sample.pdf",
        "bbox": [10.123, 20.0, 50.0, 40.0],
        "source": "mock_formula",
        "backend": "mock-formula",
        "confidence": 0.91,
        "latex": r"\alpha + \beta",
    }


def test_formulas_false_zero_overhead():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "plain.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=400, height=300)
        p.insert_text((50, 80), "Plain native text paragraph.", fontsize=12)
        b.save(pdf_path)

        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; from pdf import extract_ir; "
                    f"ir = extract_ir({str(pdf_path)!r}, figures=False, ocr=False, formulas=False); "
                    "assert ir['formulas'] == []; "
                    "bad = [m for m in ('pdf.formulas', 'pdf.backends.formula_onnx', "
                    "'rapid_latex_ocr', 'onnxruntime', 'tokenizers', 'huggingface_hub', "
                    "'torch', 'transformers', 'docling') if m in sys.modules]; "
                    "assert not bad, bad"
                ),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert proc.returncode == 0


def test_extract_ir_native_semantic_regions_formula_structure_off_and_on():
    import pdf.figures as figures_mod

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "native_fml.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=500, height=400)
        p.insert_text((60, 80), "Introduction", fontsize=14)
        p.insert_text((60, 150), "E = mc^2", fontsize=12)
        p.insert_text((60, 220), "After formula prose.", fontsize=11)
        b.save(pdf_path)

        raw_ir = ir_mod.extract_ir(pdf_path, figures=False, ocr=False, formulas=False)
        fml_elem_raw = next(e for e in raw_ir["elements"] if "E = mc^2" in e["text"])
        fx0, fy0, fx1, fy1 = fml_elem_raw["bbox"]
        sem_box = (fx0 - 4.0, fy0 - 4.0, fx1 + 4.0, fy1 + 4.0)

        def _fake_detect(_p, pages, **kw):
            sem_out = kw.get("semantic_regions_out")
            assert sem_out is not None
            if 1 in pages:
                sem_out[1] = [SemanticRegion(sem_box, "formula", 0.93)]
            return {}

        orig_detect = figures_mod.detect_figures
        figures_mod.detect_figures = _fake_detect
        try:
            for struct_flag in (False, True):
                ext = _MockFormulaExtractor([(r"E = mc^{2}", 0.96)])
                ir = ir_mod.extract_ir(
                    pdf_path,
                    figures=False,
                    ocr=False,
                    formulas=True,
                    formula_backend=ext,
                    structure=struct_flag,
                )
                assert json.loads(json.dumps(ir)) == ir
                assert len(ir["formulas"]) == 1
                fml = ir["formulas"][0]
                assert fml["id"] == "fml_0001"
                assert fml["latex"] == r"E = mc^{2}"
                assert fml["backend"] == "mock-formula"
                assert fml["source"] == "mock_formula"
                assert fml["confidence"] == 0.96
                elem = next(e for e in ir["elements"] if e["id"] == fml["element_id"])
                assert elem["metadata"]["layout_label"] == "formula"
                assert elem["metadata"]["formula_id"] == "fml_0001"
                assert elem["metadata"]["latex"] == r"E = mc^{2}"
        finally:
            figures_mod.detect_figures = orig_detect


def test_extract_ir_raster_formula_and_empty_warning():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "raster_fml.pdf"
        img = Image.new("RGB", (400, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle((40, 40, 180, 90), fill=(0, 0, 0))
        draw.rectangle((40, 140, 180, 190), fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b = TestPdfBuilder()
        p = b.new_page(width=400, height=300)
        p.insert_image((0, 0, 400, 300), stream=buf.getvalue())
        b.save(pdf_path)

        class _FakeRasterReader:
            name = "fake-raster"
            last_stats = None

            def read(self, _path, pages, **_kw):
                return {
                    pno: RasterPage(
                        blocks=(
                            TextBlock("a^2+b^2=c^2", (40.0, 40.0, 180.0, 90.0), 12.0, False, "ocr", 0.92, label="formula"),
                            TextBlock("???", (40.0, 140.0, 180.0, 190.0), 12.0, False, "ocr", 0.88, label="formula"),
                        ),
                        figures=(),
                        reading_order=(0, 1),
                    )
                    for pno in pages
                }

        ext = _MockFormulaExtractor([(r"a^2 + b^2 = c^2", 0.95), ("   ", 0.0)])
        ir = ir_mod.extract_ir(
            pdf_path,
            figures=False,
            ocr=True,
            reader=_FakeRasterReader(),
            formulas=True,
            formula_backend=ext,
        )
        assert json.loads(json.dumps(ir)) == ir
        assert len(ir["formulas"]) == 1
        fml = ir["formulas"][0]
        assert fml["id"] == "fml_0001"
        assert fml["latex"] == r"a^2 + b^2 = c^2"

        fml_elems = [e for e in ir["elements"] if e["metadata"].get("layout_label") == "formula"]
        assert len(fml_elems) == 2
        assert fml_elems[0]["metadata"]["formula_id"] == "fml_0001"
        assert fml_elems[0]["metadata"]["latex"] == r"a^2 + b^2 = c^2"
        assert "formula_id" not in fml_elems[1]["metadata"]
        assert "latex" not in fml_elems[1]["metadata"]

        empty_warns = [w for w in ir["metadata"]["warnings"] if w["code"] == "formula_parse_empty"]
        assert len(empty_warns) == 1
        assert empty_warns[0] == {
            "page": 1,
            "code": "formula_parse_empty",
            "severity": "warning",
            "reason": "formula_extractor_returned_empty_latex",
            "fallback": None,
        }


def test_extract_ir_multiline_formula_soft_merge_single_target():
    import pdf.figures as figures_mod

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "multi_fml.pdf"
        b = TestPdfBuilder()
        p = b.new_page(width=500, height=400)
        p.insert_text((60, 100), "a = b + c", fontsize=12)
        p.insert_text((60, 130), "d = e + f", fontsize=12)
        b.save(pdf_path)

        raw_ir = ir_mod.extract_ir(pdf_path, figures=False, ocr=False, formulas=False)
        fml_boxes = [e["bbox"] for e in raw_ir["elements"]]
        assert len(fml_boxes) == 2
        sem_box = (
            min(bx[0] for bx in fml_boxes) - 4.0,
            min(bx[1] for bx in fml_boxes) - 4.0,
            max(bx[2] for bx in fml_boxes) + 4.0,
            max(bx[3] for bx in fml_boxes) + 4.0,
        )

        def _fake_detect(_p, pages, **kw):
            sem_out = kw.get("semantic_regions_out")
            if sem_out is not None and 1 in pages:
                sem_out[1] = [SemanticRegion(sem_box, "formula", 0.94)]
            return {}

        orig_detect = figures_mod.detect_figures
        figures_mod.detect_figures = _fake_detect
        try:
            ext = _MockFormulaExtractor([(r"a = b + c \\ d = e + f", 0.95)])
            ir = ir_mod.extract_ir(
                pdf_path,
                figures=False,
                ocr=False,
                formulas=True,
                formula_backend=ext,
                semantic_layout="auto",
            )
        finally:
            figures_mod.detect_figures = orig_detect

        assert len(ir["formulas"]) == 1
        assert len(ir["elements"]) == 1
        assert ir["formulas"][0]["element_id"] == ir["elements"][0]["id"]
        assert ir["elements"][0]["text"] == "a = b + c\nd = e + f"
        assert ir["elements"][0]["metadata"]["layout_label"] == "formula"


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
