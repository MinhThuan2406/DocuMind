def ocr_engines():
    from _paddle_ocr.model import PaddleOcrModel

    return {"ocr_engines": [PaddleOcrModel]}


def layout_engines():
    from _doclayout_yolo.model import DocLayoutYoloModel

    return {"layout_engines": [DocLayoutYoloModel]}


def table_structure_engines():
    from _rapid_table.model import RapidTableModel

    return {"table_structure_engines": [RapidTableModel]}
