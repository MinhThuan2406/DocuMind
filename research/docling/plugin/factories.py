def layout_engines():
    from _doclayout_yolo.model import DocLayoutYoloModel
    from _pp_doclayout_v3.model import PPDocLayoutV3Model

    return {
        "layout_engines": [
            DocLayoutYoloModel,
            PPDocLayoutV3Model,
        ]
    }


def table_structure_engines():
    from _rapid_table.model import RapidTableModel

    return {"table_structure_engines": [RapidTableModel]}
