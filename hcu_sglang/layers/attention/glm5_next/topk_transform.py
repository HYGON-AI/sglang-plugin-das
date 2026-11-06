from enum import IntEnum, auto


class TopkTransformMethod(IntEnum):
    # Transform topk indices to indices to the page table (page_size = 1)
    PAGED = auto()
    # Transform topk indices to indices to ragged kv (non-paged)
    RAGGED = auto()
