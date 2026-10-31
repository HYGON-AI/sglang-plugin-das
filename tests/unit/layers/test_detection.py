from sglang_hcu_plugin.utils.detection import get_gcn_arch, is_hcu, is_hcu_native_fp8_supported

def test_detection_callable():
    assert isinstance(is_hcu(), bool)
    assert isinstance(is_hcu_native_fp8_supported(), bool)
    assert isinstance(get_gcn_arch(), str)
