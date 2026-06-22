from sglang_dcu_plugin.utils.detection import get_gcn_arch, is_dcu, is_dcu_native_fp8_supported

def test_detection_callable():
    assert isinstance(is_dcu(), bool)
    assert isinstance(is_dcu_native_fp8_supported(), bool)
    assert isinstance(get_gcn_arch(), str)
