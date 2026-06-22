def test_load_via_sglang():
    import sglang.srt.plugins as plugins_mod
    from sglang.srt.plugins import load_plugins
    plugins_mod._plugins_loaded = False
    load_plugins()
