from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_backend_image_installs_chinese_subtitle_font():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "fonts-wqy-zenhei" in dockerfile


def test_backend_image_maps_microsoft_yahei_to_chinese_font():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    fontconfig = (ROOT / "config" / "fontconfig" / "local.conf").read_text(encoding="utf-8")

    assert "config/fontconfig/local.conf" in dockerfile
    assert "/etc/fonts/local.conf" in dockerfile
    assert "Microsoft YaHei" in fontconfig
    assert "WenQuanYi Zen Hei" in fontconfig
