from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_one_click_demo_launcher_waits_for_dependency_ready():
    script = (ROOT / "scripts" / "open_demo.ps1").read_text(encoding="utf-8")
    shortcut = (ROOT / "open_demo.cmd").read_text(encoding="utf-8")
    assert "/health/ready" in script
    assert "Start-Process $url" in script
    assert "scripts\\open_demo.ps1" in shortcut
