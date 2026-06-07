import plistlib
from pathlib import Path


def test_plist_is_valid_and_keepalive():
    path = Path("deploy/launchd/com.quant.intraday-live.plist")
    data = plistlib.loads(path.read_bytes())
    assert data["Label"] == "com.quant.intraday-live"
    assert data["KeepAlive"] is True
    assert any("intraday" in str(a) and "live" in str(a) for a in data["ProgramArguments"])
