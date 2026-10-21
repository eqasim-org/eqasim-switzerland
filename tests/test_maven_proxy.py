from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
from matsim.runtime import maven


def proxy_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    for key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'no_proxy', 'NO_PROXY'):
        monkeypatch.delenv(key, raising=False)


def test_settings_use_environment_proxy_and_preserve_user_settings(tmp_path, monkeypatch):
    proxy_environment(monkeypatch, tmp_path)
    original = tmp_path / '.m2/settings.xml'
    original.parent.mkdir()
    original.write_text('<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"><mirrors><mirror><id>mine</id></mirror></mirrors></settings>')
    snapshot = original.read_bytes()
    monkeypatch.setenv('http_proxy', 'http://proxy.service.consul:3128')
    monkeypatch.setenv('https_proxy', 'http://proxy.service.consul:3128')
    monkeypatch.setenv('no_proxy', 'localhost,.internal')
    result = Path(maven._proxy_settings(tmp_path))
    assert "<settings " in result.read_text()
    tree = ET.parse(result)
    ns = {'m': 'http://maven.apache.org/SETTINGS/1.0.0'}
    proxies = tree.findall('m:proxies/m:proxy', ns)
    assert {p.findtext('m:protocol', namespaces=ns) for p in proxies} == {'http','https'}
    for proxy in proxies:
        assert proxy.findtext('m:host', namespaces=ns) == 'proxy.service.consul'
        assert proxy.findtext('m:port', namespaces=ns) == '3128'
        assert proxy.findtext('m:nonProxyHosts', namespaces=ns) == 'localhost|*.internal'
    assert tree.findtext('m:mirrors/m:mirror/m:id', namespaces=ns) == 'mine'
    assert original.read_bytes() == snapshot
    assert result.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('enabled', [False, True])
def test_maven_launch_passes_proxy_settings_when_enabled(tmp_path, monkeypatch, enabled):
    proxy_environment(monkeypatch, tmp_path)
    monkeypatch.setenv('HTTPS_PROXY', 'http://proxy.service.consul:3128')
    monkeypatch.setenv('MAVEN_USE_ENV_PROXY', str(enabled).lower())
    calls = []
    monkeypatch.setattr(maven.shutil, 'which', lambda name: '/tools/mvn')
    monkeypatch.setattr(maven.sp, 'check_call', lambda command, cwd: calls.append((command,cwd)) or 0)
    context = SimpleNamespace(path=lambda: str(tmp_path), stage=lambda name: {'binary':'mvn','skip_tests':False})
    maven.run(context, ['-Pstandalone','package'])
    command,cwd = calls[0]
    assert ('--settings' in command) == enabled
    assert command[-2:] == ['-Pstandalone','package']
    assert cwd == str(tmp_path)
    if enabled:
        assert Path(command[command.index('--settings')+1]).is_file()


def test_no_proxy_environment_keeps_default_maven_settings(tmp_path, monkeypatch):
    proxy_environment(monkeypatch, tmp_path)
    assert maven._proxy_settings(tmp_path) is None
