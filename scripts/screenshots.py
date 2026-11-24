"""Capture the React pages without inserting predictions into user history.

Usage: python scripts/screenshots.py [--base http://127.0.0.1:5000]
Uses installed Edge in headless mode; output is docs/screens/.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:5000')
    parser.add_argument('--browser', default=os.environ.get(
        'EDGE_PATH', r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'))
    parser.add_argument('--output', type=Path, default=ROOT / 'docs' / 'screens')
    args = parser.parse_args()
    if not Path(args.browser).is_file():
        parser.error('Pass --browser with the path to Edge or Chromium.')
    with urlopen(args.base, timeout=10) as response:
        if response.status != 200:
            parser.error('Start LeafCare AI before capturing screenshots.')
    args.output.mkdir(parents=True, exist_ok=True)
    for route, name, width, height in [
        ('/', 'index_empty.png', 1440, 1100),
        ('/history', 'history.png', 1440, 1100),
        ('/diseases', 'diseases.png', 1440, 1500),
        ('/about', 'about.png', 1440, 1500),
        ('/', 'mobile.png', 390, 1200),
    ]:
        target = (args.output / name).resolve()
        with tempfile.TemporaryDirectory(prefix='leafcare_browser_') as profile:
            subprocess.run([
                args.browser, '--headless=new', '--disable-gpu', '--hide-scrollbars',
                '--no-first-run', '--no-default-browser-check',
                f'--user-data-dir={profile}', f'--window-size={width},{height}',
                '--virtual-time-budget=10000', f'--screenshot={target}',
                args.base.rstrip('/') + route,
            ], check=True, capture_output=True, timeout=60)
        print(target)


if __name__ == '__main__':
    main()
