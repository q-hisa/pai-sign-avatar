"""Render VRM face and hair on robot head poses; inputs may live outside this folder."""
import argparse
import base64
import functools
import http.server
import os
from pathlib import Path
import socketserver
import sys
import threading
from playwright.sync_api import sync_playwright


def render(args):
    root = Path(__file__).resolve().parent
    inputs = {'/inputs/model.vrm': args.model.resolve(), '/inputs/motion.json': args.motion.resolve(),
              '/inputs/headpose.json': args.pose.resolve()}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True, exist_ok=True)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *values):
            pass

        def translate_path(self, path):
            clean = path.split('?', 1)[0]
            return str(inputs[clean]) if clean in inputs else super().translate_path(path)

    with socketserver.TCPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(root))) as httpd:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                options = dict(args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                                     '--ignore-gpu-blocklist', '--use-gl=angle'])
                if args.browser:
                    options['executable_path'] = str(args.browser)
                elif sys.platform == 'darwin' and Path('/Applications/Google Chrome.app').exists():
                    options['channel'] = 'chrome'
                browser = p.chromium.launch(**options)
                try:
                    pg = browser.new_page(viewport={'width': 800, 'height': 800})
                    pg.on('pageerror', lambda e: print('pageerror:', e, flush=True))
                    pg.goto(f'http://127.0.0.1:{httpd.server_address[1]}/index.html')
                    pg.wait_for_function('window.__ready === true', timeout=60000)
                    opt = dict(scale=args.scale)
                    if args.haircut is not None:
                        opt['hairCut'] = args.haircut
                    info = pg.evaluate('(opt) => loadOverlay("/inputs/model.vrm", "/inputs/motion.json", "/inputs/headpose.json", opt)', opt)
                    print(info, flush=True)
                    for i in range(info['T']):
                        data = pg.evaluate('(i) => renderOverlay(i)', i)
                        (args.out/f'h_{i:04d}.png').write_bytes(base64.b64decode(data.split(',', 1)[1]))
                        if i % 30 == 0:
                            print('frame', i, flush=True)
                finally:
                    browser.close()
        finally:
            httpd.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'motion', 'pose', 'out']:
        p.add_argument(name, type=Path)
    p.add_argument('scale', type=float, nargs='?', default=1.4)
    p.add_argument('haircut', type=float, nargs='?', default=-0.025)
    p.add_argument('--browser', type=Path, default=os.environ.get('CHROMIUM_EXECUTABLE'))
    render(p.parse_args())
