"""
Simple HTTP server for the Stellar Stream DM dashboard.

Usage:
    python dashboard/serve.py [--port 8080]

Then open http://localhost:8080 in your browser.
"""
import argparse
import http.server
import os
import sys
import webbrowser
from functools import partial
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Serve the project dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    # Serve from the dashboard directory
    dashboard_dir = Path(__file__).resolve().parent
    os.chdir(dashboard_dir)

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(dashboard_dir))
    server = http.server.HTTPServer(("localhost", args.port), handler)

    url = f"http://localhost:{args.port}"
    print(f"\n  Stellar Stream DM Dashboard")
    print(f"  Serving at: {url}")
    print(f"  Press Ctrl+C to stop\n")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
