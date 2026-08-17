import argparse
from pathlib import Path

from .builder import create_dashboard_app


def main():
    parser = argparse.ArgumentParser(description="Serve the simulation dashboard.")
    parser.add_argument(
        "--directory",
        default=".",
        help="Directory containing the MATSim output folder or the simulation output root.",
    )
    parser.add_argument("--port", type=int, default=8050, help="Port to serve the dashboard on.")
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    app = create_dashboard_app(root)
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
