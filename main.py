from livefeed import feed
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="compubox-cli", description="CLI for live compubox punch counter")
    parser.add_argument('--tmp-path', '-t', nargs="?", default="./compubox/tmp", type=str)
    parser.add_argument('--cam-id', '-c', nargs="?", default=0, type=int)
    args = parser.parse_args()
    feed.start_stream(tmp_path=args.tmp_path, cam_id=args.cam_id)
