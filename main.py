from livefeed import feed
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="compubox-cli", description="CLI for live compubox punch counter")
    parser.add_argument('--tmp-path', nargs="?", const=".compubox/tmp", type=str)
    parser.add_argument('--camera-id', nargs="?", const=0, type=int)
    feed.start_stream()
