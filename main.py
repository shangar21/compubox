import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="compubox-cli", description="CLI for live compubox punch counter")
    parser.add_argument('--path', '-p', nargs="?", default=0, type=int)
    args = parser.parse_args()


