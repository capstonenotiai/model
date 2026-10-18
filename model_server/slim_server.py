"""Slim-v9 service entry point."""
import argparse
import os

from model_server.runtime import ROOT
from model_server.server import make_server
from model_server.slim_v1 import SlimRuntime


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-mode', choices=['slim-v9'], required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--interview-type', choices=['map-to-event', 'passthrough'], default='map-to-event')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    token = os.environ.get('MODEL_API_TOKEN') or None
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not token:
        parser.error('--host outside loopback requires MODEL_API_TOKEN')
    if token and len(token) < 32:
        parser.error('MODEL_API_TOKEN must contain at least 32 characters')
    runtime = SlimRuntime(args.model_dir, prompt_mode=args.prompt_mode, interview_type=args.interview_type)
    server = make_server(runtime, args.port, args.host, token)
    label = 'slim-v9' if args.prompt_mode == 'slim-v9' else 'Slim-v1'
    print(f'{label} ready on http://{args.host}:{server.server_port}; token_required={bool(token)}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
