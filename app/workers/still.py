from core.bus.handlers import job_pic_still
from core.bus.cli import base_parser
from core.bus.worker_loop import run_worker_loop

KIND = "pic_still"


def main():
    parser = base_parser("Still pic worker (modes 1/2, mode 3 PNG)")
    args = parser.parse_args()
    run_worker_loop(KIND, lambda payload: job_pic_still(payload, None), args.db, args.poll)


if __name__ == "__main__":
    main()
