from core.bus.handlers import job_pic_video
from core.bus.cli import base_parser
from core.bus.worker_loop import run_worker_loop

KIND = "pic_video"


def main():
    parser = base_parser("Video/GIF pic worker (mode 3 animated)")
    args = parser.parse_args()
    run_worker_loop(KIND, lambda payload: job_pic_video(payload, None), args.db, args.poll)


if __name__ == "__main__":
    main()
