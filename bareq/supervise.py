"""Run bareq.train, restart it if it dies, and say why it died.

  python -m bareq.supervise --run all

Training runs as its own process with its output in <out>/console_<run>.txt; this script prints that
output as it arrives. If training ends without finishing (killed, crashed), it starts again from the last
checkpoint. Exit codes from bareq.train: 0 with a metrics file = finished, 75 = session time limit
(do not restart), anything else = restart.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

T0 = time.time()
TIME_LIMIT = 75


def say(*s):
    try:
        print(*s, flush=True)
    except OSError:  # the notebook's output pipe went away; keep training anyway
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--first-args", default="", help="extra arguments for the first attempt only (tests)")
    a, rest = ap.parse_known_args()
    out = a.out or f"/kaggle/working/bareq-run-{a.run}"
    os.makedirs(out, exist_ok=True)
    console = os.path.join(out, f"console_{a.run}.txt")
    metrics = os.path.join(out, f"metrics_{a.run}.json")
    for s in (signal.SIGHUP, signal.SIGPIPE):
        signal.signal(s, signal.SIG_IGN)

    for attempt in range(1, a.max_attempts + 1):
        left = a.max_hours - (time.time() - T0) / 3600
        if left < 0.2:
            say("no session time left; run again with this output attached as an input")
            return
        cmd = [sys.executable, "-u", "-m", "bareq.train", "--run", a.run, "--out", out,
               "--max-hours", f"{left:.3f}", *rest, *(a.first_args.split() if attempt == 1 else [])]
        with open(console, "ab") as f:
            f.write(f"\n===== attempt {attempt}: {' '.join(cmd[2:])}\n".encode())
            f.flush()
            pos = f.tell()
            started = time.time()
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        with open(console, "rb") as r:
            r.seek(pos)
            while True:
                rc = p.poll()
                chunk = r.read()
                if chunk:
                    say(chunk.decode("utf-8", "replace").rstrip("\n"))
                if rc is not None and not chunk:
                    break
                time.sleep(5)
        why = f"killed by signal {-rc} ({signal.Signals(-rc).name})" if rc < 0 else f"exit code {rc}"
        if rc == 0 and os.path.exists(metrics):
            return
        note = f"===== attempt {attempt} ended after {(time.time() - T0) / 60:.0f} min: {why}"
        dm = subprocess.run("dmesg 2>/dev/null | grep -i -E 'out of memory|oom|killed process' | tail -n 5",
                            shell=True, capture_output=True, text=True).stdout.strip()
        if dm:
            note += "\n" + dm
        with open(console, "a") as f:
            f.write(note + "\n")
        say(note)
        if rc == TIME_LIMIT:
            return
        if rc > 0 and time.time() - started < 90:  # a setup error, not a crash in training: restarting won't help
            sys.exit(rc)
        say("starting again from the last checkpoint")
    say("gave up after too many attempts")
    sys.exit(1)


if __name__ == "__main__":
    main()
