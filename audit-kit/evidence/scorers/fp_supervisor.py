"""Runner supervisor + canary (N-2 priority 1; G-730 Stage 0; G6 in the state map). Windows/Linux, stdlib only.

Launches ONE producer command detached from the console (no console-window close can kill it), watches its log for a heartbeat,
records the exit cause honestly (exit code, last log lines, whether the heartbeat had stalled, GPU memory if nvidia-smi exists) in
a supervisor receipt, and NEVER restarts anything by itself: a dead producer is a result to be read, not hidden. The canary is a
tiny pre-flight that proves the launch path end to end (python + env + working dir + log write) before the real command runs.

Usage:
  python fp_supervisor.py --name NAME --workdir DIR --log LOG --heartbeat-regex REGEX --stall-seconds N --receipt OUT.json -- CMD ARGS...
  python fp_supervisor.py --canary --workdir DIR --receipt OUT.json      (runs the canary only)
"""
import json
import os
import re
import subprocess
import sys
import time


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def gpu_mem():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        return out
    except Exception:  # noqa: BLE001
        return None


def parse(argv):
    a = {"--name": "producer", "--workdir": ".", "--log": None, "--heartbeat-regex": r"\[\d\d:\d\d:\d\d\]", "--stall-seconds": "1800", "--receipt": None, "--canary": False, "--poll-seconds": "30"}
    cmd = []
    i = 1
    while i < len(argv):
        if argv[i] == "--":
            cmd = argv[i + 1:]
            break
        if argv[i] == "--canary":
            a["--canary"] = True
            i += 1
        elif argv[i] in a and i + 1 < len(argv):
            a[argv[i]] = argv[i + 1]
            i += 2
        else:
            raise SystemExit("bad argv at %r" % argv[i])
    if a["--receipt"] is None:
        raise SystemExit("--receipt required")
    return a, cmd


def canary(workdir, receipt_path):
    """Prove the launch path: spawn a detached python child that writes a heartbeat line to a log, wait for it, read it back."""
    log = os.path.join(workdir, "supervisor_canary.log")
    if os.path.exists(log):
        os.remove(log)
    code = "import time,sys\nfor k in range(3):\n    print('[%s] canary %d' % (time.strftime('%H:%M:%S'), k), flush=True); time.sleep(1)\nprint('CANARY-DONE', flush=True)\n"
    t0 = time.time()
    p = spawn([sys.executable, "-c", code], workdir, log)
    rc = p.wait(timeout=60)
    lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
    ok = rc == 0 and any("CANARY-DONE" in l for l in lines) and sum(1 for l in lines if "canary" in l) == 3
    rec = {"kind": "supervisor_canary", "ok": ok, "exit_code": rc, "lines": lines, "elapsed_s": round(time.time() - t0, 2), "python": sys.executable, "workdir": os.path.abspath(workdir), "stamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_receipt(receipt_path, rec)
    print("CANARY", "OK" if ok else "FAIL", rc)
    return ok


def spawn(cmd, workdir, log):
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    fh = open(log, "ab", buffering=0)
    return subprocess.Popen(cmd, cwd=workdir, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags, close_fds=(os.name != "nt"))


def write_receipt(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, sort_keys=True, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def supervise(a, cmd):
    workdir, log = a["--workdir"], a["--log"] or os.path.join(a["--workdir"], "%s_supervised.log" % a["--name"])
    hb = re.compile(a["--heartbeat-regex"])
    stall, poll = float(a["--stall-seconds"]), float(a["--poll-seconds"])
    t0 = time.time()
    p = spawn(cmd, workdir, log)
    rec = {"kind": "supervisor", "name": a["--name"], "cmd": cmd, "workdir": os.path.abspath(workdir), "log": os.path.abspath(log), "pid": p.pid, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "gpu_at_start": gpu_mem(), "events": []}
    write_receipt(a["--receipt"], rec)
    last_hb, offset, stalled_flag, partial = time.time(), 0, False, b""
    while True:
        rc = p.poll()
        try:
            size = os.path.getsize(log)
        except OSError:
            size = 0
        if size > offset:                                               # G-732: read only NEW bytes at the offset; a heartbeat counts only if it is in a NEW complete line
            with open(log, "rb") as fh:
                fh.seek(offset)
                chunk = fh.read(size - offset)
            offset = size
            data = partial + chunk
            lines = data.split(b"\n")
            partial = lines[-1]
            for ln in lines[:-1]:
                if hb.search(ln.decode("utf-8", errors="replace")):
                    last_hb = time.time()
        if rc is not None:
            break
        if time.time() - last_hb > stall and not stalled_flag:
            stalled_flag = True
            rec["events"].append({"stall_detected": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds_since_heartbeat": round(time.time() - last_hb), "gpu": gpu_mem()})
            write_receipt(a["--receipt"], rec)
        time.sleep(poll)
    tail = open(log, "rb").read()[-8192:].decode("utf-8", errors="replace").splitlines()[-25:]
    rec.update({"exit_code": rc, "ended": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_s": round(time.time() - t0, 1), "stalled_before_exit": stalled_flag, "log_tail": tail, "log_sha256": sha256_file(log) if os.path.exists(log) else None, "gpu_at_end": gpu_mem(),
                "cause": ("ok" if rc == 0 else ("killed-or-crashed:%s" % rc))})
    write_receipt(a["--receipt"], rec)
    print("SUPERVISOR-DONE", a["--name"], "exit", rc, "stalled", stalled_flag)
    return rc


def main():
    a, cmd = parse(sys.argv)
    if a["--canary"]:
        ok = canary(a["--workdir"], a["--receipt"])
        raise SystemExit(0 if ok else 2)
    if not cmd:
        raise SystemExit("no command after --")
    ok = canary(a["--workdir"], a["--receipt"] + ".canary.json")
    if not ok:
        raise SystemExit("CANARY-FAILED: refusing to launch the producer")
    raise SystemExit(supervise(a, cmd))


if __name__ == "__main__":
    main()
