"""Sudoless Apple Silicon power sampling via `macmon pipe` (IOReport + SMC).

all_power = CPU+GPU+ANE (SoC); ram_power + gpu_ram_power = DRAM; sys_power = SMC whole system.
`soc` returned by energy() = SoC + DRAM (the compute+memory energy of inference).
Each sample is the average power over the interval ending at its timestamp;
window energy = sum(power * overlap of that interval with the window).
"""
import json, subprocess, threading, time
from datetime import datetime


class PowerMeter:
    def __init__(self, interval_ms=100):
        self.samples = []  # (unix_t, soc+dram W, sys W, gpu W, cpu W, gpu temp C)
        self.proc = subprocess.Popen(["macmon", "pipe", "-i", str(interval_ms)],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()
        time.sleep(1.0)

    def _read(self):
        for line in self.proc.stdout:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = datetime.fromisoformat(d["timestamp"]).timestamp()
            self.samples.append((t, d["all_power"] + d.get("ram_power", 0) + d.get("gpu_ram_power", 0), d["sys_power"],
                                 d["gpu_power"], d["cpu_power"], d.get("temp", {}).get("gpu_temp_avg", float("nan"))))

    def energy(self, t0, t1):
        """Joules (soc, sys) over [t0, t1]. Waits until a sample past t1 exists."""
        deadline = time.time() + 2
        while (not self.samples or self.samples[-1][0] < t1) and time.time() < deadline:
            time.sleep(0.02)
        # each sample is the mean power over (prev_t, t]; integrate its overlap with [t0, t1]
        soc = sys_ = 0.0
        S = self.samples
        for prev, s in zip(S, S[1:]):
            ov = min(s[0], t1) - max(prev[0], t0)
            if ov > 0:
                soc += s[1] * ov
                sys_ += s[2] * ov
        return soc, sys_

    def temp(self):
        return self.samples[-1][5] if self.samples else float("nan")

    def idle_power(self, seconds=15):
        t0 = time.time(); time.sleep(seconds); t1 = time.time()
        soc, sys_ = self.energy(t0, t1)
        return soc / (t1 - t0), sys_ / (t1 - t0)

    def close(self):
        self.proc.terminate()
