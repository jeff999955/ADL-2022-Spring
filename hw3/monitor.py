import shutil, subprocess, sys, time, os, pynvml, signal

usage = 0

args = sys.argv[1:]
args[0] = shutil.which(args[0])

pynvml.nvmlInit()
child = subprocess.Popen(args)
print(child.pid)
while child.poll() is None:
    time.sleep(1)
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)
    for process in processes:
        if process.pid == child.pid:
            usage = max(usage, (process.usedGpuMemory // 1048576))

print(usage)
