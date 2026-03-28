#!/usr/bin/env python3
"""
gem5 config for the pizza/steak/sushi matrix multiply workload.
"""

from __future__ import annotations

import argparse
import os

import m5
from m5.objects import AddrRange, Cache, DDR3_1600_8x8, L2XBar, MemCtrl, Process, Root
from m5.objects import SEWorkload, SrcClockDomain, System, SystemXBar, TimingSimpleCPU, VoltageDomain


parser = argparse.ArgumentParser()
parser.add_argument("--assoc", type=int, default=4)
parser.add_argument("--l1size", default="32KiB")
parser.add_argument("--l2size", default="256KiB")
parser.add_argument("--binary", default="./matmul_food")
args = parser.parse_args()

binary_path = os.path.abspath(args.binary)

print(f"=== gem5 config: assoc={args.assoc}, L1={args.l1size}, L2={args.l2size} ===")
print(f"=== workload: {binary_path} ===")

system = System()
system.clk_domain = SrcClockDomain(clock="1GHz", voltage_domain=VoltageDomain())
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MiB")]
system.cpu = TimingSimpleCPU()

system.cpu.dcache = Cache(
    size=args.l1size,
    assoc=args.assoc,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=4,
    tgts_per_mshr=20,
)
system.cpu.icache = Cache(
    size="32KiB",
    assoc=4,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=4,
    tgts_per_mshr=20,
)
system.l2cache = Cache(
    size=args.l2size,
    assoc=8,
    tag_latency=20,
    data_latency=20,
    response_latency=20,
    mshrs=20,
    tgts_per_mshr=12,
)

system.cpu.icache_port = system.cpu.icache.cpu_side
system.cpu.dcache_port = system.cpu.dcache.cpu_side

system.l2bus = L2XBar()
system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports
system.l2cache.cpu_side = system.l2bus.mem_side_ports

system.membus = SystemXBar()
system.l2cache.mem_side = system.membus.cpu_side_ports
system.cpu.createInterruptController()
system.system_port = system.membus.cpu_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.workload = SEWorkload.init_compatible(binary_path)

process = Process()
process.executable = binary_path
process.cmd = [binary_path]
process.cwd = os.getcwd()

system.cpu.workload = process
system.cpu.createThreads()

root = Root(full_system=False, system=system)
m5.instantiate()

print("Starting simulation...")
exit_event = m5.simulate()
print(f"Simulation done: {exit_event.getCause()}")
print(f"Ticks: {m5.curTick()}")
