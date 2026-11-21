"""Mininet topology definitions and benchmark configuration validation."""
from __future__ import annotations

from mininet.net import Mininet
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.node import OVSSwitch


class StandaloneSwitch(OVSSwitch):
    def __init__(self, name, **kwargs):
        kwargs.setdefault("failMode", "standalone")
        super().__init__(name, **kwargs)


class DumbbellTopo(Topo):
    def build(self, delay_ms=20, queue_depth=10, bandwidth_mbps=10):
        h1, h2, h3, h4 = (self.addHost(f"h{i}") for i in range(1, 5))
        s1, s2 = self.addSwitch("s1"), self.addSwitch("s2")
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(s1, s2, cls=TCLink, bw=bandwidth_mbps, delay=f"{delay_ms}ms", max_queue_size=queue_depth)
        self.addLink(h3, s2)
        self.addLink(h4, s2)


class StarTopo(Topo):
    def build(self, delay_ms=20, queue_depth=10, bandwidth_mbps=10):
        switch = self.addSwitch("s1")
        for i in range(1, 5):
            host = self.addHost(f"h{i}")
            self.addLink(host, switch, cls=TCLink, bw=bandwidth_mbps,
                         delay=f"{delay_ms}ms", max_queue_size=queue_depth)


class CustomTreeTopo(Topo):
    def build(self, delay_ms=20, queue_depth=10, bandwidth_mbps=10):
        root, left, right = (self.addSwitch(f"s{i}") for i in range(1, 4))
        self.addLink(root, left, cls=TCLink, bw=bandwidth_mbps, delay=f"{delay_ms}ms", max_queue_size=queue_depth)
        self.addLink(root, right, cls=TCLink, bw=bandwidth_mbps, delay=f"{delay_ms}ms", max_queue_size=queue_depth)
        for i, switch in enumerate((left, left, right, right), 1):
            host = self.addHost(f"h{i}")
            self.addLink(host, switch)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Benchmark config must be a JSON object")
    defaults = {"protocol": "TCP", "topology": "dumbbell", "delay_ms": 20,
                "queue_depth_start": 10, "queue_depth_end": 10, "queue_depth_step": 10,
                "trials": 3, "duration": 5, "bandwidth_mbps": 10, "udp_rate_mbps": 8}
    result = defaults | config
    result["protocol"] = str(result["protocol"]).upper()
    result["topology"] = str(result["topology"]).lower()
    if result["protocol"] not in {"TCP", "UDP"}:
        raise ValueError("protocol must be TCP or UDP")
    if result["topology"] not in {"dumbbell", "star", "tree"}:
        raise ValueError("topology must be dumbbell, star, or tree")
    for key in ("delay_ms", "queue_depth_start", "queue_depth_end", "queue_depth_step", "trials", "duration", "bandwidth_mbps", "udp_rate_mbps"):
        value = result[key]
        if isinstance(value, bool) or int(value) != float(value):
            raise ValueError(f"{key} must be an integer")
        result[key] = int(value)
    if result["delay_ms"] < 0 or result["queue_depth_start"] < 1:
        raise ValueError("delay_ms must be non-negative and queue depths must be positive")
    if result["queue_depth_end"] < result["queue_depth_start"] or result["queue_depth_step"] < 1:
        raise ValueError("queue depth range/step is invalid")
    if result["trials"] < 1 or result["trials"] > 100 or result["duration"] < 1 or result["duration"] > 300:
        raise ValueError("trials must be 1..100 and duration 1..300 seconds")
    if result["bandwidth_mbps"] < 1 or result["udp_rate_mbps"] < 1:
        raise ValueError("bandwidth and UDP target rate must be positive")
    if len(range(result["queue_depth_start"], result["queue_depth_end"] + 1, result["queue_depth_step"])) > 101:
        raise ValueError("queue sweep may contain at most 101 values")
    return result


def create_topology(config):
    config = validate_config(config)
    params = {"delay_ms": config["delay_ms"], "queue_depth": config["queue_depth_start"],
              "bandwidth_mbps": config["bandwidth_mbps"]}
    topology = {"dumbbell": DumbbellTopo, "star": StarTopo, "tree": CustomTreeTopo}[config["topology"]](**params)
    return Mininet(topo=topology, link=TCLink, switch=StandaloneSwitch,
                   controller=None, autoSetMacs=True, build=True)
