# Eridian Wiki

**Real-time 3D world reconstruction from a single camera.**

Eridian turns any webcam or phone camera into a spatial scanner. It watches what you see, understands depth, tracks camera motion, and builds a colored 3D map of your surroundings.

## Quick Links

- [[Getting Started]] — Install and run your first scan
- [[Environment Mapping Guide]] — Build a real 3D map of a room or space using LingBot-MAP
- [[LingBot-MAP (GCT) Backend]] — Technical deep-dive into the transformer-based reconstruction
- [[Classic Pipeline]] — How the default depth + optical flow backend works
- [[Python API Reference]] — Use Eridian in your own code
- [[CLI Reference]] — All command-line flags
- [[Tuning and Tips]] — Get better results from your scans

## Backends

Eridian has two reconstruction backends:

| | **Classic** (default) | **LingBot-MAP (GCT)** |
|---|---|---|
| How it works | Depth Anything V2 + optical flow + PnP | Single transformer predicts pose + depth + 3D points |
| Input | Webcam or video | Video file |
| Real-time | Yes (~5 FPS) | Batch processing |
| Best for | Live scanning, interactive use | High-quality offline 3D environment maps |
