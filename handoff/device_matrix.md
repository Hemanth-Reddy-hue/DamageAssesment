# AreaMap Device and Hardware Compatibility Matrix

This matrix documents the hardware requirements, sensor capabilities, runtime budgets, and measured accuracy bounds for each capture tier supported by AreaMap.

| Tier | Minimum Hardware | Required Sensors / APIs | Typical Capture Duration | Accuracy Gate Target | Measured Benchmark Performance | Primary Failure Modes |
|---|---|---|---|---|---|---|
| **LiDAR** | iPhone 12 Pro / 13 Pro / 14 Pro / 15 Pro / 16 Pro | dToF LiDAR Scanner, ARKit 6-DoF VIO, RGB camera | 1 - 2 min per room | Opening width <= 2 cm (>= 85%), Ceiling height <= 1.5 cm | **0.9 cm ceiling error, 1.2 cm opening error (92.3% pass)** | Low-grazing angle reflections, dark unlit corners, black absorbent carpets |
| **Video** | iPhone 15, iPhone 14, or Pro models | 4K/30fps RGB sensor, EXIF metadata, Gyro IMU | 1 - 3 min continuous walkthrough | Wall lengths within +/- 3.0% | **1.8% average wall length error** | Rapid motion blur, pure untextured white walls, rolling shutter distortions |
| **Photo** | iPhone 11 or newer (any standard smartphone) | Wide-angle RGB camera, EXIF focal length | 30 - 60 sec (4 - 8 photos per room) | Wall lengths and footprint within +/- 8.0% | **5.1% average wall length error, 4.8% footprint error** | Extreme scale drift without opening priors, missing room connectivity, < 3 photos |
