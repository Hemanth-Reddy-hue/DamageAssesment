# AreaMap iPhone Field Capture Protocol (Route 2)

A one-page operational field guide for non-engineers capturing residential and commercial properties for floor plan reconstruction, damage assessment, and repair scoping.

---

## 1. Universal Scanning Best Practices (All Tiers)
- **Lighting**: Turn on all interior lights in every room. Open internal doors completely.
- **Hazards & Specularities**:
  - Stand at a 45-degree angle to large mirrors, full-height windows, and reflective metallic surfaces to avoid specular LiDAR/depth multi-path returns.
  - Close blinds if strong direct sunlight creates high-contrast glare stripes on the floor.
- **Pacing**: Walk at a steady pace of approximately **0.5 m/s** (one deliberate footstep per second). Never swing the phone rapidly.
- **Handover**: Export the raw capture folder via AirDrop or USB cable directly into `data/raw/<room_name>/<tier>/run1/`.

---

## 2. Tier 1: LiDAR Capture (iPhone Pro with LiDAR)
1. **App Setup**:
   - Open **3D Scanner App** (or **Record3D**).
   - Set resolution to *High*, range to *5.0 meters*, confidence filter to *High*.
2. **Scan Routine**:
   - Start in a primary doorway facing into the room.
   - Walk the perimeter clockwise at a distance of 1.5m to 2.5m from walls.
   - Sweep phone smoothly vertically (floor to ceiling) to capture both baseboards and ceiling corners.
   - Finish the scan by returning to the exact doorway where you started (completing loop closure).
3. **Export Steps**:
   - Tap Share / Export -> Select **Raw Data Bundle** (Depth + Confidence + Odometry CSV + Intrinsics).
   - Save folder as `data/raw/<room>/lidar/run1/`.

---

## 3. Tier 2: Video Walkthrough Capture (Standard iPhone 15+)
1. **App Setup**:
   - Open standard iOS Camera app in **Video mode**.
   - Set resolution to **4K at 30 fps** (or 1080p 60 fps). Turn off cinematic blur.
2. **Walkthrough Routine**:
   - Hold phone with two hands at chest height, tilted slightly downward (~15 degrees) so floor-wall junctions remain visible.
   - Walk smoothly without panning abruptly.
   - Maintain continuous visual overlap between consecutive walls and doorways.
   - Walk each room in 60 to 90 seconds.
3. **Export Steps**:
   - Transfer original `.mov` or `.mp4` file directly to `data/raw/<room>/video/run1/rgb.mp4`.

---

## 4. Tier 3: Photo Folder Capture (Standard iPhone 15+)
1. **Photo Count**:
   - Minimum: **4 photos** per room (one from each corner looking toward the center).
   - Optimal: **6 to 8 photos** per room (corners + perimeter centers + opening close-ups).
2. **Framing Rules**:
   - Stand firmly in the corner; hold the camera level.
   - Ensure the image contains both the ceiling-wall junction and floor-wall junction.
   - Step into doorways and capture one photo showing both connected spaces.
3. **Export Steps**:
   - Group photos into a folder per room: `data/raw/<room>/photo/run1/*.jpg`.
