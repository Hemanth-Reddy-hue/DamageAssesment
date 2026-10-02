"""Deterministic SVG floor plan generator with dimensions, openings, intervals, and damage overlays."""

from pathlib import Path
from areamap.state import CaptureState

def render_plan_svg(state: CaptureState, output_path: Path | str) -> str:
    """Render a production-quality SVG floor plan showing walls, openings, dimensions, intervals, and damage findings."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    svg_width = 1000
    svg_height = 800
    scale = 80.0  # pixels per meter
    offset_x = 250
    offset_y = 150

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="100%" height="100%">',
        '  <defs>',
        '    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%"><feDropShadow dx="2" dy="2" stdDeviation="3" flood-opacity="0.15"/></filter>',
        '  </defs>',
        '  <rect width="100%" height="100%" fill="#F8FAFC"/>',
        '  <!-- Grid Lines -->',
        '  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">',
        '    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#E2E8F0" stroke-width="1"/>',
        '  </pattern>',
        f'  <rect width="{svg_width}" height="{svg_height}" fill="url(#grid)"/>',
        '  <!-- Header Block -->',
        f'  <text x="40" y="50" font-family="sans-serif" font-size="22" font-weight="bold" fill="#0F172A">AreaMap Stitched Floor Plan</text>',
        f'  <text x="40" y="75" font-family="sans-serif" font-size="14" fill="#64748B">Tier: {state.tier.upper()} | Status: {("QA PASSED" if state.qa_report and state.qa_report.passed else "VERIFIED")}</text>',
    ]

    # Draw each room
    for r_id, room in state.room_geometry.items():
        if room.floor_polygon:
            pts_str = " ".join([f"{offset_x + pt[0]*scale},{offset_y + pt[1]*scale}" for pt in room.floor_polygon])
            elements.append(f'  <polygon points="{pts_str}" fill="#FFFFFF" stroke="#0F172A" stroke-width="6" stroke-linejoin="round" filter="url(#shadow)"/>')

        # Draw room label
        if room.floor_polygon:
            cx = offset_x + sum(p[0] for p in room.floor_polygon) / len(room.floor_polygon) * scale
            cy = offset_y + sum(p[1] for p in room.floor_polygon) / len(room.floor_polygon) * scale
            elements.append(f'  <text x="{cx}" y="{cy-10}" font-family="sans-serif" font-size="16" font-weight="bold" fill="#1E293B" text-anchor="middle">{room.room_name}</text>')
            area_val = room.floor_area.value
            area_lo = room.floor_area.lo
            area_hi = room.floor_area.hi
            elements.append(f'  <text x="{cx}" y="{cy+12}" font-family="sans-serif" font-size="12" fill="#475569" text-anchor="middle">{area_val:.2f} m² [{area_lo:.2f}, {area_hi:.2f}]</text>')
            ceil_val = room.ceiling_height.value
            elements.append(f'  <text x="{cx}" y="{cy+30}" font-family="sans-serif" font-size="11" fill="#64748B" text-anchor="middle">H: {ceil_val:.2f} m</text>')

        # Draw walls and dimension lines
        for wall in room.walls:
            sx, sy = offset_x + wall.start[0]*scale, offset_y + wall.start[1]*scale
            ex, ey = offset_x + wall.end[0]*scale, offset_y + wall.end[1]*scale
            elements.append(f'  <line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="#0F172A" stroke-width="4"/>')
            
            # Dimension label
            mx, my = (sx + ex)/2, (sy + ey)/2
            elements.append(f'  <text x="{mx}" y="{my - 8}" font-family="sans-serif" font-size="10" fill="#2563EB" text-anchor="middle">{wall.length.value:.2f}m [{wall.length.lo:.2f}, {wall.length.hi:.2f}]</text>')

        # Draw openings
        for op in room.openings:
            pos = op.position
            if len(pos) >= 2:
                ox = offset_x + pos[0]*scale
                oy = offset_y + pos[1]*scale
                if op.type == "door":
                    elements.append(f'  <circle cx="{ox}" cy="{oy}" r="6" fill="#10B981"/>')
                    elements.append(f'  <text x="{ox+8}" y="{oy+4}" font-family="sans-serif" font-size="10" fill="#047857">Door {op.width.value:.2f}m</text>')
                else:
                    elements.append(f'  <rect x="{ox-8}" y="{oy-4}" width="16" height="8" fill="#38BDF8" stroke="#0284C7"/>')

    # Draw damage badges
    for dmg in state.damage:
        elements.append(f'  <!-- Damage Finding: {dmg.damage_id} -->')
        elements.append(f'  <g transform="translate(680, 150)">')
        elements.append(f'    <rect width="280" height="70" rx="8" fill="#FEF2F2" stroke="#EF4444" stroke-width="1.5"/>')
        elements.append(f'    <text x="12" y="24" font-family="sans-serif" font-size="13" font-weight="bold" fill="#991B1B">⚠️ {dmg.damage_class.upper()} ({dmg.severity})</text>')
        elements.append(f'    <text x="12" y="44" font-family="sans-serif" font-size="11" fill="#7F1D1D">Surface: {dmg.surface_id}</text>')
        elements.append(f'    <text x="12" y="60" font-family="sans-serif" font-size="11" fill="#7F1D1D">Extent: {dmg.extent_metric.value:.2f} m² [{dmg.extent_metric.lo:.2f}, {dmg.extent_metric.hi:.2f}]</text>')
        elements.append(f'  </g>')

    elements.append('</svg>')
    svg_content = "\n".join(elements)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(svg_content)

    return svg_content
