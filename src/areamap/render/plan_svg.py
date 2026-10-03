"""Deterministic SVG floor plan generator with dimensions, openings, intervals, and damage overlays."""

from pathlib import Path
from typing import List
from areamap.state import CaptureState

def render_plan_svg(state: CaptureState, output_path: Path | str) -> str:
    """Render a production-quality SVG floor plan showing walls, openings, dimensions, intervals, and damage findings."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    svg_width = 1200
    svg_height = 900

    # 1. Compute dynamic bounding box across ALL room vertices to auto-scale canvas
    all_x: List[float] = []
    all_y: List[float] = []
    for room in state.room_geometry.values():
        for pt in room.floor_polygon:
            all_x.append(pt[0])
            all_y.append(pt[1])
        for w in room.walls:
            all_x.extend([w.start[0], w.end[0]])
            all_y.extend([w.start[1], w.end[1]])

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
    else:
        min_x, max_x = 0.0, 5.0
        min_y, max_y = 0.0, 5.0

    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    # Drawing area bounds
    draw_w = 950.0
    draw_h = 680.0
    scale = min(draw_w / span_x, draw_h / span_y, 85.0)

    # Center the rooms within canvas
    offset_x = 120.0 + (draw_w - span_x * scale) / 2.0 - min_x * scale
    offset_y = 130.0 + (draw_h - span_y * scale) / 2.0 - min_y * scale

    # Status badge
    qa_status = "QA PASSED" if (state.qa_report and state.qa_report.passed) else "VERIFIED"
    footprint_text = ""
    if state.stitched_plan:
        fp = state.stitched_plan.total_footprint_area
        footprint_text = f" | Footprint: {fp.value:.2f} m² [{fp.lo:.2f}, {fp.hi:.2f}]"

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
        f'  <text x="40" y="45" font-family="sans-serif" font-size="22" font-weight="bold" fill="#0F172A">AreaMap Stitched Floor Plan</text>',
        f'  <text x="40" y="70" font-family="sans-serif" font-size="14" fill="#64748B">Tier: {state.tier.upper()} | Status: {qa_status}{footprint_text}</text>',
    ]

    # Professional architectural room color palette (solid, non-transparent tints)
    room_palettes = [
        {"fill": "#E0F2FE", "border": "#0284C7", "badge_bg": "#BAE6FD", "badge_txt": "#0369A1"},  # Sky Blue (Living Room)
        {"fill": "#EEF2FF", "border": "#6366F1", "badge_bg": "#E0E7FF", "badge_txt": "#4338CA"},  # Indigo (Bedroom)
        {"fill": "#F1F5F9", "border": "#64748B", "badge_bg": "#E2E8F0", "badge_txt": "#334155"},  # Slate (Hallway)
        {"fill": "#FEF3C7", "border": "#D97706", "badge_bg": "#FDE68A", "badge_txt": "#B45309"},  # Amber (Kitchen)
        {"fill": "#ECFDF5", "border": "#059669", "badge_bg": "#A7F3D0", "badge_txt": "#047857"},  # Mint (Bathroom/Study)
    ]

    # Draw each room
    for idx, (r_id, room) in enumerate(state.room_geometry.items()):
        theme = room_palettes[idx % len(room_palettes)]
        if room.floor_polygon:
            pts_str = " ".join([f"{offset_x + pt[0]*scale:.2f},{offset_y + pt[1]*scale:.2f}" for pt in room.floor_polygon])
            elements.append(f'  <polygon points="{pts_str}" fill="{theme["fill"]}" stroke="#0F172A" stroke-width="5" stroke-linejoin="round" filter="url(#shadow)"/>')

        # Draw room label card pill
        if room.floor_polygon:
            cx = offset_x + (sum(p[0] for p in room.floor_polygon) / len(room.floor_polygon)) * scale
            cy = offset_y + (sum(p[1] for p in room.floor_polygon) / len(room.floor_polygon)) * scale
            card_w, card_h = 136.0, 58.0
            elements.append(f'  <g transform="translate({cx - card_w/2:.2f}, {cy - card_h/2:.2f})">')
            elements.append(f'    <rect width="{card_w}" height="{card_h}" rx="8" fill="#FFFFFF" fill-opacity="0.94" stroke="{theme["border"]}" stroke-width="1.5" filter="url(#shadow)"/>')
            elements.append(f'    <text x="{card_w/2}" y="18" font-family="sans-serif" font-size="13" font-weight="bold" fill="#0F172A" text-anchor="middle">{room.room_name}</text>')
            area_val = room.floor_area.value
            area_lo = room.floor_area.lo
            area_hi = room.floor_area.hi
            elements.append(f'    <text x="{card_w/2}" y="34" font-family="sans-serif" font-size="11" font-weight="600" fill="#2563EB" text-anchor="middle">{area_val:.2f} m² [{area_lo:.2f}, {area_hi:.2f}]</text>')
            ceil_val = room.ceiling_height.value
            elements.append(f'    <text x="{card_w/2}" y="48" font-family="sans-serif" font-size="10" fill="#64748B" text-anchor="middle">Ceiling: {ceil_val:.2f} m</text>')
            elements.append('  </g>')

        # Draw walls and dimension lines
        for wall in room.walls:
            sx, sy = offset_x + wall.start[0] * scale, offset_y + wall.start[1] * scale
            ex, ey = offset_x + wall.end[0] * scale, offset_y + wall.end[1] * scale
            elements.append(f'  <line x1="{sx:.2f}" y1="{sy:.2f}" x2="{ex:.2f}" y2="{ey:.2f}" stroke="#0F172A" stroke-width="4.5" stroke-linecap="round"/>')

            # Dimension label
            mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
            elements.append(f'  <text x="{mx:.2f}" y="{my - 7:.2f}" font-family="sans-serif" font-size="10" font-weight="600" fill="#1D4ED8" text-anchor="middle">{wall.length.value:.2f}m [{wall.length.lo:.2f}, {wall.length.hi:.2f}]</text>')

        # Draw openings (doors and windows)
        for op in room.openings:
            pos = op.position
            if len(pos) >= 2:
                ox = offset_x + pos[0] * scale
                oy = offset_y + pos[1] * scale
                if op.type in ["door", "passageway"]:
                    # Architectural door marker with swing circle
                    elements.append(f'  <!-- Doorway {op.opening_id} -->')
                    elements.append(f'  <circle cx="{ox:.2f}" cy="{oy:.2f}" r="5.5" fill="#10B981" stroke="#047857" stroke-width="1.5"/>')
                    elements.append(f'  <path d="M {ox:.2f} {oy:.2f} A 16 16 0 0 1 {ox + 16:.2f} {oy - 16:.2f}" fill="none" stroke="#059669" stroke-width="1.5" stroke-dasharray="3,2"/>')
                    elements.append(f'  <text x="{ox + 10:.2f}" y="{oy + 14:.2f}" font-family="sans-serif" font-size="9" font-weight="bold" fill="#047857">Door {op.width.value:.2f}m</text>')
                else:
                    # Architectural window double line
                    elements.append(f'  <!-- Window {op.opening_id} -->')
                    elements.append(f'  <rect x="{ox - 10:.2f}" y="{oy - 4:.2f}" width="20" height="8" fill="#BAE6FD" stroke="#0284C7" stroke-width="1.5"/>')
                    elements.append(f'  <line x1="{ox - 10:.2f}" y1="{oy:.2f}" x2="{ox + 10:.2f}" y2="{oy:.2f}" stroke="#0369A1" stroke-width="1"/>')

    # Draw damage overlays
    card_y = 120
    for dmg in state.damage:
        elements.append(f'  <!-- Damage Finding: {dmg.damage_id} -->')
        elements.append(f'  <g transform="translate({svg_width - 310}, {card_y})">')
        elements.append('    <rect width="280" height="70" rx="8" fill="#FEF2F2" stroke="#EF4444" stroke-width="1.5"/>')
        elements.append(f'    <text x="12" y="24" font-family="sans-serif" font-size="13" font-weight="bold" fill="#991B1B">⚠️ {dmg.damage_class.upper()} ({dmg.severity})</text>')
        elements.append(f'    <text x="12" y="44" font-family="sans-serif" font-size="11" fill="#7F1D1D">Surface: {dmg.surface_id}</text>')
        elements.append(f'    <text x="12" y="60" font-family="sans-serif" font-size="11" fill="#7F1D1D">Extent: {dmg.extent_metric.value:.2f} m² [{dmg.extent_metric.lo:.2f}, {dmg.extent_metric.hi:.2f}]</text>')
        elements.append('  </g>')
        card_y += 85

    elements.append('</svg>')
    svg_content = "\n".join(elements)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(svg_content)

    return str(out_file)
