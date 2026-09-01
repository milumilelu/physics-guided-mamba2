"""Extract text content and structure from each SVG figure for review."""
import re
import os

fig_dir = r"C:\Users\RZF\Desktop\博士课题资料\physics-guided Mamba-2\专利\figures"

for i in range(1, 7):
    fname = f"fig{i}.svg"
    fpath = os.path.join(fig_dir, fname)
    print(f"\n{'='*80}")
    print(f"FIG {i}: {fname}")
    print(f"{'='*80}")

    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Get SVG dimensions
    width_match = re.search(r'width="([^"]+)"', content)
    height_match = re.search(r'height="([^"]+)"', content)
    print(f"Dimensions: {width_match.group(1) if width_match else '?'} x {height_match.group(1) if height_match else '?'}")

    # Extract all text elements
    texts = re.findall(r'<text[^>]*>(.*?)</text>', content, re.DOTALL)
    # Clean text (remove tspan tags etc)
    clean_texts = []
    for t in texts:
        clean = re.sub(r'<[^>]+>', '', t).strip()
        if clean:
            clean_texts.append(clean)

    print(f"\nText elements ({len(clean_texts)}):")
    for j, t in enumerate(clean_texts):
        print(f"  [{j:2d}] {t[:100]}")

    # Count shapes
    rects = len(re.findall(r'<rect', content))
    circles = len(re.findall(r'<circle', content))
    lines = len(re.findall(r'<line', content))
    paths = len(re.findall(r'<path', content))
    polygons = len(re.findall(r'<polygon', content))
    print(f"\nShapes: rect={rects}, circle={circles}, line={lines}, path={paths}, polygon={polygons}")

    # Check for markers (should be 0 per README)
    markers = len(re.findall(r'marker-end', content))
    print(f"marker-end references: {markers} (should be 0)")

    # Check for issues
    issues = []
    if markers > 0:
        issues.append(f"Has {markers} marker-end references (should use explicit polygon arrows)")

    # Check for very small font sizes
    font_sizes = re.findall(r'font-size["\s:]+(\d+)', content)
    small_fonts = [int(s) for s in font_sizes if int(s) < 18]
    if small_fonts:
        issues.append(f"Has {len(small_fonts)} text elements with font-size < 18 (min should be 18 for subscripts)")

    if issues:
        print(f"\nISSUES FOUND:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print(f"\n✓ No obvious issues found")
