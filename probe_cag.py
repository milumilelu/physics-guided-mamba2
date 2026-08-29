import zipfile, struct, array
from pathlib import Path

CAG = Path(r"C:\Users\RZF\Desktop\专利\氧化锆\120组直线.cag")
HEIGHT_KEY = "4d137b4a-bf22-49d5-96a8-9b07b3fc5d02"
VK4_KEYS = ["meas_conds","color_peak","color_light","light","unknown_4","unknown_5","height","unknown_7","unknown_8","color_peak_thumb","color_thumb","light_thumb","height_thumb","assembly_info","line_measure","line_thickness","string_data","reserved"]

z = zipfile.ZipFile(CAG)
entries = z.infolist()
direct = {}; vk4 = {}
for info in entries:
    parts = info.filename.split("/")
    if len(parts) == 5 and parts[1].isdigit() and parts[3] == HEIGHT_KEY and info.file_size > 1:
        direct.setdefault(int(parts[1]), {})[parts[3]] = info
    if info.file_size == 568500 and len(parts) >= 3 and parts[1].isdigit():
        if z.read(info)[:4] == b"VK4_":
            vk4[int(parts[1])] = info
print("zip entries:", len(entries))
print("direct groups:", len(direct), "vk4 groups:", len(vk4))
g = 1
buf = z.read(vk4[g])
offs = dict(zip(VK4_KEYS, struct.unpack_from("<18I", buf, 12)))
h_off = offs["height"]
w, h, bit, comp, nbytes = struct.unpack_from("<5I", buf, h_off)
print(f"group {g}: width={w} height={h} bit_depth={bit} declared_bytes={nbytes}")
raw = z.read(direct[g][HEIGHT_KEY])
a = array.array("I"); a.frombytes(raw)
assert len(a) == w * h, (len(a), w * h)
vals = a.tolist()
zmin = zmax = vals[0]; zsum = 0
for v in vals:
    if v < zmin: zmin = v
    if v > zmax: zmax = v
    zsum += v
print("sample uint32: min=%d max=%d mean=%.1f" % (zmin, zmax, zsum / len(vals)))
print("=> decodes to a %dx%d = %d point height-field (point cloud)" % (w, h, len(vals)))
