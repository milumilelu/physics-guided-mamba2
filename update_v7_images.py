"""Replace images in V7 docx with updated PNG files."""
import zipfile
import shutil
import os

v7_path = r"C:\Users\RZF\Desktop\博士课题资料\physics-guided Mamba-2\专利\超快激光多脉冲烧蚀形貌预测_专利技术交底书_V7_最终版.docx"
png_dir = r"C:\Users\RZF\Desktop\博士课题资料\physics-guided Mamba-2\专利\figures_png"
temp_path = v7_path + ".tmp"

# Mapping: docx image name -> source PNG file
image_map = {
    "word/media/image1.png": "fig1.png",
    "word/media/image2.png": "fig2.png",
    "word/media/image3.png": "fig3.png",
    "word/media/image4.png": "fig4.png",
    "word/media/image5.png": "fig5.png",
    "word/media/image6.png": "fig6.png",
}

# Read original zip and write new zip with replaced images
with zipfile.ZipFile(v7_path, 'r') as zin:
    with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename in image_map:
                # Replace with new PNG
                png_path = os.path.join(png_dir, image_map[item.filename])
                with open(png_path, 'rb') as f:
                    new_data = f.read()
                zout.writestr(item, new_data)
                print(f"Replaced {item.filename} with {image_map[item.filename]} ({len(new_data)} bytes)")
            else:
                # Copy unchanged
                zout.writestr(item, zin.read(item.filename))

# Replace original with temp
shutil.move(temp_path, v7_path)
print(f"\nV7 updated with new figures: {v7_path}")
