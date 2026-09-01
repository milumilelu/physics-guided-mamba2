from docx import Document
from docx.oxml.ns import qn

doc = Document(r"C:\Users\RZF\Desktop\博士课题资料\physics-guided Mamba-2\专利\超快激光多脉冲烧蚀形貌预测_专利技术交底书_V6_模板格式版.docx")

img_count = 0
for i, p in enumerate(doc.paragraphs):
    drawings = p._element.findall(".//" + qn("w:drawing"))
    if drawings:
        img_count += 1
        txt = p.text[:40] if p.text else "no-text"
        print(f"  [{i}] Paragraph with image: {txt}")

rel_imgs = sum(1 for rel in doc.part.rels.values() if "image" in rel.reltype)
print(f"  Paragraphs with drawings: {img_count}")
print(f"  Image relationships: {rel_imgs}")
