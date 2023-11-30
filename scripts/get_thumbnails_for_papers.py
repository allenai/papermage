import os
from glob import glob

from tqdm import tqdm

from papermage.rasterizers import PDF2ImageRasterizer

pdf_paths = [
    "tests/fixtures/2302.07302v1.pdf",
    "tests/fixtures/2020.acl-main.447.pdf",
    "tests/fixtures/2023.eacl-main.121.pdf",
    "tests/fixtures/2303.14334v2.pdf",
    "tests/fixtures/2304.02623v1.pdf",
    "tests/fixtures/papermage.pdf",
]
thumbnail_dir = "temp/"
os.makedirs(thumbnail_dir, exist_ok=True)

rasterizer = PDF2ImageRasterizer()

for pdf_path in tqdm(pdf_paths):
    pngfile = os.path.join(thumbnail_dir, os.path.basename(pdf_path).replace(".pdf", ".png"))
    if os.path.exists(pngfile):
        continue

    pdfimages = rasterizer.rasterize(input_pdf_path=pdf_path, dpi=200)
    first_page_img = pdfimages[0]

    first_page_img.save(pngfile, format="png")
