import fitz  # PyMuPDF
import os

# Get the current working directory
base_folder = os.path.dirname(os.path.abspath(__file__))

# Main folder to hold all extracted images
main_output_folder = os.path.join(base_folder, "extracted_images")
os.makedirs(main_output_folder, exist_ok=True)

# Ask the user for the PDF file name
pdf_name = input("Enter the PDF file name (e.g. document.pdf): ").strip()
pdf_path = os.path.join(base_folder, pdf_name)

# Check if file exists
if not os.path.exists(pdf_path):
    print(f"❌ '{pdf_name}' not found in the current directory!")
    exit()

# Get a base name without extension for output subfolder
pdf_base_name = os.path.splitext(pdf_name)[0]

# Create a dedicated output folder for this PDF's images
output_folder = os.path.join(main_output_folder, f"{pdf_base_name}_images")
os.makedirs(output_folder, exist_ok=True)

# Open the PDF
pdf_doc = fitz.open(pdf_path)
print(f"✅ Extracting images from {pdf_name} into '{output_folder}'...")

# Extract images
image_count = 0
for page_num in range(len(pdf_doc)):
    page = pdf_doc[page_num]
    img_list = page.get_images(full=True)

    for img_index, img in enumerate(img_list):
        xref = img[0]
        base_image = pdf_doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        output_path = os.path.join(
            output_folder,
            f"{pdf_base_name}_page{page_num+1}_{img_index+1}.{image_ext}"
        )
        with open(output_path, "wb") as img_file:
            img_file.write(image_bytes)
        image_count += 1

pdf_doc.close()

print(f"✅ Done! Extracted {image_count} image(s) to '{output_folder}'")
