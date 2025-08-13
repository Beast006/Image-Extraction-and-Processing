import os
import fitz  # PyMuPDF
import torch
import clip
from PIL import Image
import shutil

# ------------------ CLIP CLASSIFICATION CONFIG ------------------ #
categories = {
    "logo": ["tech logo", "company logo", "symbol"],
    "nature": ["forest", "beach", "mountain", "river", "sunset"],
    "person": ["person smiling", "person sad", "person neutral"],
    "animal": ["dog", "cat", "bird", "wild animal"],
    "building": ["skyscraper", "house", "temple", "bridge"],
    "chart": ["bar chart", "pie chart", "line graph"]
}

UNCERTAIN_THRESHOLD = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text_inputs_main = torch.cat([
    clip.tokenize(f"a photo of a {c}") for c in categories.keys()
]).to(device)

subcat_map = {}
subcat_labels = []
for cat, subs in categories.items():
    for sub in subs:
        subcat_labels.append(sub)
        subcat_map[sub] = cat

text_inputs_sub = torch.cat([
    clip.tokenize(f"a photo of a {s}") for s in subcat_labels
]).to(device)


def classify_images_in_folder(image_folder):
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)

        if os.path.isdir(file_path) or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            image = Image.open(file_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)

                logits_main = image_features @ model.encode_text(text_inputs_main).T
                probs_main = logits_main.softmax(dim=-1).cpu().numpy().flatten()
                max_prob_main = probs_main.max()
                predicted_main_index = probs_main.argmax()

                if max_prob_main < UNCERTAIN_THRESHOLD:
                    predicted_main_category = "uncertain"
                    dest_folder = os.path.join(image_folder, predicted_main_category)
                    predicted_subcategory = ""
                else:
                    predicted_main_category = list(categories.keys())[predicted_main_index]

                    logits_sub = image_features @ model.encode_text(text_inputs_sub).T
                    probs_sub = logits_sub.softmax(dim=-1).cpu().numpy().flatten()
                    sub_scores = [
                        (label, prob) for label, prob in zip(subcat_labels, probs_sub)
                        if subcat_map[label] == predicted_main_category
                    ]

                    if sub_scores:
                        sub_scores.sort(key=lambda x: x[1], reverse=True)
                        predicted_subcategory = sub_scores[0][0]
                    else:
                        predicted_subcategory = "general"

                    dest_folder = os.path.join(image_folder, predicted_main_category, predicted_subcategory)

            print(f"{filename} -> {predicted_main_category} / {predicted_subcategory} ({max_prob_main:.2f})")
            os.makedirs(dest_folder, exist_ok=True)
            shutil.move(file_path, os.path.join(dest_folder, filename))

        except Exception as e:
            print(f"Could not classify '{filename}': {e}")


# ------------------ PDF IMAGE EXTRACTION ------------------ #
def extract_images_from_pdf(pdf_path, main_output_folder):
    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(main_output_folder, f"{pdf_base_name}_images")
    os.makedirs(output_folder, exist_ok=True)

    pdf_doc = fitz.open(pdf_path)
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
    print(f"Extracted {image_count} images from {pdf_base_name}")
    return output_folder if image_count > 0 else None


# ------------------ MAIN SCRIPT ------------------ #
if __name__ == "__main__":
    pdf_folder = input("Enter the path to the folder containing PDFs: ").strip()
    if not os.path.isdir(pdf_folder):
        print("The path you entered is not a valid folder.")
        exit()

    output_base_folder = input("Enter the path where extracted images should be saved: ").strip()
    if not os.path.isdir(output_base_folder):
        print("The output path you entered is not a valid folder.")
        exit()

    main_output_folder = os.path.join(output_base_folder, "extracted_images")
    os.makedirs(main_output_folder, exist_ok=True)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"Processing {pdf_file}...")
            extracted_folder = extract_images_from_pdf(pdf_path, main_output_folder)

            if extracted_folder:
                classify_images_in_folder(extracted_folder)

    print("All PDFs processed and images classified.")
