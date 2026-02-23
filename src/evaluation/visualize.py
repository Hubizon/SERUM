import os
import random
from typing import Tuple, List

import torch
import torchvision.transforms as T
from PIL import Image
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import Paragraph, Frame

from ..utils.config import Config
from ..data.augmentation import Augment


def generate_comparison_grid(
        config: Config,
        clean_path: str = None,
        watermarked_path: str = None,
        output_path: str = None,
        num_rows: int = 3,
        num_columns: int = 9,
        image_size: Tuple[int, int] = (128, 128),
        text_size: int = 20,
        pair_gap: int = 20,
        save_image: bool = True,
        random_seed: int = None,
) -> Image.Image:
    """
    Generates and returns a comparison grid of clean and watermarked images.

    PNG: only images, no text (better for figure quality).
    PDF: images + vector text headers ("Clean" / "Latent Mixture") in selectable bold font.
    """
    if clean_path is None:
        clean_path = config.evaluation.clean_path
    if watermarked_path is None:
        watermarked_path = config.evaluation.watermarked_path
    if random_seed is None:
        random_seed = config.evaluation.random_seed
    if output_path is None and save_image:
        output_path = os.path.join(config.evaluation.eval_output_path, "comparison_grid.png")

    random.seed(random_seed)

    clean_files = sorted(os.listdir(clean_path))
    watermarked_files = sorted(os.listdir(watermarked_path))

    if len(clean_files) != len(watermarked_files):
        raise ValueError(
            f"Directories must have the same number of files. "
            f"Found {len(clean_files)} clean vs {len(watermarked_files)} watermarked."
        )

    num_available_pairs = len(clean_files)
    num_samples_needed = num_rows * num_columns
    if num_available_pairs < num_samples_needed:
        raise ValueError(f"Need {num_samples_needed} pairs, found {num_available_pairs}.")

    selected_indices = random.sample(range(num_available_pairs), num_samples_needed)
    selected_indices[7] = 104

    # Layout
    padding = 10
    header_height = 50
    img_w, img_h = image_size

    canvas_width = padding + (num_columns * (2 * img_w + 2 * padding)) + ((num_columns - 1) * pair_gap)
    canvas_height = header_height + (num_rows * (img_h + padding)) + padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # Paste images
    for i, pair_index in enumerate(selected_indices):
        clean_filename = clean_files[pair_index]
        wmark_filename = watermarked_files[pair_index]

        row, col_pair = divmod(i, num_columns)
        y_pos = header_height + padding + (row * (img_h + padding))

        pair_offset = col_pair * pair_gap
        clean_x = padding + pair_offset + (col_pair * 2 * (img_w + padding))
        wmark_x = clean_x + img_w + padding

        clean_img = Image.open(os.path.join(clean_path, clean_filename)).resize(image_size)
        wmark_img = Image.open(os.path.join(watermarked_path, wmark_filename)).resize(image_size)

        canvas.paste(clean_img, (clean_x, y_pos))
        canvas.paste(wmark_img, (wmark_x, y_pos))

    if save_image:
        if not output_path:
            raise ValueError("output_path must be provided when save_image is True.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        png_path = output_path if output_path.endswith(".png") else output_path.rsplit(".", 1)[0] + ".png"
        canvas.save(png_path, "PNG")
        print(f"Comparison grid saved as PNG to: {png_path}")

        # Generate PDF with vector text
        pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
        c = pdf_canvas.Canvas(pdf_path, pagesize=(canvas_width, canvas_height))

        # Place the PNG image (covers the image area; header area is part of the page above the images)
        c.drawImage(png_path, 0, 0, width=canvas_width, height=canvas_height)

        # --- Manual wrap + centered drawing (robust) ---
        font_name = "Helvetica-Bold"
        c.setFont(font_name, text_size)
        inner_margin = 6  # small horizontal padding inside the image-column for text
        leading = text_size + 2  # vertical spacing between lines

        def wrap_text_to_width_split(text: str, max_width: float):
            """Wrap text to fit within specified width."""
            words = text.split()
            lines = []
            cur = ""
            for w in words:
                candidate = w if cur == "" else f"{cur} {w}"
                if c.stringWidth(candidate, font_name, text_size) <= max_width - 2 * inner_margin:
                    cur = candidate
                else:
                    if cur == "":
                        # a single long word: split characters
                        part = ""
                        for ch in w:
                            if c.stringWidth(part + ch, font_name, text_size) <= max_width - 2 * inner_margin:
                                part += ch
                            else:
                                if part:
                                    lines.append(part)
                                part = ch
                        if part:
                            cur = part
                        else:
                            cur = ""
                    else:
                        lines.append(cur)
                        cur = w
            if cur:
                lines.append(cur)
            return lines

        # center Y of header area **including** the top image padding
        # This centers the text in the vertical space from top of page down to the top of the first image (header + padding)
        header_center_y = canvas_height - ((header_height + padding) / 2.0)  # <<--- changed line

        for i in range(num_columns):
            pair_offset = i * pair_gap
            clean_x = padding + pair_offset + (i * 2 * (img_w + padding))
            wmark_x = clean_x + img_w + padding

            for x_left, label in [(clean_x, "Clean"), (wmark_x, "Ours")]:
                max_w = img_w
                lines = wrap_text_to_width_split(label, max_w)
                if not lines:
                    continue

                # compute baseline for top line so the text block is vertically centered
                n_lines = len(lines)
                top_baseline = header_center_y + ((n_lines - 1) * leading) / 2.0

                center_x = x_left + (img_w / 2.0)
                for idx, line in enumerate(lines):
                    baseline_y = top_baseline - idx * leading
                    c.drawCentredString(center_x, baseline_y, line)

        c.save()
        print(f"Comparison grid saved as PDF to: {pdf_path}")

    return canvas


def wrap_text_to_width(canvas_obj, text: str, max_width: float, text_size: float):
        """Wrap text to fit within specified width using canvas metrics."""
        words = text.split()
        lines = []
        cur = ""
        for w in words:
            candidate = w if not cur else f"{cur} {w}"
            if canvas_obj.stringWidth(candidate, "Helvetica-Bold", text_size) <= max_width:
                cur = candidate
            else:
                if cur: lines.append(cur)
                cur = w
        if cur: lines.append(cur)
        return lines


def generate_augmentation_grid(
        config: Config,
        image_path: str = None,
        image_indices: List[int] = None,
        num_columns: int = 6,
        output_path: str = None,
        image_size: Tuple[int, int] = (128, 128),
        text_size: int = 20,
        label_with_prompts: bool = False,
        positive_prompts_path: str = None,
        save_image: bool = True,
        random_seed: int = None,
) -> Image.Image:
    """
    Generates a grid comparing a clean image to various augmented versions.

    - Rows: Clean image + 8 augmentations.
    - Columns: Different source images.
    - Labels: Augmentation names are vertical on the left (can be multiline). Image prompts are horizontal at the top.
    - Output: Saves a high-quality PNG and a paper-ready PDF with vector text.
    """
    if image_path is None:
        image_path = config.evaluation.clean_path
    if output_path is None and save_image:
        output_path = os.path.join(config.evaluation.eval_output_path, "augmentation_grid.png")
    if random_seed is None:
        random_seed = config.evaluation.random_seed

    random.seed(random_seed)
    image_files = sorted([f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        raise ValueError(f"No images found in path: {image_path}")

    if image_indices is None:
        num_to_sample = min(num_columns, len(image_files))
        image_indices = random.sample(range(len(image_files)), num_to_sample)
        print("Randomly selected image indices:", image_indices)
    else:
        num_columns = len(image_indices)

    # --- Layout Calculation ---
    row_label_width = 80
    padding = 10
    img_w, img_h = image_size
    num_augmentations = 8
    num_rows = 1 + num_augmentations

    

    # --- Prompt Handling & Header Height ---
    prompts = []
    top_header_height = 10
    if label_with_prompts:
        if positive_prompts_path is None:
            positive_prompts_path = config.evaluation.positive_prompts_path
        with open(positive_prompts_path, "r") as f:
            all_prompts = f.read().splitlines()

        temp_canvas = pdf_canvas.Canvas("temp.pdf")
        temp_canvas.setFont("Helvetica-Bold", text_size)

        max_prompt_height = 0
        for i in image_indices:
            prompt = all_prompts[i] if i < len(all_prompts) else f"Prompt {i} missing"
            prompts.append(prompt)
            lines = wrap_text_to_width(temp_canvas, prompt, img_w - 2 * padding, text_size)
            prompt_height = len(lines) * (text_size + 2)
            if prompt_height > max_prompt_height:
                max_prompt_height = prompt_height
        top_header_height = max_prompt_height + 2 * padding
        del temp_canvas

    # --- Canvas Size ---
    canvas_width = row_label_width + (num_columns * (img_w + padding)) + padding
    canvas_height = top_header_height + (num_rows * (img_h + padding)) + padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # --- Augmentation and Image Pasting ---
    augment = Augment()

    def _get_augmentation_name(transform):
        """Extract augmentation name from transform class."""
        name = transform.__class__.__name__
        return {
            "FullRotate": "Rotation",
            "Compose": "JPEG\nCompression",
            "RandomResizedCrop": "Resized\nCrop",
            "RandomDrop": "Patch\nDrop",
            "SaltAndPepperNoise": "Salt &\nPepper",
            "AddGaussianNoise": "Gaussian\nNoise",
            "ColorJitter": "Brightness",
            "MedianFilterBlur": "Median\nBlur",
            "RandomRotation": "Random\nRotation",
            "GaussianBlur": "Gaussian\nBlur",
        }.get(name, name)

    row_labels = ["Clean"] + [_get_augmentation_name(t) for t in augment.transforms[:num_augmentations]]

    for col_index, image_file_index in enumerate(image_indices):
        image_filename = image_files[image_file_index]
        img_pil = Image.open(os.path.join(image_path, image_filename)).convert("RGB").resize(image_size)
        img_tensor = T.ToTensor()(img_pil).unsqueeze(0) * 2 - 1

        x_pos = row_label_width + padding + (col_index * (img_w + padding))

        # Row 0: Clean image
        y_pos_clean = top_header_height + padding
        canvas.paste(img_pil, (x_pos, y_pos_clean))

        # Rows 1+: Augmented images
        for aug_row_idx in range(num_augmentations):
            y_pos_aug = top_header_height + padding + ((aug_row_idx + 1) * (img_h + padding))
            augmented_tensor = augment(img_tensor, idx=aug_row_idx)
            augmented_pil = T.ToPILImage()((augmented_tensor.squeeze(0).clamp(-1, 1) + 1) / 2)
            canvas.paste(augmented_pil.resize(image_size), (x_pos, y_pos_aug))

    if save_image:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        png_path = output_path if output_path.endswith(".png") else f"{os.path.splitext(output_path)[0]}.png"
        canvas.save(png_path, "PNG")
        print(f"Augmentation grid saved as PNG to: {png_path}")

        pdf_path = f"{os.path.splitext(output_path)[0]}.pdf"
        c = pdf_canvas.Canvas(pdf_path, pagesize=(canvas_width, canvas_height))
        c.drawImage(png_path, 0, 0, width=canvas_width, height=canvas_height)
        c.setFont("Helvetica-Bold", text_size)

        # Draw HORIZONTAL prompt labels (at the top)
        if label_with_prompts:
            for col_index, prompt in enumerate(prompts):
                lines = wrap_text_to_width(c, prompt, img_w - 2 * padding, text_size)
                line_height = text_size + 2
                total_text_height = len(lines) * line_height

                start_y = canvas_height - (top_header_height / 2) + (total_text_height / 2) - line_height
                col_center_x = row_label_width + padding + (col_index * (img_w + padding)) + (img_w / 2)

                for i, line in enumerate(lines):
                    c.drawCentredString(col_center_x, start_y - i * line_height, line)

        # Draw VERTICAL augmentation labels (on the left)
        for row_index, label in enumerate(row_labels):
            img_y_pil = top_header_height + padding + (row_index * (img_h + padding))
            img_center_y_pil = img_y_pil + img_h / 2
            y_pdf_coord = canvas_height - img_center_y_pil
            x_pdf_coord = row_label_width / 2

            c.saveState()
            c.translate(x_pdf_coord, y_pdf_coord)
            c.rotate(90)

            # Handle multiline vertical text
            lines = label.split('\n')
            line_height = text_size + 2
            total_text_height = (len(lines) - 1) * line_height
            start_y_offset = total_text_height / 2.0

            for i, line in enumerate(lines):
                c.drawCentredString(0, start_y_offset - i * line_height, line)

            c.restoreState()

        c.save()
        print(f"Augmentation grid saved as PDF to: {pdf_path}")

    return canvas


def generate_small_comparison_grid(
        config: Config,
        clean_path: str = None,
        watermarked_path: str = None,
        output_path: str = None,
        prompts: List[str] = None,
        image_indices: List[int] = None,
        num_columns: int = 9,
        image_size: Tuple[int, int] = (128, 128),
        text_size: int = 20,
        save_image: bool = True,
        random_seed: int = None,
) -> Image.Image:
    """
    Generates a 2-row comparison grid of clean and watermarked images with vertical labels.

    - Rows: "Clean" and "Ours" (watermarked).
    - Columns: Different source images, selectable by index or randomly sampled.
    - Labels: Row names are vertical on the left. Prompts can be displayed above columns.
    - Output: Saves a high-quality PNG and a paper-ready PDF with vector text.
    """
    if clean_path is None:
        clean_path = config.evaluation.clean_path
    if watermarked_path is None:
        watermarked_path = config.evaluation.watermarked_path
    if output_path is None and save_image:
        output_path = os.path.join(config.evaluation.eval_output_path, "small_comparison_grid.png")
    if random_seed is None:
        random_seed = config.evaluation.random_seed

    random.seed(random_seed)

    clean_files = sorted(os.listdir(clean_path))
    watermarked_files = sorted(os.listdir(watermarked_path))

    if len(clean_files) != len(watermarked_files):
        raise ValueError(
            f"Directories must have the same number of files. "
            f"Found {len(clean_files)} clean vs {len(watermarked_files)} watermarked."
        )

    num_available_pairs = len(clean_files)

    if image_indices is None:
        if num_available_pairs < num_columns:
            raise ValueError(f"Need {num_columns} pairs, but only {num_available_pairs} are available.")
        image_indices = random.sample(range(num_available_pairs), num_columns)
        print("Randomly selected image indices:", image_indices)
    else:
        num_columns = len(image_indices)

    # --- Layout Calculation ---
    row_label_width = 80
    padding = 10
    img_w, img_h = image_size
    num_rows = 2


    # --- Prompt Handling & Header Height ---
    top_header_height = 0
    if prompts:
        if len(prompts) != num_columns:
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of columns ({num_columns}).")

        temp_canvas = pdf_canvas.Canvas("temp.pdf")
        temp_canvas.setFont("Helvetica-Bold", text_size)

        max_prompt_height = 0
        for prompt in prompts:
            lines = wrap_text_to_width(temp_canvas, prompt, img_w - 2 * padding, text_size)
            prompt_height = len(lines) * (text_size + 2)
            if prompt_height > max_prompt_height:
                max_prompt_height = prompt_height
        top_header_height = max_prompt_height + 2 * padding
        del temp_canvas

    # --- Canvas Size ---
    canvas_width = row_label_width + (num_columns * (img_w + padding)) + padding
    canvas_height = top_header_height + (num_rows * (img_h + padding)) + padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # --- Image Pasting ---
    row_labels = ["Clean", "Ours"]

    for col_index, pair_index in enumerate(image_indices):
        clean_filename = clean_files[pair_index]
        wmark_filename = watermarked_files[pair_index]

        clean_img = Image.open(os.path.join(clean_path, clean_filename)).resize(image_size)
        wmark_img = Image.open(os.path.join(watermarked_path, wmark_filename)).resize(image_size)

        x_pos = row_label_width + padding + (col_index * (img_w + padding))

        # Row 0: Clean image
        y_pos_clean = top_header_height + padding
        canvas.paste(clean_img, (x_pos, y_pos_clean))

        # Row 1: Watermarked image ("Ours")
        y_pos_wmark = top_header_height + padding + (img_h + padding)
        canvas.paste(wmark_img, (x_pos, y_pos_wmark))

    if save_image:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        png_path = output_path if output_path.endswith(".png") else f"{os.path.splitext(output_path)[0]}.png"
        canvas.save(png_path, "PNG")
        print(f"Small comparison grid saved as PNG to: {png_path}")

        pdf_path = f"{os.path.splitext(output_path)[0]}.pdf"
        c = pdf_canvas.Canvas(pdf_path, pagesize=(canvas_width, canvas_height))
        c.drawImage(png_path, 0, 0, width=canvas_width, height=canvas_height)
        c.setFont("Helvetica-Bold", text_size)

        # Draw HORIZONTAL prompt labels (at the top)
        if prompts:
            for col_index, prompt in enumerate(prompts):
                lines = wrap_text_to_width(c, prompt, img_w - 2 * padding, text_size)
                line_height = text_size + 2
                total_text_height = len(lines) * line_height

                start_y = canvas_height - (top_header_height / 2) + (total_text_height / 2) - line_height
                col_center_x = row_label_width + padding + (col_index * (img_w + padding)) + (img_w / 2)

                for i, line in enumerate(lines):
                    c.drawCentredString(col_center_x, start_y - i * line_height, line)

        # Draw VERTICAL row labels (on the left)
        for row_index, label in enumerate(row_labels):
            img_y_pil = top_header_height + padding + (row_index * (img_h + padding))
            img_center_y_pil = img_y_pil + img_h / 2
            y_pdf_coord = canvas_height - img_center_y_pil
            x_pdf_coord = row_label_width / 2

            c.saveState()
            c.translate(x_pdf_coord, y_pdf_coord)
            c.rotate(90)
            c.drawCentredString(0, 0, label)
            c.restoreState()

        c.save()
        print(f"Small comparison grid saved as PDF to: {pdf_path}")

    return canvas


def generate_small_augmentation_grid(
    config: Config,
    augmentation_paths: List[str],
    augmentation_labels: List[str],
    output_path: str = None,
    image_indices: List[int] = None,
    num_rows: int = 5,
    image_size: Tuple[int, int] = (128, 128),
    text_size: int = 20,
    save_image: bool = True,
    random_seed: int = None,
) -> Image.Image:
    """
    Generates a comparison grid of different augmentations.

    - Columns: Different augmentation types, with labels at the top.
    - Rows: Different image samples for each augmentation.
    - Output: Saves a high-quality PNG and a paper-ready PDF with vector text.
    """
    if output_path is None and save_image:
        output_path = os.path.join(config.evaluation.eval_output_path, "small_augmentation_grid.png")
    if random_seed is None:
        random_seed = config.evaluation.random_seed

    random.seed(random_seed)

    if len(augmentation_paths) != len(augmentation_labels):
        raise ValueError("Length of augmentation_paths and augmentation_labels must be the same.")

    num_columns = len(augmentation_paths)
    all_image_files = [sorted(os.listdir(p)) for p in augmentation_paths]

    if image_indices:
        num_rows = len(image_indices)
        # Check if all indices are valid for all augmentation folders
        for i, files in enumerate(all_image_files):
            if any(idx >= len(files) for idx in image_indices):
                raise ValueError(f"image_index out of bounds for augmentation path: {augmentation_paths[i]}")
    else:
        # Check if there are enough images to sample from in all folders
        min_files = min(len(f) for f in all_image_files)
        if min_files < num_rows:
            raise ValueError(f"Cannot sample {num_rows} images; one folder has only {min_files} files.")
        # Sample indices that will be used for all augmentation types
        image_indices = random.sample(range(min_files), num_rows)
        print(f"Randomly selected {num_rows} image indices: {image_indices}")

    # --- Layout Calculation ---
    padding = 10
    img_w, img_h = image_size

    # --- Header Height Calculation (similar to prompt handling) ---
    top_header_height = 0
    temp_canvas = pdf_canvas.Canvas("temp.pdf")
    temp_canvas.setFont("Helvetica-Bold", text_size)

    max_label_height = 0
    for label in augmentation_labels:
        lines = wrap_text_to_width(temp_canvas, label, img_w - 2 * padding, text_size)
        label_height = len(lines) * (text_size + 2)
        if label_height > max_label_height:
            max_label_height = label_height
    top_header_height = max_label_height + 2 * padding
    del temp_canvas

    # --- Canvas Size ---
    canvas_width = (num_columns * (img_w + padding)) + padding
    canvas_height = top_header_height + (num_rows * (img_h + padding)) + padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # --- Image Pasting ---
    for col_idx, (path, files) in enumerate(zip(augmentation_paths, all_image_files)):
        x_pos = padding + (col_idx * (img_w + padding))
        for row_idx, img_idx in enumerate(image_indices):
            y_pos = top_header_height + padding + (row_idx * (img_h + padding))
            img_path = os.path.join(path, files[img_idx])
            img = Image.open(img_path).resize(image_size)
            canvas.paste(img, (x_pos, y_pos))

    if save_image:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        png_path = output_path if output_path.endswith(".png") else f"{os.path.splitext(output_path)[0]}.png"
        canvas.save(png_path, "PNG")
        print(f"Small augmentation grid saved as PNG to: {png_path}")

        pdf_path = f"{os.path.splitext(output_path)[0]}.pdf"
        c = pdf_canvas.Canvas(pdf_path, pagesize=(canvas_width, canvas_height))
        c.drawImage(png_path, 0, 0, width=canvas_width, height=canvas_height)
        c.setFont("Helvetica-Bold", text_size)

        # Draw HORIZONTAL column labels (at the top)
        for col_idx, label in enumerate(augmentation_labels):
            lines = wrap_text_to_width(c, label, img_w - 2 * padding, text_size)
            line_height = text_size + 2
            total_text_height = len(lines) * line_height

            start_y = canvas_height - (top_header_height / 2) + (total_text_height / 2) - line_height
            col_center_x = padding + (col_idx * (img_w + padding)) + (img_w / 2)

            for i, line in enumerate(lines):
                c.drawCentredString(col_center_x, start_y - i * line_height, line)

        c.save()
        print(f"Small augmentation grid saved as PDF to: {pdf_path}")

    return canvas
