"""
Extract page images from a downloaded PDF for annotation.

Requires: pip install Pillow pdf2image
Also requires poppler-utils: apt-get install poppler-utils (Linux)
                              brew install poppler (macOS)

Usage:
    python extract_pdf_pages.py --input ./data/neurosurgery/raw/cushing_pituitary.pdf \
        --output-dir ./data/neurosurgery/raw/cushing_pituitary_pages \
        --pages 50-150 --dpi 200
"""

import argparse
import os


def extract_pages(input_pdf, output_dir, pages=None, dpi=200, fmt="jpg"):
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("Error: pdf2image is required. Install with: pip install pdf2image")
        print("Also install poppler-utils:")
        print("  Linux:  apt-get install poppler-utils")
        print("  macOS:  brew install poppler")
        return

    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(input_pdf))[0]

    print(f"Extracting pages from: {input_pdf}")
    print(f"Output directory: {output_dir}")
    print(f"DPI: {dpi}")

    kwargs = {"dpi": dpi, "fmt": fmt}
    if pages:
        kwargs["first_page"] = pages[0]
        kwargs["last_page"] = pages[-1]

    images = convert_from_path(input_pdf, **kwargs)

    first_page = pages[0] if pages else 1
    for i, image in enumerate(images):
        page_num = first_page + i
        if pages and page_num not in pages:
            continue
        filename = f"{basename}_page_{page_num:04d}.{fmt}"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        print(f"  Saved: {filename}")

    print(f"\nDone! Extracted {len(images)} pages to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract page images from a PDF for annotation."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input PDF file"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save extracted page images",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page range to extract, e.g. '50-150' (1-indexed). "
        "Default: all pages.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution for extracted images (default: 200)",
    )
    parser.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format (default: jpg)",
    )

    args = parser.parse_args()

    pages = None
    if args.pages:
        parts = args.pages.split("-")
        if len(parts) == 2:
            pages = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            pages = [int(p) for p in args.pages.split(",")]

    extract_pages(args.input, args.output_dir, pages, args.dpi, args.format)


if __name__ == "__main__":
    main()
