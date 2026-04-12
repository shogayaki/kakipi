"""
Download public domain neurosurgery images from Internet Archive.

These books are in the public domain (published before 1928):
- Harvey Cushing, "The Pituitary Body and Its Disorders" (1912) - 319 illustrations
- Harvey Cushing, "Tumors of the Nervus Acusticus" (1917)
- Fedor Krause, "Surgery of the Brain and Spinal Cord" (1909-1912)

Usage:
    python download_neurosurgery_images.py --output-dir ./data/neurosurgery/raw
    python download_neurosurgery_images.py --output-dir ./data/neurosurgery/raw --book cushing_pituitary
    python download_neurosurgery_images.py --output-dir ./data/neurosurgery/raw --book krause_v1 --pages 50-100
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
import json
import time

# Public domain neurosurgery books on Internet Archive
BOOKS = {
    "cushing_pituitary": {
        "identifier": "pituitarybodyits00cushuoft",
        "title": "The Pituitary Body and Its Disorders (Cushing, 1912)",
        "description": "Harvey Cushing's seminal work on pituitary disorders. "
        "Contains 319 illustrations of brain anatomy, surgical procedures, "
        "and pathological findings.",
        "year": 1912,
    },
    "cushing_pituitary_google": {
        "identifier": "pituitarybodyan01cushgoog",
        "title": "The Pituitary Body and Its Disorders - Google scan (Cushing, 1912)",
        "description": "Google-digitized version of Cushing's pituitary work. "
        "Higher quality scans from University of California collection.",
        "year": 1912,
    },
    "cushing_acoustic": {
        "identifier": "cu31924032537403",
        "title": "Tumors of the Nervus Acusticus (Cushing, 1917)",
        "description": "Cushing's work on acoustic nerve tumors. "
        "Contains detailed surgical illustrations and anatomical plates.",
        "year": 1917,
    },
    "krause_v1": {
        "identifier": "b21272116_001",
        "title": "Surgery of the Brain and Spinal Cord, Vol.1 (Krause, 1909)",
        "description": "Fedor Krause's comprehensive neurosurgery textbook, Volume 1. "
        "Rich in anatomical illustrations and operative photography.",
        "year": 1909,
    },
    "krause_v2": {
        "identifier": "surgeryofbrainsp02krauuoft",
        "title": "Surgery of the Brain and Spinal Cord, Vol.2 (Krause, 1912)",
        "description": "Fedor Krause's comprehensive neurosurgery textbook, Volume 2.",
        "year": 1912,
    },
    "krause_v3": {
        "identifier": "surgerybrainand01kraugoog",
        "title": "Surgery of the Brain and Spinal Cord, Vol.3 (Krause, 1912)",
        "description": "Fedor Krause's comprehensive neurosurgery textbook, Volume 3.",
        "year": 1912,
    },
}


def get_book_metadata(identifier):
    """Fetch metadata for an Internet Archive item."""
    url = f"https://archive.org/metadata/{identifier}"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "kakipi-neurosurgery-downloader/1.0")
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Error fetching metadata: {e}")
        return None


def get_page_count(metadata):
    """Get the total number of pages from metadata."""
    if metadata is None:
        return None
    files = metadata.get("files", [])
    # Look for the JP2 zip or scandata to determine page count
    for f in files:
        name = f.get("name", "")
        if name.endswith("_scandata.xml"):
            # Could parse this for exact count
            pass
        if name.endswith("_jp2.zip"):
            return f.get("name")
    # Try to get from metadata
    imagecount = metadata.get("metadata", {}).get("imagecount")
    if imagecount:
        return int(imagecount)
    return None


def download_page_image(identifier, page_num, output_path, scale=4):
    """
    Download a single page image from Internet Archive.

    Uses the IIIF Image API or the page image endpoint.
    Scale: 1=full, 2=1/2, 4=1/4, 8=1/8
    """
    # Internet Archive page image URL
    url = f"https://archive.org/download/{identifier}/page/n{page_num}_w800.jpg"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "kakipi-neurosurgery-downloader/1.0")
        with urllib.request.urlopen(req, timeout=60) as response:
            with open(output_path, "wb") as f:
                f.write(response.read())
        return True
    except urllib.error.HTTPError:
        # Try alternative URL format
        alt_url = (
            f"https://archive.org/services/img/{identifier}/page/n{page_num}_w800"
        )
        try:
            req = urllib.request.Request(alt_url)
            req.add_header("User-Agent", "kakipi-neurosurgery-downloader/1.0")
            with urllib.request.urlopen(req, timeout=60) as response:
                with open(output_path, "wb") as f:
                    f.write(response.read())
            return True
        except urllib.error.URLError:
            return False
    except urllib.error.URLError as e:
        print(f"  Error downloading page {page_num}: {e}")
        return False


def download_book_pdf(identifier, output_path):
    """Download the full PDF of a book from Internet Archive."""
    url = f"https://archive.org/download/{identifier}/{identifier}.pdf"
    print(f"Downloading PDF from: {url}")
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "kakipi-neurosurgery-downloader/1.0")
        with urllib.request.urlopen(req, timeout=300) as response:
            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        print(f"  Saved: {output_path}")
        return True
    except urllib.error.URLError as e:
        print(f"  Error downloading PDF: {e}")
        return False


def parse_page_range(page_range_str, max_pages=None):
    """Parse a page range string like '10-50' or '10,20,30' or '10-50,60-70'."""
    pages = []
    for part in page_range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            if max_pages:
                end = min(end, max_pages - 1)
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def list_books():
    """Print available books."""
    print("\nAvailable public domain neurosurgery books:\n")
    print(f"{'Key':<30} {'Year':<6} Title")
    print("-" * 80)
    for key, info in BOOKS.items():
        print(f"{key:<30} {info['year']:<6} {info['title']}")
        print(f"{'':>30}        {info['description'][:70]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download public domain neurosurgery images from Internet Archive."
    )
    parser.add_argument(
        "--output-dir",
        default="./data/neurosurgery/raw",
        help="Directory to save downloaded images (default: ./data/neurosurgery/raw)",
    )
    parser.add_argument(
        "--book",
        choices=list(BOOKS.keys()),
        default="cushing_pituitary",
        help="Book to download images from (default: cushing_pituitary)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page range to download, e.g. '50-100' or '10,20,30-40' "
        "(default: illustration pages only)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Download the full PDF instead of individual page images",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available books and exit",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between downloads in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    if args.list:
        list_books()
        return

    book = BOOKS[args.book]
    identifier = book["identifier"]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nBook: {book['title']}")
    print(f"Internet Archive ID: {identifier}")
    print(f"URL: https://archive.org/details/{identifier}")
    print(f"Output directory: {args.output_dir}\n")

    if args.pdf:
        pdf_path = os.path.join(args.output_dir, f"{args.book}.pdf")
        download_book_pdf(identifier, pdf_path)
        print(
            "\nTo extract images from the PDF, use:"
            f"\n  python extract_pdf_pages.py --input {pdf_path} "
            f"--output-dir {args.output_dir}/{args.book}_pages"
        )
        return

    # Fetch metadata to determine page count
    print("Fetching book metadata...")
    metadata = get_book_metadata(identifier)
    total_pages = get_page_count(metadata)
    if total_pages:
        print(f"Total pages: {total_pages}")

    # Determine pages to download
    if args.pages:
        pages = parse_page_range(args.pages, total_pages)
    else:
        # Default: download a representative sample of pages likely to contain
        # illustrations (typically scattered throughout medical textbooks)
        if total_pages:
            # Sample pages across the book, focusing on illustration-heavy sections
            step = max(1, total_pages // 50)
            pages = list(range(0, total_pages, step))[:50]
        else:
            # Default range if we can't determine total pages
            pages = list(range(0, 100))

    print(f"Downloading {len(pages)} page images...")
    print(f"Delay between downloads: {args.delay}s\n")

    downloaded = 0
    failed = 0
    for i, page_num in enumerate(pages):
        filename = f"{args.book}_page_{page_num:04d}.jpg"
        output_path = os.path.join(args.output_dir, filename)

        if os.path.exists(output_path):
            print(f"  [{i + 1}/{len(pages)}] Skip (exists): {filename}")
            downloaded += 1
            continue

        print(f"  [{i + 1}/{len(pages)}] Downloading page {page_num}...", end="")
        success = download_page_image(identifier, page_num, output_path)
        if success:
            print(f" OK -> {filename}")
            downloaded += 1
        else:
            print(" FAILED")
            failed += 1

        if i < len(pages) - 1:
            time.sleep(args.delay)

    print(f"\nDone! Downloaded: {downloaded}, Failed: {failed}")
    print(f"Images saved to: {args.output_dir}")
    print(
        "\nNext steps:"
        "\n  1. Review downloaded images and select ones containing anatomical/surgical content"
        "\n  2. Resize images:  python resize_images.py --raw-dir ./data/neurosurgery/raw "
        "--save-dir ./data/neurosurgery/images --ext jpg"
        "\n  3. Split into train/test directories"
        "\n  4. Annotate with labelImg using the neurosurgery label classes"
    )


if __name__ == "__main__":
    main()
