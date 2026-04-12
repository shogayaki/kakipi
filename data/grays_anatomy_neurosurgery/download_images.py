#!/usr/bin/env python3
"""
Download neurosurgery-related plates from Gray's Anatomy 20th edition (1918).

All images are from Wikimedia Commons and are in the public domain.
Source: Henry Gray, Anatomy of the Human Body, 20th U.S. edition, 1918.

Usage:
    python download_images.py [--output-dir DIR]

Alternative (wget):
    wget -i urls.txt -P images/

Alternative (curl):
    xargs -n1 -P4 curl -O < urls.txt
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error

# Neurosurgery-related plates from Gray's Anatomy (1918), 20th edition
# Organized by anatomical category
PLATES = [
    # =========================================================================
    # Brain Blood Supply (Angiology)
    # =========================================================================
    {
        "filename": "Gray516.png",
        "figure": 516,
        "title": "The arteries of the base of the brain",
        "category": "brain_blood_supply",
        "description": "Circle of Willis and arterial supply to the base of the brain"
    },
    {
        "filename": "Gray517.png",
        "figure": 517,
        "title": "The internal carotid and vertebral arteries (right side)",
        "category": "brain_blood_supply",
        "description": "Internal carotid and vertebral arteries, right side view"
    },
    {
        "filename": "Gray518.png",
        "figure": 518,
        "title": "Diagram of the arterial circulation at the base of the brain",
        "category": "brain_blood_supply",
        "description": "Schematic diagram of arterial circulation at the base of the brain"
    },
    {
        "filename": "Gray519.png",
        "figure": 519,
        "title": "The middle meningeal artery",
        "category": "brain_blood_supply",
        "description": "Outer surface of the dura mater showing the middle meningeal artery"
    },
    {
        "filename": "Gray520.png",
        "figure": 520,
        "title": "The ophthalmic artery and its branches",
        "category": "brain_blood_supply",
        "description": "Ophthalmic artery and its branches"
    },
    {
        "filename": "Gray521.png",
        "figure": 521,
        "title": "The internal carotid artery (right side, cavernous portion)",
        "category": "brain_blood_supply",
        "description": "Internal carotid artery viewed from the right side"
    },

    # =========================================================================
    # Dural Venous Sinuses and Brain Veins
    # =========================================================================
    {
        "filename": "Gray488.png",
        "figure": 488,
        "title": "Transverse section of a thoracic vertebra (venous drainage reference)",
        "category": "brain_venous",
        "description": "Reference for spinal venous system"
    },
    {
        "filename": "Gray564.png",
        "figure": 564,
        "title": "Dural sinuses and veins of the brain (sagittal section)",
        "category": "brain_venous",
        "description": "Superior sagittal sinus and great cerebral vein"
    },
    {
        "filename": "Gray565.png",
        "figure": 565,
        "title": "The sinuses at the base of the skull",
        "category": "brain_venous",
        "description": "Dural venous sinuses at the base of the skull"
    },
    {
        "filename": "Gray566.png",
        "figure": 566,
        "title": "Cavernous sinus (cross-section)",
        "category": "brain_venous",
        "description": "Cross-section of the cavernous sinus showing internal carotid artery and cranial nerves"
    },
    {
        "filename": "Gray567.png",
        "figure": 567,
        "title": "Veins of the diploë",
        "category": "brain_venous",
        "description": "Diploic veins of the skull"
    },
    {
        "filename": "Gray568.png",
        "figure": 568,
        "title": "Emissary veins of the skull",
        "category": "brain_venous",
        "description": "Sagittal section showing emissary veins"
    },

    # =========================================================================
    # Brain Stem: Medulla Oblongata
    # =========================================================================
    {
        "filename": "Gray689.png",
        "figure": 689,
        "title": "Median sagittal section of the brain",
        "category": "brain_stem",
        "description": "Median sagittal section of the brain showing major structures"
    },
    {
        "filename": "Gray694.png",
        "figure": 694,
        "title": "Transverse section of medulla oblongata (decussation of pyramids)",
        "category": "brain_stem",
        "description": "Transverse section through the medulla oblongata at the level of decussation of the pyramids"
    },
    {
        "filename": "Gray695.png",
        "figure": 695,
        "title": "Transverse section of medulla oblongata (middle of olive)",
        "category": "brain_stem",
        "description": "Transverse section through the medulla oblongata at about the middle of the olive"
    },
    {
        "filename": "Gray696.png",
        "figure": 696,
        "title": "Transverse section of medulla oblongata (inferior portion of fourth ventricle)",
        "category": "brain_stem",
        "description": "Section of medulla oblongata at about the level of the inferior portion of the fourth ventricle"
    },
    {
        "filename": "Gray697.png",
        "figure": 697,
        "title": "Transverse section of medulla oblongata near ponto-medullary junction",
        "category": "brain_stem",
        "description": "Section at or near the ponto-medullary junction"
    },

    # =========================================================================
    # Brain Stem: Pons
    # =========================================================================
    {
        "filename": "Gray698.png",
        "figure": 698,
        "title": "Transverse section through the pons (lower part)",
        "category": "brain_stem",
        "description": "Transverse section through the lower part of the pons"
    },
    {
        "filename": "Gray699.png",
        "figure": 699,
        "title": "Transverse section through the pons (upper part)",
        "category": "brain_stem",
        "description": "Transverse section through the upper part of the pons"
    },

    # =========================================================================
    # Cerebellum
    # =========================================================================
    {
        "filename": "Gray702.png",
        "figure": 702,
        "title": "Upper surface of the cerebellum",
        "category": "cerebellum",
        "description": "Superior surface of the cerebellum"
    },
    {
        "filename": "Gray703.png",
        "figure": 703,
        "title": "Under surface of the cerebellum",
        "category": "cerebellum",
        "description": "Inferior surface of the cerebellum"
    },
    {
        "filename": "Gray704.png",
        "figure": 704,
        "title": "Sagittal section of the cerebellum",
        "category": "cerebellum",
        "description": "Sagittal section through the cerebellum near the middle line"
    },
    {
        "filename": "Gray705.png",
        "figure": 705,
        "title": "Deep connections of the cerebellum",
        "category": "cerebellum",
        "description": "Dissection showing the deep connections of the cerebellum"
    },
    {
        "filename": "Gray706.png",
        "figure": 706,
        "title": "Scheme of the cerebellum",
        "category": "cerebellum",
        "description": "Schematic representation of cerebellar organization"
    },
    {
        "filename": "Gray707.png",
        "figure": 707,
        "title": "Fourth ventricle and cerebellum (posterior view)",
        "category": "cerebellum",
        "description": "Posterior view of the fourth ventricle and its relation to the cerebellum"
    },

    # =========================================================================
    # Mid-Brain (Mesencephalon)
    # =========================================================================
    {
        "filename": "Gray710.png",
        "figure": 710,
        "title": "Transverse section of mid-brain (superior colliculus)",
        "category": "midbrain",
        "description": "Transverse section through the mid-brain at the level of the superior colliculus"
    },
    {
        "filename": "Gray711.png",
        "figure": 711,
        "title": "Transverse section of mid-brain (inferior colliculus)",
        "category": "midbrain",
        "description": "Transverse section through the mid-brain at the level of the inferior colliculus"
    },
    {
        "filename": "Gray712.png",
        "figure": 712,
        "title": "Corpora quadrigemina and geniculate bodies",
        "category": "midbrain",
        "description": "Dorsal view of the corpora quadrigemina and medial and lateral geniculate bodies"
    },

    # =========================================================================
    # Diencephalon / Thalamus
    # =========================================================================
    {
        "filename": "Gray713.png",
        "figure": 713,
        "title": "The thalamus (dorsal view)",
        "category": "diencephalon",
        "description": "Dorsal view of the thalami with third ventricle"
    },
    {
        "filename": "Gray714.png",
        "figure": 714,
        "title": "Scheme of thalamic nuclei",
        "category": "diencephalon",
        "description": "Schematic diagram of the thalamic nuclei"
    },
    {
        "filename": "Gray715.png",
        "figure": 715,
        "title": "Connections of thalamus",
        "category": "diencephalon",
        "description": "Dissection showing the connection of the thalamus with cortex"
    },

    # =========================================================================
    # Ventricles of the Brain
    # =========================================================================
    {
        "filename": "Gray716.png",
        "figure": 716,
        "title": "The third ventricle (lateral view)",
        "category": "ventricles",
        "description": "Third ventricle seen from the lateral aspect"
    },
    {
        "filename": "Gray717.png",
        "figure": 717,
        "title": "Coronal section of lateral and third ventricles",
        "category": "ventricles",
        "description": "Coronal section through the lateral and third ventricles"
    },
    {
        "filename": "Gray718.png",
        "figure": 718,
        "title": "Corpus callosum from above",
        "category": "ventricles",
        "description": "Dissection of corpus callosum from above"
    },
    {
        "filename": "Gray719.png",
        "figure": 719,
        "title": "The fornix and corpus callosum from below",
        "category": "ventricles",
        "description": "Dissection showing the fornix and hippocampal commissure"
    },
    {
        "filename": "Gray720.png",
        "figure": 720,
        "title": "Dissection showing the ventricles of the brain",
        "category": "ventricles",
        "description": "Lateral ventricles exposed from above after removal of corpus callosum"
    },
    {
        "filename": "Gray721.png",
        "figure": 721,
        "title": "Central part of the lateral ventricles (from above)",
        "category": "ventricles",
        "description": "Central part of the lateral ventricles exposed from above"
    },
    {
        "filename": "Gray722.png",
        "figure": 722,
        "title": "Posterior and inferior horns of lateral ventricle",
        "category": "ventricles",
        "description": "Posterior and inferior horns of the right lateral ventricle exposed"
    },
    {
        "filename": "Gray723.png",
        "figure": 723,
        "title": "Diagram showing relations of the ventricles",
        "category": "ventricles",
        "description": "Diagram showing the relations of the ventricles to the surface of the brain"
    },
    {
        "filename": "Gray724.png",
        "figure": 724,
        "title": "Scheme of roof of fourth ventricle",
        "category": "ventricles",
        "description": "Scheme showing the relation of the fourth ventricle"
    },
    {
        "filename": "Gray725.png",
        "figure": 725,
        "title": "Choroid plexus of the fourth ventricle",
        "category": "ventricles",
        "description": "Choroid plexus of the fourth ventricle seen from behind"
    },

    # =========================================================================
    # Cerebral Hemispheres (Surface Anatomy)
    # =========================================================================
    {
        "filename": "Gray726.png",
        "figure": 726,
        "title": "Lateral surface of left cerebral hemisphere",
        "category": "cerebrum",
        "description": "Lateral surface of the left cerebral hemisphere viewed from the side"
    },
    {
        "filename": "Gray727.png",
        "figure": 727,
        "title": "Medial surface of left cerebral hemisphere",
        "category": "cerebrum",
        "description": "Medial surface of the left cerebral hemisphere"
    },
    {
        "filename": "Gray728.png",
        "figure": 728,
        "title": "Diagram of principal fissures and lobes (lateral surface)",
        "category": "cerebrum",
        "description": "Diagram showing the principal fissures and lobes of the cerebrum on its lateral surface"
    },
    {
        "filename": "Gray729.png",
        "figure": 729,
        "title": "Diagram of principal fissures and lobes (medial surface)",
        "category": "cerebrum",
        "description": "Diagram showing the principal fissures and lobes of the cerebrum on its medial surface"
    },
    {
        "filename": "Gray730.png",
        "figure": 730,
        "title": "Tentorial surface of the cerebrum",
        "category": "cerebrum",
        "description": "The tentorial surface of the cerebrum"
    },
    {
        "filename": "Gray731.png",
        "figure": 731,
        "title": "Orbital surface of the cerebrum",
        "category": "cerebrum",
        "description": "The orbital surface (base) of the cerebrum"
    },
    {
        "filename": "Gray732.png",
        "figure": 732,
        "title": "The insula (Island of Reil)",
        "category": "cerebrum",
        "description": "The insula of the left side, exposed by removing opercula"
    },
    {
        "filename": "Gray733.png",
        "figure": 733,
        "title": "Cerebral sulci (lateral surface)",
        "category": "cerebrum",
        "description": "Detailed view of cerebral sulci on the lateral surface"
    },
    {
        "filename": "Gray734.png",
        "figure": 734,
        "title": "Cerebral sulci (medial surface)",
        "category": "cerebrum",
        "description": "Detailed view of cerebral sulci on the medial surface"
    },

    # =========================================================================
    # Cerebral Internal Structure
    # =========================================================================
    {
        "filename": "Gray735.png",
        "figure": 735,
        "title": "Coronal section through the brain (anterior commissure)",
        "category": "cerebrum_internal",
        "description": "Coronal section of the brain through the anterior commissure"
    },
    {
        "filename": "Gray736.png",
        "figure": 736,
        "title": "Horizontal section showing the internal capsule",
        "category": "cerebrum_internal",
        "description": "Horizontal section of the brain showing the lentiform nucleus and internal capsule"
    },
    {
        "filename": "Gray737.png",
        "figure": 737,
        "title": "Scheme showing the course of fibers in the internal capsule",
        "category": "cerebrum_internal",
        "description": "Scheme showing the positions of fibers in the internal capsule"
    },
    {
        "filename": "Gray738.png",
        "figure": 738,
        "title": "Horizontal section showing the corona radiata",
        "category": "cerebrum_internal",
        "description": "Horizontal section of the brain at the level of the corona radiata"
    },
    {
        "filename": "Gray739.png",
        "figure": 739,
        "title": "Dissection of the cortical association fibers",
        "category": "cerebrum_internal",
        "description": "Dissection of the lateral aspect of the cerebral hemisphere showing association fibers"
    },
    {
        "filename": "Gray740.png",
        "figure": 740,
        "title": "Commissural fibers of the cerebral hemispheres",
        "category": "cerebrum_internal",
        "description": "Dissection showing the commissural fibers of the telencephalon"
    },
    {
        "filename": "Gray741.png",
        "figure": 741,
        "title": "The rhinencephalon and hippocampal formation",
        "category": "cerebrum_internal",
        "description": "The rhinencephalon seen on the inferior surface of the brain"
    },

    # =========================================================================
    # Meninges of the Brain
    # =========================================================================
    {
        "filename": "Gray766.png",
        "figure": 766,
        "title": "Diagrammatic section of the skull and meninges",
        "category": "meninges",
        "description": "Diagrammatic representation of a section across the top of the skull, showing the meninges"
    },
    {
        "filename": "Gray767.png",
        "figure": 767,
        "title": "Dura mater and its processes",
        "category": "meninges",
        "description": "Dura mater and its processes seen from the inner surface of the skull"
    },
    {
        "filename": "Gray768.png",
        "figure": 768,
        "title": "Tentorium cerebelli (seen from above)",
        "category": "meninges",
        "description": "The tentorium cerebelli seen from above, showing falx cerebri and tentorium"
    },
    {
        "filename": "Gray769.png",
        "figure": 769,
        "title": "Superior sagittal sinus opened up",
        "category": "meninges",
        "description": "The superior sagittal sinus opened after removal of the skull cap"
    },
    {
        "filename": "Gray770.png",
        "figure": 770,
        "title": "The sinuses at the base of the skull",
        "category": "meninges",
        "description": "The dural venous sinuses at the base of the skull"
    },

    # =========================================================================
    # Cranial Nerves
    # =========================================================================
    {
        "filename": "Gray776.png",
        "figure": 776,
        "title": "Scheme showing cranial nerve nuclei",
        "category": "cranial_nerves",
        "description": "Scheme showing the motor and sensory nuclei of the cranial nerves"
    },
    {
        "filename": "Gray778.png",
        "figure": 778,
        "title": "The optic nerve and optic tract",
        "category": "cranial_nerves",
        "description": "The left optic nerve and the optic tracts"
    },
    {
        "filename": "Gray784.png",
        "figure": 784,
        "title": "The trigeminal nerve (semilunar ganglion and branches)",
        "category": "cranial_nerves",
        "description": "Distribution of the trigeminal nerve"
    },
    {
        "filename": "Gray788.png",
        "figure": 788,
        "title": "Nerves of the orbit (seen from above)",
        "category": "cranial_nerves",
        "description": "Nerves of the orbit and ophthalmic division of the trigeminal"
    },
    {
        "filename": "Gray789.png",
        "figure": 789,
        "title": "The facial nerve and its branches",
        "category": "cranial_nerves",
        "description": "Plan of the facial nerve"
    },
    {
        "filename": "Gray790.png",
        "figure": 790,
        "title": "The hypoglossal nerve and its branches",
        "category": "cranial_nerves",
        "description": "The hypoglossal nerve, cervical plexus, and their branches"
    },
    {
        "filename": "Gray791.png",
        "figure": 791,
        "title": "The glossopharyngeal, vagus, and accessory nerves",
        "category": "cranial_nerves",
        "description": "Plan of the glossopharyngeal, vagus, and accessory nerves"
    },
    {
        "filename": "Gray792.png",
        "figure": 792,
        "title": "Course and distribution of the glossopharyngeal, vagus, and accessory nerves",
        "category": "cranial_nerves",
        "description": "Upper portions of the glossopharyngeal, vagus, and accessory nerves"
    },

    # =========================================================================
    # Skull / Cranium (relevant to neurosurgery)
    # =========================================================================
    {
        "filename": "Gray188.png",
        "figure": 188,
        "title": "Side view of the skull",
        "category": "skull",
        "description": "Side view of the skull (norma lateralis)"
    },
    {
        "filename": "Gray190.png",
        "figure": 190,
        "title": "The skull from the front",
        "category": "skull",
        "description": "The skull viewed from the front (norma frontalis)"
    },
    {
        "filename": "Gray193.png",
        "figure": 193,
        "title": "Base of the skull (interior surface)",
        "category": "skull",
        "description": "Base of the skull, upper surface showing anterior, middle and posterior cranial fossae"
    },
    {
        "filename": "Gray194.png",
        "figure": 194,
        "title": "Base of the skull (exterior surface)",
        "category": "skull",
        "description": "Base of the skull, exterior surface"
    },
    {
        "filename": "Gray197.png",
        "figure": 197,
        "title": "Median sagittal section of the skull",
        "category": "skull",
        "description": "Sagittal section of the skull"
    },

    # =========================================================================
    # Spinal Cord (cervical, relevant to neurosurgery)
    # =========================================================================
    {
        "filename": "Gray664.png",
        "figure": 664,
        "title": "The spinal cord (posterior view)",
        "category": "spinal_cord",
        "description": "Posterior view of the spinal cord showing nerve roots"
    },
    {
        "filename": "Gray665.png",
        "figure": 665,
        "title": "Cross-section of the spinal cord",
        "category": "spinal_cord",
        "description": "Transverse section of the spinal cord"
    },
    {
        "filename": "Gray666.png",
        "figure": 666,
        "title": "Spinal cord cross-sections at different levels",
        "category": "spinal_cord",
        "description": "Cross-sections of the spinal cord at various levels"
    },
    {
        "filename": "Gray667.png",
        "figure": 667,
        "title": "Spinal cord (anterior view)",
        "category": "spinal_cord",
        "description": "Anterior median fissure of the spinal cord"
    },
    {
        "filename": "Gray668.png",
        "figure": 668,
        "title": "Meninges of the spinal cord",
        "category": "spinal_cord",
        "description": "The spinal cord with dura mater and arachnoid opened"
    },
    {
        "filename": "Gray769.png",
        "figure": 769,
        "title": "Diagram of the meninges and their relation to brain and spinal cord",
        "category": "meninges",
        "description": "Diagrammatic section showing the relationship of meninges to CNS structures"
    },
]

# Remove duplicate entries (Gray769 appears twice)
seen = set()
unique_plates = []
for p in PLATES:
    if p["filename"] not in seen:
        seen.add(p["filename"])
        unique_plates.append(p)
PLATES = unique_plates


def get_wikimedia_url(filename):
    """Compute the Wikimedia Commons direct download URL for a file."""
    md5 = hashlib.md5(filename.encode()).hexdigest()
    return f"https://upload.wikimedia.org/wikipedia/commons/{md5[0]}/{md5[0:2]}/{filename}"


def download_image(filename, output_dir, retries=3, delay=1):
    """Download an image from Wikimedia Commons."""
    url = get_wikimedia_url(filename)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"  [SKIP] {filename} already exists")
        return True

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "GraysAnatomyDownloader/1.0 (educational/research use)"
            })
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
                with open(output_path, "wb") as f:
                    f.write(data)
                print(f"  [OK]   {filename} ({len(data)} bytes)")
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if attempt < retries - 1:
                print(f"  [RETRY] {filename} - {e} (attempt {attempt + 1}/{retries})")
                time.sleep(delay * (attempt + 1))
            else:
                print(f"  [FAIL] {filename} - {e}")
                return False
    return False


def save_metadata(plates, output_dir):
    """Save plate metadata as JSON."""
    metadata = {
        "source": "Gray's Anatomy of the Human Body, 20th U.S. edition (1918)",
        "author": "Henry Gray",
        "illustrator": "Henry Vandyke Carter",
        "editor": "Warren H. Lewis",
        "license": "Public Domain",
        "wikimedia_category": "https://commons.wikimedia.org/wiki/Category:Gray%27s_Anatomy_plates_of_nervous_system",
        "plates": []
    }

    for plate in plates:
        entry = {
            "filename": plate["filename"],
            "figure_number": plate["figure"],
            "title": plate["title"],
            "category": plate["category"],
            "description": plate["description"],
            "wikimedia_url": f"https://commons.wikimedia.org/wiki/File:{plate['filename']}",
            "direct_download_url": get_wikimedia_url(plate["filename"])
        }
        metadata["plates"].append(entry)

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved to {metadata_path}")


def main():
    output_dir = "."
    if len(sys.argv) > 2 and sys.argv[1] == "--output-dir":
        output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    categories = {}
    for plate in PLATES:
        cat = plate["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(plate)

    print("=" * 70)
    print("Gray's Anatomy 20th Edition (1918) - Neurosurgery-Related Plates")
    print("=" * 70)
    print(f"Total plates: {len(PLATES)}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print()

    total = len(PLATES)
    success = 0
    failed = 0

    for cat_name, cat_plates in categories.items():
        cat_display = cat_name.replace("_", " ").title()
        print(f"\n--- {cat_display} ({len(cat_plates)} plates) ---")
        for plate in cat_plates:
            if download_image(plate["filename"], output_dir):
                success += 1
            else:
                failed += 1

    save_metadata(PLATES, output_dir)

    print(f"\n{'=' * 70}")
    print(f"Download complete: {success}/{total} succeeded, {failed} failed")
    print(f"{'=' * 70}")

    if failed > 0:
        print("\nNote: Some downloads failed. Re-run the script to retry.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
