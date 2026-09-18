from pathlib import Path
import shutil


# WHERE YOUR PDFs ARE CURRENTLY
SOURCE_FOLDER = Path("C:\\Users\\flexy\\Downloads\\files (1)")


# WHERE YOU WANT THEM TO GO
DESTINATION_FOLDER = Path(
    "C:\\Users\\flexy\\OneDrive\\Desktop\\hackathon_man\\documents"
)


# Create destination folder if it doesn't exist
DESTINATION_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# Find every PDF
pdf_files = SOURCE_FOLDER.glob("*.pdf")


count = 0

for pdf in pdf_files:

    destination = DESTINATION_FOLDER / pdf.name

    shutil.copy2(
        pdf,
        destination
    )

    print(f"Copied: {pdf.name}")

    count += 1


print()
print("=" * 50)
print(f"Copied {count} PDF files.")
print("=" * 50)