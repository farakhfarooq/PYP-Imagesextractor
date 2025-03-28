import os
import re
import cv2
import pytesseract
import pandas as pd
from glob import glob

# If Tesseract is not in your system PATH, uncomment and specify its location:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path, method="otsu", use_morphology=False):
    """
    Reads an image, converts to grayscale, and applies thresholding.
    Offers two methods: 'otsu' or 'adaptive'.
    Optionally applies a morphological operation to remove noise.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding
    if method.lower() == "adaptive":
        # Adaptive thresholding can help if lighting varies across the image
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or ADAPTIVE_THRESH_MEAN_C
            cv2.THRESH_BINARY,
            11,   # blockSize (typical odd values: 11, 15, 21, etc.)
            2     # C constant subtracted from mean
        )
    else:
        # Default: Otsu threshold
        _, thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Optional morphological operation to reduce noise
    if use_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return thresh

def extract_text_from_image(image):
    """
    Performs OCR on a preprocessed (thresholded) image and returns the extracted text.
    """
    text = pytesseract.image_to_string(image)
    # Text Cleanup: remove non-ASCII chars, collapse multiple spaces/newlines
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_data(text):
    """
    Uses regex patterns to extract relevant fields:
    - Sender
    - Receiver
    - Amount
    - Bank Details
    - Transaction ID / Reference
    - Transaction Status
    Demonstrates a secondary pass (manual override) if needed.
    """
    data = {
        "Sender": None,
        "Receiver": None,
        "Amount": None,
        "Bank_Details": None,
        "Transaction_ID": None,
        "Reference_Number": None,
        "Transaction_Status": None
    }

    # --- 1) Sender ---
    #   Based on sample images: "From", "Funding Source", "Source Acc. Title", etc.
    sender_patterns = [
        r"(?:From|Sender|Funding Source|Source Acc\.? Title|Paid by|Payer)\s*[:\-]?\s*(.*)",
    ]
    data["Sender"] = find_first_match(text, sender_patterns)

    # --- 2) Receiver ---
    #   Keywords: "To", "Receiver", "Sent to", "Beneficiary", "Destination Acc. Title", etc.
    receiver_patterns = [
        r"(?:To|Receiver|Sent to|Beneficiary|Payee|Destination Acc\.? Title)\s*[:\-]?\s*(.*)",
    ]
    data["Receiver"] = find_first_match(text, receiver_patterns)

    # --- 3) Amount ---
    #   Matches lines like "Rs. 210.00", "PKR 1,000", "Amount Sent: Rs. 3,500", "Total Amount Rs. 500.00"
    amount_patterns = [
        r"(?:Rs\.?|PKR)\s*[\.:]?\s*([0-9,]+\.?[0-9]*)",
        r"(?:Amount Sent|Total Amount|Amount)\s*[:\-]?\s*(?:Rs\.?|PKR)?\s*([0-9,]+\.?[0-9]*)"
    ]
    data["Amount"] = find_first_match(text, amount_patterns)

    # --- 4) Bank Details ---
    #   Could be "Silk Bank", "Easypaisa Bank-8848", "********4508", or explicit account #.
    #   We'll also look for lines containing 'Bank' or 'Account' or partial masked accounts.
    bank_patterns = [
        r"(?:Silk Bank|Easypaisa Bank\-?\d*|NayaPay|.*\*{4,}\d{3,4}|[0-9]{9,14})",
        r"(?:Bank\s*Account|Acct\s*No\.?|Account)\s*[:\-]?\s*([\w\-*]+)"
    ]
    data["Bank_Details"] = find_first_match(text, bank_patterns)

    # --- 5) Transaction ID or Reference ---
    #   Matches lines like "Transaction ID 67d1...", "Ref# 34760665004", etc.
    transaction_id_patterns = [
        r"(?:Transaction ID|Trans\.? ID|Tx ID|Ref#|Reference|Ref)\s*[:\-]?\s*([\w\d\-]+)"
    ]
    data["Transaction_ID"] = find_first_match(text, transaction_id_patterns)

    # If you want to specifically separate "Transaction_ID" from "Reference_Number", 
    # you could create a separate pattern for references only, e.g.:
    reference_patterns = [
        r"(?:Ref|Ref#|Reference)\s*[:\-]?\s*([\w\d\-]+)"
    ]
    data["Reference_Number"] = find_first_match(text, reference_patterns)

    # --- 6) Transaction Status ---
    #   Lines like "Transaction Successful", "Money has been sent", etc.
    status_patterns = [
        r"(Transaction Successful|Transaction successful|Money has been sent|Successfully Sent)",
    ]
    data["Transaction_Status"] = find_first_match(text, status_patterns)

    # --- Manual Overrides (Secondary Pass) ---
    #   If certain fields are still None, do an alternate search or custom logic.
    #   For example, if "Sender" was not found above, look for a line containing "By" or "SENDER" in uppercase:
    if not data["Sender"]:
        override_sender_pattern = r"(?:by)\s*[:\-]?\s*(.*)"
        data["Sender"] = find_first_match(text, [override_sender_pattern])

    return data

def find_first_match(text, patterns):
    """
    Iterates over a list of regex patterns.
    Returns the first matching group(1) or None if no match is found.
    Splits on newline if needed to avoid capturing too much text.
    """
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            # We often only want the first match line
            raw_value = matches[0].strip()
            # If it’s multiple lines, take just the first line to avoid capturing extra
            value = raw_value.split('\n')[0].strip()
            # Remove trailing punctuation if needed
            value = re.sub(r'[.,:;]+$', '', value)
            return value
    return None

if __name__ == "__main__":
    # Folder path containing your images
    folder_path = "C:\Users\farak\Downloads\PYP-Images"
    
    # Collect all image files in that folder (adjust extension/pattern if needed)
    # Sorting by int(filename) if your images are named "1.jpg", "2.jpg", etc.
    image_files = sorted(
        glob(os.path.join(folder_path, "*.*")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    
    # Prepare a list to store the results
    all_data = []

    for img_file in image_files:
        try:
            # Example: Use Otsu threshold + morphological operation
            preprocessed = preprocess_image(img_file, method="otsu", use_morphology=True)
            ocr_text = extract_text_from_image(preprocessed)
            parsed = extract_data(ocr_text)

            # Keep track of which image this came from
            parsed["Image"] = os.path.basename(img_file)

            all_data.append(parsed)
            print(f"Processed: {img_file}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    # Create a DataFrame and save to Excel
    df = pd.DataFrame(
        all_data,
        columns=[
            "Image",
            "Sender",
            "Receiver",
            "Amount",
            "Bank_Details",
            "Transaction_ID",
            "Reference_Number",
            "Transaction_Status"
        ]
    )
    output_file = "extracted_data.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\nData saved to {output_file}")
