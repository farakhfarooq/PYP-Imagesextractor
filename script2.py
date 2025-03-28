import os
import re
import cv2
import pytesseract
import pandas as pd
from glob import glob

# If needed, specify the path to your Tesseract executable:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path, method="otsu", use_morphology=False):
    """
    Reads an image, converts it to grayscale, and applies thresholding.
    Offers two methods: 'otsu' or 'adaptive'.
    Optionally applies a morphological operation to reduce noise.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method.lower() == "adaptive":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if use_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return thresh

def extract_text_from_image(image):
    """
    Performs OCR on a preprocessed image and cleans the resulting text.
    """
    text = pytesseract.image_to_string(image)
    # Remove non-ASCII characters and collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_clean_data(text):
    """
    Cleans up the OCR text and applies refined regex patterns to extract:
    - Sender
    - Receiver
    - Total Amount
    - Sender Bank
    - Receiver Bank
    """
    data = {
        "Sender": None,
        "Receiver": None,
        "Total_Amount": None,
        "Sender_Bank": None,
        "Receiver_Bank": None
    }
    
    # Clean text further (remove extra spaces)
    text = re.sub(r'\s+', ' ', text)
    
    # --- Extract Sender ---
    # Look for "Source Acc. Title" or "Sent by"
    sender_match = re.search(r'Source Acc\.?\s*Title\s*([\w &]+)', text, re.IGNORECASE)
    if sender_match:
        data["Sender"] = sender_match.group(1).strip()
    else:
        sender_match = re.search(r'Sent by\s*([\w &]+)', text, re.IGNORECASE)
        if sender_match:
            data["Sender"] = sender_match.group(1).strip()
    
    # --- Extract Receiver ---
    # Look for "Destination Acc. Title" or fallback to "To"
    receiver_match = re.search(r'Destination Acc\.?\s*Title\s*([\w &]+)', text, re.IGNORECASE)
    if receiver_match:
        data["Receiver"] = receiver_match.group(1).strip()
    else:
        receiver_match = re.search(r'To\s*([\w &]+)', text, re.IGNORECASE)
        if receiver_match:
            data["Receiver"] = receiver_match.group(1).strip()
    
    # --- Extract Total Amount ---
    # Look for "Total Amount" or "Amount" with optional currency indicators (Rs., PKR, etc.)
    amount_match = re.search(r'(?:Total Amount|Amount)\s*(?:Rs\.?\s*)?([\d,]+\.\d{2}|[\d,]+)', text, re.IGNORECASE)
    if amount_match:
        data["Total_Amount"] = amount_match.group(1).replace(',', '').strip()
    
    # --- Extract Sender Bank Details ---
    # Look for "Source Bank"
    sender_bank_match = re.search(r'Source Bank\s*([\w]+)', text, re.IGNORECASE)
    if sender_bank_match:
        data["Sender_Bank"] = sender_bank_match.group(1).strip()
    
    # --- Extract Receiver Bank Details ---
    # Look for "Destination Bank"
    receiver_bank_match = re.search(r'Destination Bank\s*([\w]+)', text, re.IGNORECASE)
    if receiver_bank_match:
        data["Receiver_Bank"] = receiver_bank_match.group(1).strip()
    
    return data

if __name__ == "__main__":
    # Update this path to point to your folder of images
    folder_path = r"C:\Users\farak\Downloads\PYP-Images"
    
    # Get all image files in the folder in the order they are listed
    image_files = glob(os.path.join(folder_path, "*.*"))
    
    # List to store cleaned data for each image
    all_data = []

    for img_file in image_files:
        try:
            preprocessed = preprocess_image(img_file, method="otsu", use_morphology=True)
            ocr_text = extract_text_from_image(preprocessed)
            parsed = extract_clean_data(ocr_text)
            parsed["Image"] = os.path.basename(img_file)
            all_data.append(parsed)
            print(f"Processed: {img_file}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    # Create a DataFrame with more descriptive columns
    df = pd.DataFrame(all_data, columns=["Image", "Sender", "Receiver", "Total_Amount", "Sender_Bank", "Receiver_Bank"])
    output_file = "cleaned_extracted_data.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\nCleaned data saved to {output_file}")
