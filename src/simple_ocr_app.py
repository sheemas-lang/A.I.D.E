import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import cv2
import io
from pdf2image import convert_from_bytes
import os

# --- Import localization functions ---
from locales import LOCALES, get_translation
# -------------------------------------

# =========================================================
# !!! CRITICAL: TESSERACT & POPPLER CONFIGURATION !!!
# =========================================================

# 1. TESSERACT CONFIGURATION:
# UNCOMMENT the line below and replace the path with your exact Tesseract-OCR\tesseract.exe location.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 

# 2. POPPLER CONFIGURATION (for PDF support):
# UNCOMMENT the lines below and replace the path with the location of your Poppler 'bin' folder.
# POPPLER_PATH = r'C:\path\to\poppler-xx\Library\bin' 
# os.environ['PATH'] += os.pathsep + POPPLER_PATH

# --- OCR CORE FUNCTION (UNCHANGED) ---

def extract_text_from_document(uploaded_file):
    """
    Extracts text from an uploaded image or PDF file using Tesseract.
    (This function remains unchanged as per user request)
    """
    text = ""
    file_type = uploaded_file.type
    
    try:
        # --- 1. Process Image Files (JPG, PNG) ---
        if file_type in ["image/jpeg", "image/png"]:
            st.info("Processing Image file...")
            
            image_data = uploaded_file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error("Could not decode image. File might be corrupted.")
                return ""
            
            # Preprocessing: Grayscale, Thresholding, Denoising
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            denoised = cv2.medianBlur(thresh, 3) 
            pil_image = Image.fromarray(denoised)
            
            text = pytesseract.image_to_string(pil_image, lang='eng')

        # --- 2. Process PDF Files ---
        elif file_type == "application/pdf":
            st.info("Processing PDF file...")
            
            pdf_bytes = uploaded_file.read()
            images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
            
            if images:
                text = pytesseract.image_to_string(images[0], lang='eng')
            else:
                st.error("Could not convert PDF to image for OCR. Check Poppler configuration.")

        else:
            st.warning(f"Unsupported file type: {file_type}")
            return ""

        return text

    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract OCR is not found. Please check its installation and the path configuration in the code.")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred during OCR: {e}")
        return ""

# --- STREAMLIT UI ---

st.set_page_config(page_title="Simple OCR Extractor", layout="wide")
st.title("📄 Document Text Extractor (OCR)")

# --- Language Selection Sidebar ---
with st.sidebar:
    st.header("1. Result Language")
    
    # User selects the language for the final ACCEPTED/REJECTED message
    selected_lang_name = st.selectbox(
        "Display Loan Status In:",
        options=list(LOCALES["language_options"].values()),
        index=0 
    )
    # Get the language code (e.g., 'hin') from the name ('Hindi (hin)')
    selected_lang_code = next(
        (code for code, name in LOCALES["language_options"].items() if name == selected_lang_name), 
        'en' 
    )

    # Display translated info message
    info_lang_select = get_translation("INFO_LANG_SELECT", selected_lang_code)
    st.info(info_lang_select)

    st.header("2. Document Upload")

uploaded_file = st.file_uploader(
    "Upload a document (JPG, PNG, or PDF)", 
    type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file is not None:
    
    with st.spinner("Extracting text and determining status..."):
        # 1. EXTRACT TEXT
        extracted_text = extract_text_from_document(uploaded_file)
        
        # 2. DUMMY LOAN DECISION LOGIC (NEW FEATURE)
        loan_status = False
        if extracted_text and len(extracted_text) > 100:
            # DUMMY CHECK: ACCEPT if the extracted text is over 100 characters long
            # PARAMETERS TO CHECK: "Salary" in extracted_text or "Income" in extracted_text or any required field.
            loan_status = True
        
    st.markdown("---")
    
    if extracted_text.strip():
        st.subheader("✅ Extracted Text (English)")
        st.code(extracted_text, language='text', height=250)
        
        # --- FINAL TRANSLATED RESULT DISPLAY ---
        st.subheader("Final Loan Status")
        if loan_status is True:
            # Get the ACCEPTED message in the user's selected regional language
            result_message = get_translation("STATUS_ACCEPTED", selected_lang_code)
            st.success(f"## {result_message}")
        else:
            # Get the REJECTED message in the user's selected regional language
            result_message = get_translation("STATUS_REJECTED", selected_lang_code)
            st.error(f"## {result_message}")

    else:
        st.error("Text extraction failed or returned no text. Cannot determine loan status.")

else:
    st.info("Upload a document using the uploader above to begin loan processing.")
