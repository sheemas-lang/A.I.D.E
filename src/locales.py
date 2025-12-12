# locales.py

LOCALES = {
    # Available Language Options for the User Dropdown
    "language_options": {
        "en": "English",
        "hin": "Hindi (hin)",
        "kan": "Kannada (kan)",
        "tam": "Tamil (tam)",
        "mal": "Malayalam (mal)",
        # You can add other languages here if you have the traineddata and translations
    },
    # Translatable strings
    "translations": {
        "STATUS_ACCEPTED": {
            "en": "Loan Application Accepted! 🎉",
            "hin": "लोन एप्लीकेशन स्वीकार कर लिया गया है! 🎉",
            "kan": "ಸಾಲದ ಅರ್ಜಿಯನ್ನು ಸ್ವೀಕರಿಸಲಾಗಿದೆ! 🎉",
            "tam": "கடன் விண்ணப்பம் ஏற்றுக்கொள்ளப்பட்டது! 🎉",
            "mal": "വായ്പാ അപേക്ഷ സ്വീകരിച്ചു! 🎉",
        },
        "STATUS_REJECTED": {
            "en": "Loan Application Rejected 😞",
            "hin": "लोन एप्लीकेशन रिजेक्ट हो गया है 😞",
            "kan": "ಸಾಲದ ಅರ್ಜಿಯನ್ನು ತಿರಸ್ಕರಿಸಲಾಗಿದೆ 😞",
            "tam": "கடன் விண்ணப்பம் நிராகரிக்கப்பட்டது 😞",
            "mal": "വായ്പാ അപേക്ഷ നിരസിച്ചു 😞",
        },
        "INFO_LANG_SELECT": {
            "en": "OCR is performed in English. The language selected below is only for the final result status.",
            "hin": "डेटा एनालिसिस इंग्लिश में हो रहा है",
            "kan": "ಡೇಟಾ ವಿಶ್ಲೇಷಣೆ ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ ನಡೆಯುತ್ತಿದೆ.",
            "tam": "தரவு பகுப்பாய்வு ஆங்கிலத்தில் நடக்கிறது.",
            "mal": "ഡാറ്റ വിശകലനം ഇംഗ്ലീഷിൽ നടക്കുന്നു.",
        }
    }
}

def get_translation(key, lang_code):
    """Retrieves the translated string for a given key and language code."""
    # Fallback to English if the translation is missing for the chosen language
    return LOCALES["translations"].get(key, {}).get(lang_code, LOCALES["translations"][key]["en"])
