# A.I.D.E
Agentic Intake Decision and Explanation system (PS-5)
# Team Name : Caffeine Crew
# Problem Statement : 5
### AI-Powered Credit Decisioning + Face Verification + OCR

This project is a *fast loan approval system* built for the hackathon.  
It provides *sub-60 second credit decisions* using our ML model, OCR-based document extraction, and face-match based identity verification.

The system works for:
- Salary advance loans for employees  
- Small personal loans  
- Micro-loans for small merchants  

Everything runs automatically through our pipelines.

---

## 🚀 Features
### ✅ *1. AI Credit Decision Engine*
- Uses trained ML models (XGBoost )
- Gives instant *approve / reject / limit* decisions
- Uses affordability checks (DTI (Debt to Income ratio), LTI (Loan to Income Ratio) , income, employment stability , age)

### ✅ *2. OCR Document Extraction*
- Reads PAN/Aadhaar/ID photos
- Extracts text for KYC verification
- Cleans + parses fields like name, DOB, ID number

### ✅ *3. Face Verification*
- Compares *selfie vs ID card photo*
- Confirms identity match before loan approval

### ✅ *4. Explainable AI*
- Shows *why* a user was approved or declined  
- Uses feature importance + simple rules

### ✅ *5. Clean UI / App Flow*
- Upload documents  
- Take a selfie  
- Fill loan details  
- Get instant decision + EMI plan
