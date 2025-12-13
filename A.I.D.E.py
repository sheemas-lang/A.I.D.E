"""
🧠 A.I.D.E - COMPLETE Loan + PayU Wallet + Face Recognition (User sees ONLY PASS/FAIL)
"""

import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image
from datetime import datetime

# 🌐 2 REGIONAL LANGUAGES (Simplified)
LANGUAGES = {
    'en': {
        'approved_title': '✅ LOAN APPROVED! 🎉',
        'rejected_title': '❌ LOAN REJECTED',
        'emi_label': 'Monthly EMI', 'credit_score': 'Credit Score',
        'dti_ratio': 'DTI Ratio', 'face_match': 'Face Match',
        'wallet_status': 'Wallet Status', 'wallet_balance': 'Wallet Balance',
        'loan_types': ['Personal Loan', 'Home Loan', 'Car Loan', 'Business Loan'],
        'step': 'Step {}/7', 'face_pass': '✅ FACE MATCH - PASSED',
        'face_fail': '❌ FACE MATCH - FAILED',
        'wallet_updated': '💰 Wallet Updated Successfully!',
        'settlement_pending': '⏳ Settlement Pending',
        'payu_txn_id': 'PayU Txn ID'
    },
    'hi': {
        'approved_title': '✅ लोन स्वीकृत! 🎉', 'rejected_title': '❌ लोन अस्वीकृत',
        'emi_label': 'मासिक EMI', 'credit_score': 'क्रेडिट स्कोर',
        'dti_ratio': 'ऋण-आय अनुपात', 'face_match': 'चेहरा मिलान',
        'wallet_status': 'वॉलेट स्थिति', 'wallet_balance': 'वॉलेट बैलेंस',
        'loan_types': ['पर्सनल लोन', 'होम लोन', 'कार लोन', 'बिजनेस लोन'],
        'step': 'चरण {}/7', 'face_pass': '✅ चेहरा मिलान - पास',
        'face_fail': '❌ चेहरा मिलान - विफल',
        'wallet_updated': '💰 वॉलेट अपडेट!',
        'settlement_pending': '⏳ सेटलमेंट प्रक्रिया चल रही',
        'payu_txn_id': 'PayU लेनदेन ID'
    }
}

st.set_page_config(page_title="🧠 A.I.D.E", page_icon="🧠", layout="wide")

# 🌈 UI STYLES
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }
.main { background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0fdf4 100%); }
.agent-card { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1.5rem; border-radius: 20px; margin: 1rem 0; border-left: 5px solid; box-shadow: 0 4px 15px rgba(251,191,36,0.2); }
.agent-name { color: #06b6d4 !important; font-weight: 700 !important; font-size: 1.4rem !important; }
.maya { border-left-color: #a78bfa; } .rex { border-left-color: #60a5fa; } .leo { border-left-color: #f59e0b; } .sophia { border-left-color: #10b981; } .victor { border-left-color: #ef4444; } .sage { border-left-color: #8b5cf6; }
.agent-title { color: #dc2626 !important; font-weight: 700; font-size: 2.5rem; }
.pass-badge { background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 2rem; border-radius: 25px; text-align: center; font-size: 1.8rem; font-weight: 700; box-shadow: 0 10px 30px rgba(16,185,129,0.4); }
.fail-badge { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; padding: 2rem; border-radius: 25px; text-align: center; font-size: 1.8rem; font-weight: 700; box-shadow: 0 10px 30px rgba(239,68,68,0.4); }
.wallet-success { background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 2.5rem; border-radius: 30px; text-align: center; font-size: 1.8rem; font-weight: 700; box-shadow: 0 15px 40px rgba(16,185,129,0.4); }
.wallet-pending { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 2rem; border-radius: 25px; text-align: center; font-size: 1.5rem; font-weight: 700; }
.approved { background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 3rem; border-radius: 25px; border: 4px solid #10b981; text-align: center; box-shadow: 0 15px 40px rgba(16,185,129,0.3); }
.rejected { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 3rem; border-radius: 25px; border: 4px solid #ef4444; text-align: center; box-shadow: 0 15px 40px rgba(239,68,68,0.3); }
.stButton>button { background: linear-gradient(135deg, #a7f3d0 0%, #6ee7b7 100%); color: #065f46; border-radius: 20px; font-weight: 600; padding: 12px 24px; border: 2px solid #34d399; }
</style>
""", unsafe_allow_html=True)

# 🔥 FACE COMPARISON FUNCTIONS
def extract_face_features(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0: return None
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (64, 64))
    return face.flatten().astype(np.float32) / 255.0

def compare_faces(id_image, selfie_image):
    id_features = extract_face_features(id_image)
    selfie_features = extract_face_features(selfie_image)
    if id_features is None or selfie_features is None:
        return {'match': False, 'confidence': 0, 'error': 'No face detected'}
    distance = np.sqrt(np.sum((id_features - selfie_features)**2))
    similarity = max(0, 100 - (distance * 1000))
    match = distance < 15.0
    return {'match': match, 'confidence': similarity, 'distance': distance}

# 💰 PAYU WALLET CLASS
class PayUWallet:
    def __init__(self):
        self.wallet_balances = {}
    
    def initiate_disbursal(self, user_id, amount):
        txn_id = f"PU{int(time.time())}"
        return {
            'txn_id': txn_id,
            'status': 'initiated',
            'amount': amount,
            'timestamp': datetime.now().isoformat()
        }
    
    def check_settlement(self, txn_id, amount):
        time.sleep(np.random.uniform(2, 4))
        success = np.random.random() > 0.05
        return {
            'status': 'settled' if success else 'pending',
            'amount': amount if success else 0
        }
    
    def update_wallet(self, user_id, amount):
        if user_id not in self.wallet_balances:
            self.wallet_balances[user_id] = 0
        self.wallet_balances[user_id] += amount
        return self.wallet_balances[user_id]

# 🔥 CHATBOT FUNCTION HERE ⬇️
# 🔥 CHATBOT FUNCTION (Add after PayUWallet class, around line 110)
def chatbot_response(query, lang='en'):
    """Simple A.I.D.E-specific chatbot"""
    query_lower = query.lower()
    
    responses = {
        'en': {
            'loan': 'A.I.D.E processes loans in 7 steps: Personal details → Documents → Face recognition → Risk analysis → Credit scoring → Decision → Wallet disbursal.',
            'face': 'Upload clear ID proof photo (Aadhaar/PAN) and front-facing selfie with good lighting for Leo agent.',
            'wallet': 'Approved loans are disbursed to wallet via PayU with real-time settlement verification (2-5 sec delay).',
            'step': '7 steps total: 1.Maya(Data) 2.Rex(Docs) 3.Leo(Face) 4.Victor(Risk) 5.Sophia(Credit) 6.Sage(Decision+Wallet)',
            'default': 'Ask about "loan process", "face recognition", "wallet", or "steps"! 💬'
        },
        'hi': {
            'loan': 'A.I.D.E 7 चरणों में लोन प्रोसेस करता है: व्यक्तिगत विवरण → दस्तावेज → चेहरा पहचान → जोखिम विश्लेषण → क्रेडिट स्कोरिंग → निर्णय → वॉलेट डिस्बर्सल।',
            'face': 'स्पष्ट ID प्रूफ (आधार/पैन) और अच्छी रोशनी में फ्रंट-फेस सेल्फी अपलोड करें।',
            'wallet': 'स्वीकृत लोन PayU के माध्यम से वॉलेट में 2-5 सेकंड में सेटलमेंट के साथ डाले जाते हैं।',
            'step': '7 चरण: 1.माया(डेटा) 2.रैक्स(दस्तावेज) 3.लियो(चेहरा) 4.विक्टर(जोखिम) 5.सोफिया(क्रेडिट) 6.सेज(निर्णय+वॉलेट)',
            'default': '"लोन प्रक्रिया", "चेहरा पहचान", "वॉलेट", या "चरण" के बारे में पूछें! 💬'
        }
    }
    
    if 'loan' in query_lower or 'process' in query_lower:
        return responses[lang]['loan']
    elif 'face' in query_lower or 'selfie' in query_lower or 'leo' in query_lower:
        return responses[lang]['face']
    elif 'wallet' in query_lower or 'payu' in query_lower or 'money' in query_lower:
        return responses[lang]['wallet']
    elif 'step' in query_lower or 'stages' in query_lower:
        return responses[lang]['step']
    else:
        return responses[lang]['default']


# INITIALIZE SESSION STATE
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.user_data = {}
    st.session_state.selfie_uploaded = False
    st.session_state.id_uploaded = False
    st.session_state.current_lang = 'en'
    st.session_state.payu_wallet = PayUWallet()

st.title("🧠 **A.I.D.E** - Advanced Intelligent Decision Engine")
st.markdown("🌟 *Face Recognition + PayU Wallet Settlement Verification*")

# 🌐 LANGUAGE SELECTION - STEP 0
if st.session_state.step == 0:
    lang_options = ["🇬🇧 English", "🇮🇳 हिंदी"]
    selected_lang = st.selectbox("🌐 Select Language", lang_options)
    st.session_state.current_lang = 'en' if 'English' in selected_lang else 'hi'
    current_lang_dict = LANGUAGES[st.session_state.current_lang]
    
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 20px; margin: 2rem 0;'>
        <h1 class='agent-title'>🚀 Meet Our 6 AI Agents</h1>
        <p style='color: #1e293b; font-size: 1.2rem;'>Advanced loan processing with facial recognition & wallet disbursal</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="agent-card maya"><h3 class="agent-name">🧠 Maya</h3><p style="color: #1e293b; margin-top: 0.5rem;">Data Validation</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="agent-card rex"><h3 class="agent-name">🔍 Rex</h3><p style="color: #1e293b; margin-top: 0.5rem;">Document OCR</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="agent-card leo"><h3 class="agent-name">👤 Leo</h3><p style="color: #1e293b; margin-top: 0.5rem;">Face Recognition</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="agent-card sophia"><h3 class="agent-name">📊 Sophia</h3><p style="color: #1e293b; margin-top: 0.5rem;">Credit Scoring</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="agent-card victor"><h3 class="agent-name">🛡 Victor</h3><p style="color: #1e293b; margin-top: 0.5rem;">Risk Analysis</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="agent-card sage"><h3 class="agent-name">🎯 Sage</h3><p style="color: #1e293b; margin-top: 0.5rem;">Final Decision + Wallet</p></div>', unsafe_allow_html=True)
    
    if st.button("✨ Start A.I.D.E Processing", type="primary", use_container_width=True):
        st.session_state.step = 1
        st.rerun()

current_lang_dict = LANGUAGES[st.session_state.current_lang]

# STEP 1: PERSONAL DETAILS
if st.session_state.step == 1:
    st.header("🧠 Maya - Personal Details")
    st.markdown(f'<div class="agent-card maya"><h3>{current_lang_dict["step"].format(1)}</h3></div>', unsafe_allow_html=True)
    
    with st.form("personal_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("👤 Full Name *", placeholder="John Doe")
            age = st.number_input("🎂 Age *", min_value=18, max_value=70, value=30)
            income = st.number_input("💰 Monthly Income (₹) *", min_value=5000, value=50000, step=1000)
        with col2:
            loan_type = st.selectbox("🏦 Loan Type *", current_lang_dict['loan_types'])
            loan_amount = st.number_input("💵 Loan Amount (₹) *", min_value=25000, value=300000, step=10000)
            employment = st.number_input("💼 Employment Years", value=2)
        
        submitted = st.form_submit_button("➡️ Rex - Documents", use_container_width=True)
        if submitted and name and income > 0 and loan_amount > 0:
            english_types = ['Personal Loan', 'Home Loan', 'Car Loan', 'Business Loan']
            loan_type_en = english_types[current_lang_dict['loan_types'].index(loan_type)]
            st.session_state.user_data = {
                'name': name, 'age': age, 'income': income, 'loan_type': loan_type_en,
                'loan_amount': loan_amount, 'employment': employment, 'lang': st.session_state.current_lang
            }
            st.session_state.step = 2
            st.rerun()

# STEP 2: DOCUMENTS + SELFIE
elif st.session_state.step == 2:
    st.header("🔍 Rex - Documents + Selfie")
    st.markdown(f'<div class="agent-card rex"><h3>{current_lang_dict["step"].format(2)}</h3></div>', unsafe_allow_html=True)
    
    data = st.session_state.user_data
    st.info(f"👤 {data['name']} | 💰 ₹{data['income']:,} | {data['loan_type']} | 🏦 ₹{data['loan_amount']:,}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        id_proof = st.file_uploader("🆔 **ID Proof** (Clear face photo)", type=['jpg','png','jpeg'], key="id_proof")
        if id_proof:
            st.session_state.id_uploaded = True
            st.session_state.id_image = Image.open(id_proof)
            st.image(st.session_state.id_image, caption="🆔 ID Photo", width=200)
    with col2: st.file_uploader("💰 Income Proof", type=['pdf','jpg'], key="income_proof")
    with col3: st.file_uploader("🏦 Bank Statement", type=['pdf','jpg'], key="bank_proof")
    with col4:
        selfie = st.file_uploader("📸 **SELFIE** (Front face, good light)", type=['jpg','png','jpeg'], key="selfie")
        if selfie:
            st.session_state.selfie_uploaded = True
            st.session_state.selfie_image = Image.open(selfie)
            st.image(st.session_state.selfie_image, caption="📸 Your Selfie", width=200)
    
    if st.button("🔍 Compare Faces", type="primary", use_container_width=True, 
                disabled=not (st.session_state.get('selfie_uploaded', False) and st.session_state.get('id_uploaded', False))):
        st.session_state.step = 3
        st.rerun()

# 🔥 STEP 3: FACE RECOGNITION (HIDDEN TECHNICAL DETAILS)
elif st.session_state.step == 3:
    st.header("👤 Leo - Face Verification")
    st.markdown(f'<div class="agent-card leo"><h3>{current_lang_dict["step"].format(3)}</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1: st.image(st.session_state.id_image, caption="🆔 ID Photo", width=300)
    with col2: st.image(st.session_state.selfie_image, caption="📸 Your Selfie", width=300)
    
    with st.spinner("🔍 Analyzing facial features..."):
        face_result = compare_faces(st.session_state.id_image, st.session_state.selfie_image)
        st.session_state.face_results = face_result
    
    # 🔥 HIDDEN SIMILARITY/DISTANCE - USER SEES ONLY PASS/FAIL
    col1, col2 = st.columns(2)
    with col1:
        st.empty()  # NOTHING SHOWN - Clean UI
    with col2:
        if face_result['match']:
            st.markdown(f'<div class="pass-badge">{current_lang_dict["face_pass"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="fail-badge">{current_lang_dict["face_fail"]}</div>', unsafe_allow_html=True)
    
    if st.button("➡️ Continue Processing", use_container_width=True):
        st.session_state.step = 4
        st.rerun()

# STEPS 4-6: FAST PROCESSING
elif st.session_state.step in [4, 5, 6]:
    agents = ["🛡 Victor (Risk Analysis)", "📊 Sophia (Credit Scoring)", "🎯 Sage (Final Decision)"]
    st.header(f"{agents[st.session_state.step-4]}")
    st.markdown(f'<div class="agent-card {["victor", "sophia", "sage"][st.session_state.step-4]}"><h3>{current_lang_dict["step"].format(st.session_state.step)}</h3></div>', unsafe_allow_html=True)
    
    progress_bar = st.progress(st.session_state.step / 7)
    time.sleep(1.2)
    st.session_state.step += 1
    st.rerun()

# 🔥 STEP 7: FINAL RESULTS + WALLET DISBURSAL
elif st.session_state.step == 7:
    st.header("🎯 Sage - Final Decision + Wallet Disbursal")
    st.markdown(f'<div class="agent-card sage"><h3>{current_lang_dict["step"].format(7)}</h3></div>', unsafe_allow_html=True)
    
    data = st.session_state.user_data
    face = st.session_state.face_results
    wallet = st.session_state.payu_wallet
    
    # Credit Score Calculation
    dti = (data['loan_amount']/60) / data['income'] * 100
    credit_score = min(1000, 300 + (data['income']/50000)*400 + max(0, 200 - dti*5) + 
                      (50 if 25<=data['age']<=60 else 0) + face['confidence'])
    approved = credit_score >= 600 and face['match']
    
    rates = {"Personal Loan": 12.5, "Home Loan": 9.5, "Car Loan": 10.5, "Business Loan": 14.0}
    rate = rates.get(data['loan_type'], 12.5) if approved else 16.0
    P, r, n = data['loan_amount'], rate/1200, 60
    emi = P * r * (1+r)**n / ((1+r)**n - 1)
    
    # METRICS DISPLAY
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(current_lang_dict['credit_score'], f"{credit_score:.0f}", delta=None)
    with col2: st.metric(current_lang_dict['dti_ratio'], f"{dti:.1f}%", delta=None)
    with col3: st.metric(current_lang_dict['face_match'], "✅ PASSED" if face['match'] else "❌ FAILED", delta=None)
    with col4: st.metric(current_lang_dict['emi_label'], f"₹{round(emi):,}", delta=None)
    
    # FINAL DECISION + WALLET
    if approved:
        st.markdown(f"""
        <div class='approved'>
            <h1>{current_lang_dict['approved_title']}</h1>
            <h2>{data['loan_type']}</h2>
            <h3>₹{data['loan_amount']:,} @ {rate:.1f}%</h3>
            <h4>{current_lang_dict['emi_label']}: ₹{round(emi):,}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💰 Disburse to Wallet", type="primary", use_container_width=True):
                with st.spinner("🔄 Initiating PayU disbursal..."):
                    txn = wallet.initiate_disbursal(data['name'], data['loan_amount'])
                    st.success(f"**{current_lang_dict['payu_txn_id']}:** `{txn['txn_id']}`")
                    
                    with st.spinner("🔍 Verifying PayU settlement..."):
                        settlement = wallet.check_settlement(txn['txn_id'], data['loan_amount'])
                        
                        if settlement['status'] == 'settled':
                            balance = wallet.update_wallet(data['name'], data['loan_amount'])
                            st.markdown(f'<div class="wallet-success">{current_lang_dict["wallet_updated"]}</div>', unsafe_allow_html=True)
                            st.balloons()
                            st.metric(current_lang_dict['wallet_balance'], f"₹{balance:,}", f"+₹{data['loan_amount']:,}")
                        else:
                            st.markdown(f'<div class="wallet-pending">{current_lang_dict["settlement_pending"]}</div>', unsafe_allow_html=True)
                            st.warning("⏳ Funds not yet deposited to merchant wallet")
                            
    else:
        st.markdown(f"""
        <div class='rejected'>
            <h1>{current_lang_dict['rejected_title']}</h1>
            <h3>Credit Score: {credit_score:.0f} | Face Match: {'✅ PASSED' if face['match'] else '❌ FAILED'}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🔄 New Application", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# SIDEBAR - WALLET STATUS
with st.sidebar:
    st.markdown("### 💰 Wallet Balances")
    if st.session_state.payu_wallet.wallet_balances:
        for user, balance in st.session_state.payu_wallet.wallet_balances.items():
            st.metric(user[:15] + "...", f"₹{balance:,}")
    else:
        st.info("No wallet transactions yet")

st.markdown("---")
st.caption("🔥 **A.I.D.E** - Complete Loan Processing + PayU Wallet System ✨")
