"""
High-quality face matching utility for Streamlit or CLI.

Features:
- Uses `face_recognition` encodings for robust similarity.
- Converts distance to an intuitive 0–100% confidence (lower distance = higher confidence).
- Clear, structured output for Match/No Match with confidence.
- Modular functions for easy reuse in Streamlit or scripts.
"""

from typing import Dict, Tuple
import numpy as np
import face_recognition


# --- Core utilities --------------------------------------------------------- #

def load_encoding(image_path: str, model: str = "hog") -> np.ndarray:
    """
    Load a face image and return the first face encoding.
    model: "hog" (CPU fast) or "cnn" (GPU/slow but more accurate).
    Raises ValueError if no face is found.
    """
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image, model=model)
    if not locations:
        raise ValueError(f"No face detected in: {image_path}")
    encodings = face_recognition.face_encodings(image, locations)
    return encodings[0]


def distance_to_confidence(distance: float, threshold: float = 0.6) -> float:
    """
    Convert face distance to a 0–100 confidence score.
    - distance 0.0 => 100%
    - distance == threshold => ~50%
    - distance >= 1.0 => 0%
    This is a linear mapping for interpretability in demos.
    """
    if distance < 0:
        distance = 0
    # Linear scale: 0 -> 100, threshold -> 50, clamp to [0, 100]
    conf = (1 - (distance / max(threshold, 1e-6))) * 50 + 50
    conf = max(0.0, min(100.0, conf))
    return conf


def compare_faces(
    reference_path: str, candidate_path: str, threshold: float = 0.6, model: str = "hog"
) -> Dict[str, object]:
    """
    Compare two face images and return a rich result dictionary.
    - threshold: typical face_recognition default is 0.6.
    - model: "hog" (fast CPU) or "cnn" (accurate, requires dlib with CUDA).
    """
    ref_enc = load_encoding(reference_path, model=model)
    cand_enc = load_encoding(candidate_path, model=model)

    # Euclidean distance
    distance = np.linalg.norm(ref_enc - cand_enc)
    confidence = distance_to_confidence(distance, threshold=threshold)
    is_match = distance <= threshold

    return {
        "match": bool(is_match),
        "distance": float(distance),
        "confidence": round(confidence, 2),
        "threshold": threshold,
        "model": model,
        "reference": reference_path,
        "candidate": candidate_path,
    }


# --- Pretty-print helpers --------------------------------------------------- #

def format_result(result: Dict[str, object]) -> str:
    """
    Create a clean, informative string for CLI/console.
    """
    status = "MATCH ✅" if result["match"] else "NO MATCH ❌"
    return (
        f"{status}\n"
        f"- Confidence: {result['confidence']}%\n"
        f"- Distance: {result['distance']:.4f} (threshold: {result['threshold']})\n"
        f"- Model: {result['model']}\n"
        f"- Reference: {result['reference']}\n"
        f"- Candidate: {result['candidate']}"
    )


def streamlit_display(st, result: Dict[str, object]) -> None:
    """
    Streamlit-friendly renderer. Pass `st` (the streamlit module) and the result dict.
    """
    if result["match"]:
        st.success(f"Match ✅  | Confidence: {result['confidence']}%")
    else:
        st.error(f"No Match ❌ | Confidence: {result['confidence']}%")
    st.caption(
        f"Distance: {result['distance']:.4f} (threshold {result['threshold']}), "
        f"model: {result['model']}"
    )


# --- CLI entrypoint (optional) ---------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face matcher with confidence scores")
    parser.add_argument("reference", help="Reference face image path")
    parser.add_argument("candidate", help="Candidate face image path")
    parser.add_argument("--threshold", type=float, default=0.6, help="Match threshold")
    parser.add_argument(
        "--model", choices=["hog", "cnn"], default="hog", help="face_recognition model"
    )
    args = parser.parse_args()

    try:
        res = compare_faces(
            reference_path=args.reference,
            candidate_path=args.candidate,
            threshold=args.threshold,
            model=args.model,
        )
        print(format_result(res))
    except Exception as e:
        print(f"Error: {e}")
