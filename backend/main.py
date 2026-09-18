from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import os
import re
import io
import numpy as np

from pypdf import PdfReader


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"E:\Resume_Analysis"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resume_classifier.joblib"
)

MODEL_NAME = "TF-IDF + LinearSVC"
MODEL_VERSION = "5.0.0"

TEST_ACCURACY = 99.48
STRICT_ACCURACY = 91.18


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"""
============================================================
MODEL NOT FOUND
============================================================

Expected location:

{MODEL_PATH}

Please run train_model.py first.
"""
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Resume Screening API",

    description="""
AI Resume Screening System.

Features:
- Resume text classification
- PDF resume classification
- 25 trained ML categories
- Education sector detection
- Law sector detection
- Technology sector detection
- Finance sector detection
- Engineering sector detection
- Top 5 possible categories
- Relative match scores
- Decision margin
""",

    version=MODEL_VERSION
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ResumeRequest(BaseModel):

    resume_text: str


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str):

    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SECTOR KEYWORDS
# ============================================================

SECTOR_KEYWORDS = {

    "Education": {

        "teacher": 5,
        "teaching": 5,
        "educator": 5,
        "education": 4,
        "classroom": 5,
        "lesson plan": 5,
        "lesson plans": 5,
        "curriculum": 4,
        "school": 4,
        "student": 3,
        "students": 3,
        "faculty": 4,
        "professor": 5,
        "lecturer": 5,
        "academic": 4,
        "academics": 4,
        "principal": 5,
        "head teacher": 6,
        "assistant teacher": 6,
        "associate teacher": 6,
        "teaching assistant": 6,
        "instruction": 3,
        "instructional": 4,
        "tutor": 5,
        "tutoring": 5,
        "pedagogy": 5,
        "educational": 4,
        "learning environment": 5,
        "student development": 5,
        "academic progress": 5,
        "school administration": 5,
        "college": 3,
        "university": 3
    },


    "Law": {

        "lawyer": 6,
        "attorney": 6,
        "advocate": 6,
        "legal": 4,
        "law": 4,
        "litigation": 6,
        "litigate": 6,
        "court": 5,
        "courtroom": 6,
        "judicial": 5,
        "judge": 6,
        "judiciary": 6,
        "legal research": 6,
        "legal advice": 6,
        "legal counsel": 6,
        "counsel": 4,
        "law firm": 6,
        "legal drafting": 6,
        "contract law": 6,
        "corporate law": 6,
        "criminal law": 6,
        "civil law": 5,
        "constitutional law": 6,
        "intellectual property": 5,
        "arbitration": 6,
        "mediation": 5,
        "case law": 6,
        "case management": 3,
        "pleading": 6,
        "pleadings": 6,
        "legal compliance": 5,
        "barrister": 6,
        "solicitor": 6,
        "paralegal": 6,
        "legal assistant": 6,
        "juris doctor": 6,
        "llb": 6,
        "ll.m": 6,
        "llm": 6
    },


    "Technology": {

        "software engineer": 6,
        "software developer": 6,
        "developer": 3,
        "programming": 4,
        "python": 4,
        "java": 4,
        "javascript": 4,
        "react": 4,
        "node.js": 4,
        "nodejs": 4,
        "machine learning": 6,
        "deep learning": 6,
        "artificial intelligence": 6,
        "data science": 5,
        "data scientist": 6,
        "database": 4,
        "sql": 3,
        "cloud": 3,
        "aws": 4,
        "azure": 4,
        "devops": 6,
        "docker": 4,
        "kubernetes": 4,
        "cybersecurity": 6,
        "cyber security": 6,
        "network security": 6,
        "network engineer": 5,
        "web developer": 6,
        "full stack": 6,
        "frontend": 5,
        "backend": 5,
        "api development": 5
    },


    "Finance": {

        "accountant": 6,
        "accounting": 5,
        "finance": 5,
        "financial": 4,
        "financial analysis": 6,
        "financial analyst": 6,
        "finance manager": 6,
        "accounting manager": 6,
        "corporate controller": 6,
        "controller": 4,
        "auditing": 5,
        "audit": 4,
        "budgeting": 5,
        "forecasting": 5,
        "payroll": 4,
        "accounts payable": 6,
        "accounts receivable": 6,
        "bookkeeping": 6,
        "taxation": 5,
        "tax": 4,
        "gaap": 5,
        "ifrs": 5,
        "sap": 3
    },


    "Engineering": {

        "engineer": 3,
        "engineering": 4,
        "civil engineer": 6,
        "mechanical engineer": 6,
        "electrical engineer": 6,
        "electronics engineer": 6,
        "structural engineer": 6,
        "site engineer": 6,
        "design engineer": 6,
        "mechanical design": 6,
        "autocad": 4,
        "solidworks": 5,
        "catia": 5,
        "matlab": 4,
        "construction": 4,
        "manufacturing": 4,
        "production engineering": 6
    },


    "Human Resources": {

        "human resources": 6,
        "human resource": 6,
        "hr manager": 6,
        "hr officer": 6,
        "recruitment": 5,
        "recruiting": 5,
        "talent acquisition": 6,
        "employee relations": 6,
        "payroll": 3,
        "performance management": 5,
        "training and development": 5
    },


    "Marketing & Sales": {

        "marketing": 5,
        "marketing manager": 6,
        "sales manager": 6,
        "sales executive": 6,
        "business development": 5,
        "business development executive": 6,
        "digital marketing": 6,
        "seo": 5,
        "social media marketing": 6,
        "brand management": 5,
        "customer acquisition": 5
    }
}


# ============================================================
# SECTOR DETECTION
# ============================================================

def detect_sector(text: str):

    text_lower = text.lower()

    sector_scores = {}

    matched_keywords = {}

    for sector, keywords in SECTOR_KEYWORDS.items():

        score = 0

        matches = []

        for keyword, weight in keywords.items():

            # Word/phrase matching
            if keyword in text_lower:

                score += weight

                matches.append(keyword)

        sector_scores[sector] = score

        matched_keywords[sector] = matches


    # Sort sectors
    ranked_sectors = sorted(
        sector_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # Best sector
    best_sector = ranked_sectors[0][0]

    best_score = ranked_sectors[0][1]


    # Second sector
    if len(ranked_sectors) > 1:

        second_sector = ranked_sectors[1][0]

        second_score = ranked_sectors[1][1]

    else:

        second_sector = None

        second_score = 0


    # --------------------------------------------------------
    # Sector confidence
    # --------------------------------------------------------

    if best_score == 0:

        sector_confidence = 0

        detected_sector = "General / Unknown"

    else:

        total_score = sum(
            sector_scores.values()
        )

        if total_score > 0:

            sector_confidence = (
                best_score /
                total_score
            ) * 100

        else:

            sector_confidence = 0

        detected_sector = best_sector


    # --------------------------------------------------------
    # Top sectors
    # --------------------------------------------------------

    top_sectors = []

    for sector, score in ranked_sectors[:3]:

        if score <= 0:
            continue

        top_sectors.append({

            "sector": sector,

            "score": score,

            "matched_keywords":
                matched_keywords[sector][:10]
        })


    return {

        "sector":
            detected_sector,

        "sector_score":
            best_score,

        "sector_confidence":
            round(
                sector_confidence,
                2
            ),

        "alternative_sectors":
            top_sectors
    }


# ============================================================
# ROLE DETECTION FOR SPECIAL SECTORS
# ============================================================

def detect_specialized_role(
    text: str,
    sector: str
):

    text_lower = text.lower()


    # ========================================================
    # EDUCATION
    # ========================================================

    if sector == "Education":

        education_roles = {

            "Teacher": [
                "teacher",
                "classroom",
                "lesson plan",
                "lesson plans",
                "teaching"
            ],

            "Associate Teacher": [
                "associate teacher"
            ],

            "Professor / Lecturer": [
                "professor",
                "lecturer"
            ],

            "Tutor": [
                "tutor",
                "tutoring"
            ],

            "Academic Coordinator": [
                "academic coordinator",
                "academic coordination"
            ],

            "Principal / School Administrator": [
                "principal",
                "school administrator"
            ],

            "Teaching Assistant": [
                "teaching assistant",
                "teacher assistant"
            ]
        }


        role_scores = {}

        for role, keywords in education_roles.items():

            score = 0

            for keyword in keywords:

                if keyword in text_lower:

                    score += 1


            role_scores[role] = score


        best_role = max(
            role_scores,
            key=role_scores.get
        )


        if role_scores[best_role] > 0:

            return best_role


        return "Education Professional"


    # ========================================================
    # LAW
    # ========================================================

    if sector == "Law":

        law_roles = {

            "Lawyer / Advocate": [
                "lawyer",
                "attorney",
                "advocate"
            ],

            "Legal Counsel": [
                "legal counsel",
                "corporate counsel"
            ],

            "Litigation Lawyer": [
                "litigation",
                "litigate",
                "courtroom"
            ],

            "Legal Researcher": [
                "legal research",
                "legal researcher"
            ],

            "Paralegal": [
                "paralegal",
                "legal assistant"
            ],

            "Judge / Judicial Professional": [
                "judge",
                "judicial",
                "judiciary"
            ]
        }


        role_scores = {}

        for role, keywords in law_roles.items():

            score = 0

            for keyword in keywords:

                if keyword in text_lower:

                    score += 1


            role_scores[role] = score


        best_role = max(
            role_scores,
            key=role_scores.get
        )


        if role_scores[best_role] > 0:

            return best_role


        return "Legal Professional"


    return None


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_resume(resume_text: str):

    resume_text = clean_text(
        resume_text
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(resume_text) < 20:

        raise HTTPException(
            status_code=400,

            detail=(
                "Resume text is too short. "
                "Please provide a valid resume."
            )
        )


    # ========================================================
    # ML PREDICTION
    # ========================================================

    ml_prediction = model.predict(
        [resume_text]
    )[0]


    decision_scores = model.decision_function(
        [resume_text]
    )


    if len(decision_scores.shape) == 2:

        scores = decision_scores[0]

    else:

        scores = decision_scores


    classes = model.classes_


    # ========================================================
    # RANKING
    # ========================================================

    ranked_indices = scores.argsort()[::-1]


    best_index = ranked_indices[0]

    best_score = float(
        scores[best_index]
    )


    # ========================================================
    # SECOND BEST
    # ========================================================

    if len(ranked_indices) > 1:

        second_score = float(
            scores[
                ranked_indices[1]
            ]
        )

    else:

        second_score = 0.0


    # ========================================================
    # DECISION MARGIN
    # ========================================================

    decision_margin = (
        best_score -
        second_score
    )


    # ========================================================
    # CONFIDENCE INDICATOR
    # ========================================================

    score_min = float(
        scores.min()
    )

    score_max = float(
        scores.max()
    )

    score_range = (
        score_max -
        score_min
    )


    if score_range > 0:

        confidence_indicator = (

            (
                best_score -
                score_min
            )
            /
            score_range

        ) * 100

    else:

        confidence_indicator = 0.0


    # ========================================================
    # RELATIVE MATCH SCORES
    # ========================================================

    shifted_scores = (
        scores -
        scores.max()
    )


    exp_scores = np.exp(
        shifted_scores
    )


    match_scores = (

        exp_scores /
        exp_scores.sum()

    ) * 100


    # ========================================================
    # TOP 5 ML CATEGORIES
    # ========================================================

    possible_categories = []


    for rank, index in enumerate(
        ranked_indices[:5],
        start=1
    ):

        possible_categories.append({

            "rank":
                rank,

            "category":
                str(classes[index]),

            "match_score":
                round(
                    float(
                        match_scores[index]
                    ),
                    2
                ),

            "decision_score":
                round(
                    float(scores[index]),
                    4
                )
        })


    # ========================================================
    # SECTOR INTELLIGENCE
    # ========================================================

    sector_result = detect_sector(
        resume_text
    )


    detected_sector = (
        sector_result["sector"]
    )


    # ========================================================
    # SPECIALIZED ROLE
    # ========================================================

    specialized_role = detect_specialized_role(
        resume_text,
        detected_sector
    )


    # ========================================================
    # FINAL CATEGORY
    # ========================================================

    primary_category = str(
        ml_prediction
    )


    # ========================================================
    # SECTOR-AWARE RESULT
    # ========================================================

    sector_override = False


    if specialized_role is not None:

        # Education / Law are not sufficiently
        # represented in the 25-category dataset.
        #
        # Therefore we expose the sector-specific
        # classification separately rather than
        # pretending it came from the ML model.

        sector_override = True


    # ========================================================
    # ALTERNATIVE MESSAGE
    # ========================================================

    alternatives = [

        item["category"]

        for item in possible_categories[1:3]
    ]


    if len(alternatives) >= 2:

        ml_message = (

            f"ML primary match: "
            f"{primary_category}. "

            f"The model may also relate this resume "
            f"to {alternatives[0]} or "
            f"{alternatives[1]}."
        )

    elif len(alternatives) == 1:

        ml_message = (

            f"ML primary match: "
            f"{primary_category}. "

            f"The model may also relate this resume "
            f"to {alternatives[0]}."
        )

    else:

        ml_message = (

            f"ML primary match: "
            f"{primary_category}."
        )


    # ========================================================
    # PREDICTION STRENGTH
    # ========================================================

    if decision_margin < 0.10:

        prediction_strength = "Close match"

    elif decision_margin < 0.25:

        prediction_strength = "Moderate match"

    else:

        prediction_strength = "Strong match"


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "status":
            "classified",

        # ----------------------------------------------------
        # ML RESULT
        # ----------------------------------------------------

        "predicted_category":
            primary_category,

        "match_score":
            round(
                float(
                    match_scores[best_index]
                ),
                2
            ),

        "confidence_indicator":
            round(
                float(
                    confidence_indicator
                ),
                2
            ),

        "decision_margin":
            round(
                float(
                    decision_margin
                ),
                4
            ),

        "prediction_strength":
            prediction_strength,

        "possible_categories":
            possible_categories,

        # ----------------------------------------------------
        # SECTOR RESULT
        # ----------------------------------------------------

        "detected_sector":
            detected_sector,

        "sector_confidence":
            sector_result[
                "sector_confidence"
            ],

        "specialized_role":
            specialized_role,

        "sector_override":
            sector_override,

        "sector_matches":
            sector_result[
                "alternative_sectors"
            ],

        # ----------------------------------------------------
        # MESSAGE
        # ----------------------------------------------------

        "message":
            ml_message,

        "resume_length":
            len(resume_text)
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def home():

    return {

        "status":
            "online",

        "service":
            "AI Resume Screening API",

        "model":
            MODEL_NAME,

        "version":
            MODEL_VERSION,

        "ml_categories":
            len(model.classes_),

        "sector_detection":
            True,

        "supported_sectors": [

            "Education",

            "Law",

            "Technology",

            "Finance",

            "Engineering",

            "Human Resources",

            "Marketing & Sales"
        ],

        "endpoints": [

            "/",

            "/health",

            "/model-info",

            "/predict",

            "/predict-pdf"
        ]
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "model_loaded":
            True,

        "sector_detection":
            True,

        "model":
            MODEL_NAME,

        "categories":
            len(model.classes_)
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "model":
            MODEL_NAME,

        "version":
            MODEL_VERSION,

        "dataset":
            "UpdatedResumeDataSet_clean.csv",

        "test_accuracy":
            f"{TEST_ACCURACY:.2f}%",

        "strict_duplicate_free_accuracy":
            f"{STRICT_ACCURACY:.2f}%",

        "ml_categories":
            len(model.classes_),

        "sector_detection":
            True,

        "sector_categories": [

            "Education",

            "Law",

            "Technology",

            "Finance",

            "Engineering",

            "Human Resources",

            "Marketing & Sales"
        ],

        "category_names": [

            str(category)

            for category in model.classes_
        ],

        "note":
            (
                "ML match scores are relative decision-score "
                "distributions and are not calibrated probabilities. "
                "Sector detection is an additional rule-based "
                "intelligence layer."
            )
    }


# ============================================================
# TEXT PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    request: ResumeRequest
):

    result = predict_resume(
        request.resume_text
    )

    result["input_type"] = "text"

    return result


# ============================================================
# PDF PREDICTION
# ============================================================

@app.post("/predict-pdf")
async def predict_pdf(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,

            detail="No file was provided."
        )


    # --------------------------------------------------------
    # Validate PDF
    # --------------------------------------------------------

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,

            detail=(
                "Only PDF files are supported."
            )
        )


    try:

        # ====================================================
        # READ PDF
        # ====================================================

        contents = await file.read()


        if not contents:

            raise HTTPException(
                status_code=400,

                detail=(
                    "Uploaded PDF is empty."
                )
            )


        # ====================================================
        # PDF READER
        # ====================================================

        reader = PdfReader(
            io.BytesIO(contents)
        )


        # ====================================================
        # EXTRACT TEXT
        # ====================================================

        extracted_text = ""


        for page in reader.pages:

            page_text = page.extract_text()


            if page_text:

                extracted_text += (
                    page_text +
                    "\n"
                )


        # ====================================================
        # CLEAN
        # ====================================================

        extracted_text = clean_text(
            extracted_text
        )


        # ====================================================
        # VALIDATE
        # ====================================================

        if len(extracted_text) < 20:

            raise HTTPException(
                status_code=400,

                detail=(
                    "Could not extract enough text "
                    "from this PDF. The PDF may be "
                    "scanned or image-based."
                )
            )


        # ====================================================
        # PREDICT
        # ====================================================

        result = predict_resume(
            extracted_text
        )


        # ====================================================
        # PDF INFORMATION
        # ====================================================

        result["filename"] = (
            file.filename
        )

        result["input_type"] = "PDF"

        result["pages"] = len(
            reader.pages
        )


        return result


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                f"PDF processing error: {str(e)}"
            )
        )


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="127.0.0.1",

        port=8000,

        reload=True
    )
