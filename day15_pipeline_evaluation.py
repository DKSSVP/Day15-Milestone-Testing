
import os
import json
import re
import uuid
import requests
from datetime import datetime
from dotenv import load_dotenv
from jsonschema import validate, ValidationError
import firebase_admin
from firebase_admin import credentials, firestore


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SHEETDB_API_URL = os.getenv("SHEETDB_API_URL")
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")

MODEL_NAME = "openai/gpt-4o"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_REPAIR_ATTEMPTS = 2
AUDIT_COLLECTION = "day15_audit_logs"


# --------------------------------------------------
# 2. JSON SCHEMA
# --------------------------------------------------

RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": ["string", "null"]},
        "email": {"type": ["string", "null"]},
        "phone": {"type": ["string", "null"]},
        "pan": {"type": ["string", "null"]},
        "dob": {"type": ["string", "null"]},
    },
    "required": ["name", "email", "phone", "pan", "dob"],
    "additionalProperties": False,
}


# --------------------------------------------------
# 3. FIREBASE INITIALIZATION
# --------------------------------------------------

def initialize_firebase():
    """Initialize Firebase and return the Firestore client."""

    if not FIREBASE_CREDENTIALS_PATH:
        raise RuntimeError(
            "FIREBASE_CREDENTIALS_PATH is missing from .env."
        )

    if not os.path.isfile(FIREBASE_CREDENTIALS_PATH):
        raise RuntimeError(
            "Firebase credential file was not found."
        )

    try:
        app = firebase_admin.get_app()
    except ValueError:
        firebase_credential = credentials.Certificate(
            FIREBASE_CREDENTIALS_PATH
        )
        app = firebase_admin.initialize_app(firebase_credential)

    return firestore.client(app=app)


# --------------------------------------------------
# 4. VALIDATION
# --------------------------------------------------

def validate_resume_data(data):
    """Validate JSON structure, PAN format, and DOB."""

    try:
        validate(instance=data, schema=RESUME_SCHEMA)
    except ValidationError as exc:
        raise ValueError(
            f"Schema validation failed: {exc.message}"
        ) from exc

    pan = data["pan"]

    if pan is not None:
        if not isinstance(pan, str) or not re.fullmatch(
            r"[A-Z]{5}[0-9]{4}[A-Z]", pan
        ):
            raise ValueError("Invalid PAN format.")

    dob = data["dob"]

    if dob is not None:
        if not isinstance(dob, str):
            raise ValueError("DOB must be a string or null.")

        try:
            parsed_dob = datetime.strptime(dob, "%d/%m/%Y")

            if parsed_dob.strftime("%d/%m/%Y") != dob:
                raise ValueError("DOB must use DD/MM/YYYY.")

            if parsed_dob.date() > datetime.today().date():
                raise ValueError("DOB cannot be in the future.")

        except ValueError as exc:
            raise ValueError(
                "Invalid DOB. Use a real date in DD/MM/YYYY format."
            ) from exc

    return True


# --------------------------------------------------
# 5. OPENROUTER API
# --------------------------------------------------

def call_openrouter(messages):
    """Call OpenRouter and return the model's response text."""

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing from .env.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 500,
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )

        if not response.ok:
            try:
                error_data = response.json().get("error", {})
                if not isinstance(error_data, dict):
                    error_data = {}

                message = error_data.get(
                    "message", "Request failed"
                )
            except ValueError:
                message = "Non-JSON error response"

            print(f"OpenRouter HTTP status: {response.status_code}")
            print(f"OpenRouter message: {str(message)[:250]}")
            response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Model returned empty content.")

        return content

    except requests.RequestException as exc:
        raise RuntimeError(
            f"OpenRouter request failed: {type(exc).__name__}"
        ) from exc

    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid API response: {type(exc).__name__}"
        ) from exc


# --------------------------------------------------
# 6. RECURSIVE REPAIR
# --------------------------------------------------

def extract_and_validate_with_repair(
    content, resume_text, attempts_left=MAX_REPAIR_ATTEMPTS
):
    """Parse, validate, and recursively repair invalid model output."""

    try:
        candidate = json.loads(content)
        validate_resume_data(candidate)
        return candidate

    except (json.JSONDecodeError, ValueError) as exc:
        if attempts_left <= 0:
            raise ValueError(
                "Invalid model output after repair attempts."
            ) from exc

        attempt_number = MAX_REPAIR_ATTEMPTS - attempts_left + 1
        print(f"Requesting model repair ({attempt_number}/"
              f"{MAX_REPAIR_ATTEMPTS})...")

        repair_messages = [
            {
                "role": "system",
                "content": (
                    "Repair extracted resume data. "
                    "Return exactly one valid JSON object."
                ),
            },
            {
                "role": "user",
                "content": f"""
The previous response failed validation.

Validation error: {str(exc)[:300]}

Required JSON schema:
{json.dumps(RESUME_SCHEMA)}

Rules:
- Include every required key.
- Use null for missing fields.
- Do not invent information.
- PAN must contain five uppercase letters, four digits,
  and one uppercase letter.
- DOB must be a real date in DD/MM/YYYY format
  and cannot be in the future.
- Return JSON only without Markdown fences.

Original resume:
{resume_text}

Previous response:
{content}
""",
            },
        ]

        repaired_content = call_openrouter(repair_messages)

        return extract_and_validate_with_repair(
            repaired_content,
            resume_text,
            attempts_left - 1,
        )


def extract_resume_data(resume_text):
    """Extract resume fields and validate the result."""

    messages = [
        {
            "role": "system",
            "content": (
                "Extract resume information accurately. "
                "Return one valid JSON object only."
            ),
        },
        {
            "role": "user",
            "content": f"""
Extract name, email, phone, pan, and dob.

Include all five keys. Use null for missing values.
Do not invent information. Use DD/MM/YYYY for DOB.
Return JSON only, without Markdown fences.

Resume:
{resume_text}
""",
        },
    ]

    content = call_openrouter(messages)

    return extract_and_validate_with_repair(content, resume_text)


# --------------------------------------------------
# 7. FIREBASE AUDIT LOGGING
# --------------------------------------------------

def write_audit_log(db, transaction_id, status, error_type=None,
                    http_status=None):
    """
    Write an audit record for a SheetDB transaction.
    No candidate PAN, DOB, API key, or full resume is logged.
    """

    audit_record = {
        "transaction_id": transaction_id,
        "status": status,
        "error_type": error_type,
        "sheetdb_http_status": http_status,
        "model": MODEL_NAME,
        "timestamp": firestore.SERVER_TIMESTAMP,
    }

    db.collection(AUDIT_COLLECTION).document(
        transaction_id
    ).set(audit_record)


# --------------------------------------------------
# 8. SHEETDB TRANSACTION
# --------------------------------------------------

def push_to_sheetdb(candidate):
    """Submit one validated candidate to the configured SheetDB URL."""

    if not SHEETDB_API_URL:
        raise RuntimeError("SHEETDB_API_URL is missing from .env.")

    payload = {"data": [candidate]}

    response = requests.post(
        SHEETDB_API_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )

    if not response.ok:
        raise requests.HTTPError(
            f"SheetDB returned HTTP {response.status_code}",
            response=response,
        )

    return response.status_code


# --------------------------------------------------
# 9. END-TO-END PROCESSING
# --------------------------------------------------

def process_resume(resume_text, db):
    """Extract, validate, write to SheetDB, and audit the result."""

    transaction_id = str(uuid.uuid4())
    sheetdb_status = "fail"
    error_type = None
    http_status = None

    try:
        candidate = extract_resume_data(resume_text)
        validate_resume_data(candidate)

        # Attempt the SheetDB transaction.
        http_status = push_to_sheetdb(candidate)
        sheetdb_status = "pass"

        print("SheetDB submission successful.")

        return {
            "transaction_id": transaction_id,
            "status": "pass",
            "candidate": candidate,
        }

    except (RuntimeError, ValueError, requests.RequestException) as exc:
        error_type = type(exc).__name__
        print(f"Transaction failed: {error_type}")

        return {
            "transaction_id": transaction_id,
            "status": "fail",
            "error_type": error_type,
        }

    finally:
        # Audit both successful and failed SheetDB attempts.
        # If extraction fails before SheetDB is reached, this also
        # records the pipeline failure for diagnosis.
        try:
            write_audit_log(
                db=db,
                transaction_id=transaction_id,
                status=sheetdb_status,
                error_type=error_type,
                http_status=http_status,
            )
            print("Firebase audit log written.")

        except Exception as exc:
            # Never claim the audit was recorded if it failed.
            print(
                "CRITICAL: Firebase audit logging failed "
                f"({type(exc).__name__})."
            )


# --------------------------------------------------
# 10. LOCAL VALIDATION TESTS
# --------------------------------------------------

def run_validation_tests():
    valid_data = {
        "name": "Damodhara Prasad",
        "email": "damodhara.prasad@craftechco.co",
        "phone": "9876543210",
        "pan": "ABCDE1234F",
        "dob": "14/10/2002",
    }

    validate_resume_data(valid_data)
    print("PASS: Valid candidate data accepted.")

    cases = [
        (
            "Missing email",
            {k: v for k, v in valid_data.items() if k != "email"},
        ),
        ("Invalid PAN", {**valid_data, "pan": "BADPAN"}),
        ("Impossible DOB", {**valid_data, "dob": "31/02/1998"}),
        ("Future DOB", {**valid_data, "dob": "01/01/2099"}),
    ]

    for label, test_data in cases:
        try:
            validate_resume_data(test_data)
            print(f"FAIL: {label} was accepted.")
        except ValueError:
            print(f"PASS: {label} rejected.")


# --------------------------------------------------
# 11. MAIN
# --------------------------------------------------

if __name__ == "__main__":
    run_validation_tests()

    try:
        firestore_client = initialize_firebase()
        print("Firebase initialized successfully.")

        sample_resume = """
        Name: Damodhara Prasad
        Email: damodhara.prasad@craftechco.co
        Phone: 9876543210
        PAN: ABCDE1234F
        DOB: 14/10/2002
        """

        result = process_resume(sample_resume, firestore_client)

        print("\nPipeline result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as exc:
        print(f"Pipeline setup failed: {type(exc).__name__}")