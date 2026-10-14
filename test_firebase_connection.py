from dotenv import load_dotenv
import os
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

credential_path = os.getenv("FIREBASE_CREDENTIALS_PATH")

if not credential_path:
    raise SystemExit("ERROR: FIREBASE_CREDENTIALS_PATH is missing from .env")

if not Path(credential_path).is_file():
    raise SystemExit("ERROR: Firebase credential file was not found")

try:
    cred = credentials.Certificate(credential_path)
    app = firebase_admin.initialize_app(cred)
    db = firestore.client(app=app)

    # Read the database metadata through a harmless collection query.
    list(db.collection("_day15_connection_test").limit(1).stream())

    print("SUCCESS: Firebase connection established.")
except Exception as exc:
    print(f"ERROR: Firebase connection failed ({type(exc).__name__}).")
