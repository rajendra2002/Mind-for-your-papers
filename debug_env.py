
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
try:
    import fitz
    print("SUCCESS: fitz (PyMuPDF) imported.")
    print(f"fitz file: {fitz.__file__}")
except ImportError as e:
    print(f"ERROR: {e}")

try:
    import streamlit
    print("SUCCESS: streamlit imported.")
except ImportError as e:
    print(f"ERROR: streamlit not found: {e}")
