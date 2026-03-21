"""
PDF Processor — Extracts text, images, and contact info from PDFs.
Uses overlapping chunking to preserve context including phone/email/LinkedIn.
"""

import os
import io
import re
import base64
from PIL import Image
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class PDFProcessor:
    def __init__(self, pdf_path, chunk_size=500, chunk_overlap=100):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.vision_model = "llama-3.2-11b-vision-preview"

    def extract_content(self):
        """
        Extracts text, images, AND contact info from the PDF.
        Returns list of dicts: {type, content, page, metadata}
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required but not installed (module 'fitz' missing).\n"
                "Install it with: pip install PyMuPDF"
            )

        doc = fitz.open(self.pdf_path)
        extracted_data = []
        full_text_all_pages = ""

        for page_num, page in enumerate(doc):
            # -------- 1. Extract text --------
            text = page.get_text()
            if text.strip():
                full_text_all_pages += text + "\n"

                # Split into overlapping chunks
                chunks = self._chunk_text(text)
                for chunk in chunks:
                    extracted_data.append({
                        "type": "text",
                        "content": chunk,
                        "page": page_num + 1,
                        "metadata": {
                            "source": self.pdf_path,
                            "page": page_num + 1,
                        },
                    })

            # -------- 2. Extract images --------
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    if pil_image.width < 50 or pil_image.height < 50:
                        continue

                    description = self._generate_image_description(image_bytes)

                    extracted_data.append({
                        "type": "image",
                        "content": f"[IMAGE DESCRIPTION: {description}]",
                        "page": page_num + 1,
                        "metadata": {
                            "source": self.pdf_path,
                            "page": page_num + 1,
                            "image_index": img_index,
                        },
                    })
                    print(f"Processed image {img_index} on page {page_num + 1}")

                except Exception as e:
                    print(f"Error processing image {img_index} on page {page_num+1}: {e}")

        # -------- 3. Extract contact info as a dedicated chunk --------
        contact_chunk = self._extract_contact_info(full_text_all_pages)
        if contact_chunk:
            extracted_data.insert(0, {
                "type": "contact_info",
                "content": contact_chunk,
                "page": 1,
                "metadata": {
                    "source": self.pdf_path,
                    "page": 1,
                    "type": "contact_info",
                },
            })

        return extracted_data

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks for better retrieval."""
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence/paragraph boundary
            if end < len(text):
                # Look for a good break point
                for sep in ["\n\n", "\n", ". ", ", ", " "]:
                    break_pos = text.rfind(sep, start, end)
                    if break_pos > start:
                        end = break_pos + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def _extract_contact_info(self, full_text: str) -> str:
        """
        Extract phone numbers, email addresses, and LinkedIn URLs
        from the full PDF text and return as a dedicated chunk.
        """
        findings = []

        # ---- Email addresses ----
        emails = re.findall(
            r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
            full_text,
        )
        if emails:
            unique_emails = list(dict.fromkeys(emails))  # preserve order, dedupe
            for e in unique_emails:
                findings.append(f"Email: {e}")

        # ---- Phone numbers ----
        # Matches various formats: +91-1234567890, (123) 456-7890, 123-456-7890, etc.
        phones = re.findall(
            r'(?:\+?\d{1,3}[\s\-.]?)?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}',
            full_text,
        )
        if phones:
            # Filter out short matches that are likely not phone numbers
            valid_phones = []
            for p in phones:
                digits = re.sub(r'\D', '', p)
                if 7 <= len(digits) <= 15:
                    valid_phones.append(p.strip())
            unique_phones = list(dict.fromkeys(valid_phones))
            for p in unique_phones:
                findings.append(f"Phone: {p}")

        # ---- LinkedIn URLs ----
        linkedin = re.findall(
            r'(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9\-_%]+/?',
            full_text,
            re.IGNORECASE,
        )
        if linkedin:
            unique_linkedin = list(dict.fromkeys(linkedin))
            for li in unique_linkedin:
                findings.append(f"LinkedIn: {li}")

        # ---- GitHub URLs ----
        github = re.findall(
            r'(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9\-_%]+/?',
            full_text,
            re.IGNORECASE,
        )
        if github:
            unique_github = list(dict.fromkeys(github))
            for gh in unique_github:
                findings.append(f"GitHub: {gh}")

        # ---- Website URLs ----
        websites = re.findall(
            r'https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+',
            full_text,
        )
        if websites:
            # Exclude linkedin/github already captured
            other_urls = [
                w for w in websites
                if "linkedin.com" not in w.lower() and "github.com" not in w.lower()
            ]
            unique_urls = list(dict.fromkeys(other_urls))
            for url in unique_urls[:5]:  # limit to 5
                findings.append(f"Website: {url}")

        if not findings:
            return ""

        return "CONTACT INFORMATION:\n" + "\n".join(findings)

    def _generate_image_description(self, image_bytes):
        """Uses Groq Vision model to describe the image."""
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail. If it contains a table or chart, summarize the data.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model=self.vision_model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
