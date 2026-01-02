import os
import json
import time
from openai import OpenAI, OpenAIError
from pinecone import Pinecone
import google.generativeai as genai
import httpx
import base64

def extract_text_from_pdf(pdf_path, gemini_api_key):
    """
    Extract text from PDF using Google Gemini API with comprehensive formatting
    """
    try:
        genai.configure(api_key=gemini_api_key)
        
        # Upload the PDF file to Gemini
        print(f"Uploading PDF: {os.path.basename(pdf_path)}")
        uploaded_file = genai.upload_file(pdf_path)
        
        # Wait for file to be processed
        print("⏳ Waiting for file to be processed...")
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            raise ValueError(f"File processing failed: {uploaded_file.state.name}")
        
        generation_config = {
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "object",
                "properties": {
                    "pages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "page_number": {"type": "integer"},
                                "content": {"type": "string"}
                            },
                            "required": ["page_number", "content"]
                        }
                    }
                },
                "required": ["pages"]
            }
        }
        
        # Use Gemini to extract text with comprehensive formatting
        model = genai.GenerativeModel(
            model_name='gemini-2.5-pro',
            generation_config=generation_config
            )
        
        prompt = """Extract ALL text from this document with clear formatting and logical structure.

📋 FORMATTING GUIDELINES:

1. PAGE MARKERS:
   Start each page with a clear separator:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   📄 PAGE [NUMBER]
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    NOTE: Use the absolute physical page number (1 for the first page, 2 for the second, etc.) regardless of any page numbers printed on the document itself.

2. HIERARCHY & STRUCTURE:
   - Document title: # TITLE (use single #)
   - Major sections: ## SECTION NAME (use double ##)
   - Subsections: ### SUBSECTION NAME (use triple ###)
   - Always add blank lines before and after headings

3. TABLES:
   Mark tables clearly and preserve structure:
   📊 TABLE: [Brief description if title exists]
   | Column 1    | Column 2    | Column 3    |
   |-------------|-------------|-------------|
   | Data 1      | Data 2      | Data 3      |
   | Data 4      | Data 5      | Data 6      |
   
   Add blank line before and after tables.

4. LISTS:
   - Numbered lists: Use 1., 2., 3., etc.
   - Bullet points: Use • or - consistently
   - Indent sub-items with 2-4 spaces
   - Keep list items together (don't break mid-list)

5. SPECIAL ELEMENTS:
   Mark different content types clearly:
   - Warnings: ⚠️ WARNING: [text]
   - Cautions: ⚠️ CAUTION: [text]
   - Important notes: 📌 NOTE: [text]
   - Tips: 💡 TIP: [text]
   - Instructions: 📝 INSTRUCTIONS:
   - Examples: 💬 EXAMPLE:

6. VISUAL CONTENT:
   Describe non-text elements:
   - Images: 🖼️ [IMAGE: Brief description]
   - Diagrams: 📊 [DIAGRAM: What it shows]
   - Charts: 📈 [CHART: Type and content]
   - Photos: 📷 [PHOTO: Subject]
   - Icons/symbols: [ICON: Description]

7. DATA & SPECIFICATIONS:
   For key-value pairs, use consistent format:
   Property: Value
   Another Property: Another Value
   
   Group related information together with blank lines between groups.

8. PROCEDURES & STEPS:
   For sequential instructions:
   ## Procedure Name
   
   1. First step description
      - Sub-point if needed
      - Another sub-point
   
   2. Second step description
   
   3. Third step description

9. CONTACT INFORMATION:
   Mark clearly:
   📞 CONTACT INFORMATION
   - Phone: [number]
   - Email: [address]
   - Website: [url]

10. LEGAL/WARRANTY TEXT:
    Mark sections:
    📜 [SECTION TYPE: Warranty/Terms/Legal]
    Keep original numbering (1., a., i., etc.)

11. SEMANTIC ORDER:
    - Follow natural reading order (top-to-bottom, left-to-right)
    - Keep related content together
    - Don't split tables, lists, or procedures
    - Maintain logical flow of information

12. SPACING & READABILITY:
    - Blank line between different sections
    - Blank line before/after tables
    - Blank line before/after special callouts
    - Blank line before/after lists
    - No excessive blank lines (max 2 in a row)

13. TEXT EXTRACTION:
    - Extract text from images using OCR
    - Extract text from embedded screenshots
    - Extract data from charts/graphs if text is visible
    - Preserve text in headers/footers if important
    - Include text in watermarks if relevant

14. FORMATTING CONSISTENCY:
    - Use the same style throughout the document
    - Be consistent with bullets (• or -)
    - Be consistent with emphasis markers
    - Maintain consistent indentation

IMPORTANT:
- DO NOT skip any content
- DO NOT summarize - extract everything verbatim
- DO maintain logical structure and readability
- DO describe visual elements that contain information
- DO preserve the semantic meaning and organization

Begin extraction now."""
        
        print("🔍 Extracting content from PDF...")
        response = model.generate_content([uploaded_file, prompt])
        
        # Clean up the uploaded file
        try:
            genai.delete_file(uploaded_file.name)
            print("✅ Temporary file deleted from Gemini servers")
        except:
            print("⚠️ Could not delete temporary file (not critical)")
        
        return response.text
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_pdf_openai(pdf_path, openai_api_key):
    """
    Extract text from PDF using OpenAI API with comprehensive formatting
    """
    try:
        client = OpenAI(api_key=openai_api_key)
        
        print(f"Uploading PDF: {os.path.basename(pdf_path)}")
        
        with open(pdf_path, "rb") as f:
            data = f.read()
        
        base64_string = base64.b64encode(data).decode("utf-8")
        
        print("🔍 Extracting content from PDF...")
        
        response = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": os.path.basename(pdf_path),
                            "file_data": f"data:application/pdf;base64,{base64_string}",
                        },
                        {
                            "type": "input_text",
                            "text": """Extract ALL text from this document with clear formatting and logical structure.
 
📋 FORMATTING GUIDELINES:
 
1. PAGE MARKERS:
   Start each page with a clear separator:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   📄 PAGE [NUMBER]
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    NOTE: Use the absolute physical page number (1 for the first page, 2 for the second, etc.) regardless of any page numbers printed on the document itself.
 
2. HIERARCHY & STRUCTURE:
   - Document title: # TITLE (use single #)
   - Major sections: ## SECTION NAME (use double ##)
   - Subsections: ### SUBSECTION NAME (use triple ###)
   - Always add blank lines before and after headings
 
3. TABLES:
   Mark tables clearly and preserve structure:
   📊 TABLE: [Brief description if title exists]
   | Column 1    | Column 2    | Column 3    |
   |-------------|-------------|-------------|
   | Data 1      | Data 2      | Data 3      |
   | Data 4      | Data 5      | Data 6      |
   
   Add blank line before and after tables.
 
4. LISTS:
   - Numbered lists: Use 1., 2., 3., etc.
   - Bullet points: Use • or - consistently
   - Indent sub-items with 2-4 spaces
   - Keep list items together (don't break mid-list)
 
5. SPECIAL ELEMENTS:
   Mark different content types clearly:
   - Warnings: ⚠️ WARNING: [text]
   - Cautions: ⚠️ CAUTION: [text]
   - Important notes: 📌 NOTE: [text]
   - Tips: 💡 TIP: [text]
   - Instructions: 📝 INSTRUCTIONS:
   - Examples: 💬 EXAMPLE:
 
6. VISUAL CONTENT:
   Describe non-text elements:
   - Images: 🖼️ [IMAGE: Brief description]
   - Diagrams: 📊 [DIAGRAM: What it shows]
   - Charts: 📈 [CHART: Type and content]
   - Photos: 📷 [PHOTO: Subject]
   - Icons/symbols: [ICON: Description]
 
7. DATA & SPECIFICATIONS:
   For key-value pairs, use consistent format:
   Property: Value
   Another Property: Another Value
   
   Group related information together with blank lines between groups.
 
8. PROCEDURES & STEPS:
   For sequential instructions:
   ## Procedure Name
   
   1. First step description
      - Sub-point if needed
      - Another sub-point
   
   2. Second step description
   
   3. Third step description
 
9. CONTACT INFORMATION:
   Mark clearly:
   📞 CONTACT INFORMATION
   - Phone: [number]
   - Email: [address]
   - Website: [url]
 
10. LEGAL/WARRANTY TEXT:
    Mark sections:
    📜 [SECTION TYPE: Warranty/Terms/Legal]
    Keep original numbering (1., a., i., etc.)
 
11. SEMANTIC ORDER:
    - Follow natural reading order (top-to-bottom, left-to-right)
    - Keep related content together
    - Don't split tables, lists, or procedures
    - Maintain logical flow of information
 
12. SPACING & READABILITY:
    - Blank line between different sections
    - Blank line before/after tables
    - Blank line before/after special callouts
    - Blank line before/after lists
    - No excessive blank lines (max 2 in a row)
 
13. TEXT EXTRACTION:
    - Extract text from images using OCR
    - Extract text from embedded screenshots
    - Extract data from charts/graphs if text is visible
    - Preserve text in headers/footers if important
    - Include text in watermarks if relevant
 
14. FORMATTING CONSISTENCY:
    - Use the same style throughout the document
    - Be consistent with bullets (• or -)
    - Be consistent with emphasis markers
    - Maintain consistent indentation
 
IMPORTANT:
- DO NOT skip any content
- DO NOT summarize - extract everything verbatim
- DO maintain logical structure and readability
- DO describe visual elements that contain information
- DO preserve the semantic meaning and organization
 
Begin extraction now.""",
                        },
                    ],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "pdf_extraction",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "pages": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "page_number": {"type": "integer"},
                                        "content": {"type": "string"}
                                    },
                                    "required": ["page_number", "content"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["pages"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        
        print("✅ PDF content extracted successfully")
        return response.output_text
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def chunk_text(pages_data, chunk_size=1000, overlap=400):
    """
    Split text into overlapping chunks for better context preservation
    """
    if not pages_data:
        return []
    
    all_chunks = []
    
    for page in pages_data:
        page_number = page.get("page_number")
        content = page.get("content", "").strip()
        
        if not content:
            continue
        
        content_length = len(content)

        if content_length <= chunk_size:
            all_chunks.append({
                "text": content,
                "page_number": page_number,
            })
        else:
            # Apply the original chunking logic to this page's content
            page_chunks = []
            start = 0
            content_length = len(content)
            
            while start < content_length:
                end = min(start + chunk_size, content_length)
                
                chunk_text = content[start:end].strip()
                if chunk_text:
                    page_chunks.append(chunk_text)
                
                # Move to next chunk with overlap, ensuring forward progress
                if end >= content_length:
                    break
                start = end - overlap
            
            for chunk_text in page_chunks:
                all_chunks.append({
                    "text": chunk_text,
                    "page_number": page_number,
                })
    
    return all_chunks

def embed_text(text, openai_api_key, model="text-embedding-3-small"):
    """
    Create embedding for text using OpenAI
    """
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except OpenAIError as e:
        print(f"OpenAI embedding error: {e}")
        return None

def upload_to_pinecone(chunks, embeddings, pdf_filename, pinecone_api_key, pinecone_index_name):
    """
    Upload chunks and their embeddings to Pinecone
    """
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        
        vectors_to_upsert = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{pdf_filename}_chunk_{i}"
            
            metadata = {
                "text": chunk["text"],
                "source": pdf_filename,
                "chunk_index": i,
                "page_number": chunk["page_number"]
            }
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
        
        return True
        
    except Exception as e:
        print(f"Error uploading to Pinecone: {e}")
        return False

def process_pdf_and_upload(pdf_path, gemini_api_key, openai_api_key, pinecone_api_key, pinecone_index_name, use_gemini=False):
    """
    Main pipeline: Extract text from PDF, chunk it, embed, and upload to Pinecone
    """
    try:
        # Get filename for metadata
        pdf_filename = os.path.basename(pdf_path)
        print(f"Processing {pdf_filename}...")
        
        # Step 1: Extract text using selected API
        print("Extracting text from PDF...")
        if use_gemini:
            json_response = extract_text_from_pdf(pdf_path, gemini_api_key)
        else:
            json_response = extract_text_from_pdf_openai(pdf_path, openai_api_key)
        
        if not json_response:
            print("Failed to extract text from PDF")
            return False
        
        try:
            parsed_data = json.loads(json_response)
            pages_data = parsed_data.get("pages", [])
            
            if not pages_data:
                print("No pages found in JSON response")
                return False
            
            print(f"Extracted {len(pages_data)} pages")
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return False
        
        # Step 2: Chunk the text
        print("Chunking text...")
        try:
            chunks = chunk_text(pages_data, chunk_size=1000, overlap=400)
            print(f"Created {len(chunks)} chunks")
        except Exception as chunk_error:
            print(f"Error during chunking: {chunk_error}")
            import traceback
            traceback.print_exc()
            return False
        
        if not chunks:
            print("No chunks created")
            return False
        
        # Step 3: Create embeddings for each chunk
        print("Creating embeddings...")
        embeddings = []
        for i, chunk_dict in enumerate(chunks):
            if i % 10 == 0:
                print(f"  Embedding chunk {i+1}/{len(chunks)} (Page {chunk_dict['page_number']})...")
            
            embedding = embed_text(chunk_dict["text"], openai_api_key)
            if embedding:
                embeddings.append(embedding)
            else:
                print(f"Failed to create embedding for chunk {i} (Page {chunk_dict['page_number']})")
                return False
        
        print(f"Created {len(embeddings)} embeddings")
        
        # Step 4: Upload to Pinecone
        print("Uploading to Pinecone...")
        success = upload_to_pinecone(chunks, embeddings, pdf_filename, pinecone_api_key, pinecone_index_name)
        
        if success:
            print(f"Successfully processed {pdf_filename}")
            return True
        else:
            print(f"Failed to upload {pdf_filename} to Pinecone")
            return False
            
    except Exception as e:
        print(f"Error in process_pdf_and_upload: {e}")
        import traceback
        traceback.print_exc()
        return False

def render_pdf_page_to_png_bytes(pdf_url: str, page_number: int = 1, zoom: float = 2.0) -> bytes:
    
    try:
        import fitz
    except Exception as e:
        raise RuntimeError("PyMuPDF (fitz) is required for rendering. Install pymupdf.") from e

    if page_number < 1:
        raise ValueError("page_number must be >= 1")

    # Fetch PDF from URL
    try:
        response = httpx.get(pdf_url, timeout=30.0)
        response.raise_for_status()
        pdf_bytes = response.content
    except httpx.HTTPError as e:
        raise Exception(f"Failed to fetch PDF: {str(e)}") from e

    # Open PDF and render page
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    try:
        if page_number > doc.page_count:
            raise ValueError(f"page_number out of range (1..{doc.page_count})")

        page = doc.load_page(page_number - 1)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        return png_bytes
    finally:
        doc.close()
