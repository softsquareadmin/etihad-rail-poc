import os
import json
import time
from openai import OpenAI, OpenAIError
from pinecone import Pinecone
import google.generativeai as genai

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
        
        # Use Gemini to extract text with comprehensive formatting
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        prompt = """Extract ALL text from this document with clear formatting and logical structure.

📋 FORMATTING GUIDELINES:

1. PAGE MARKERS:
   Start each page with a clear separator:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   📄 PAGE [NUMBER]
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks for better context preservation
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move to next chunk with overlap, ensuring forward progress
        if end >= text_length:
            break
        start = end - overlap
    
    return chunks

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
                "text": chunk,
                "source": pdf_filename,
                "chunk_index": i,
                "total_chunks": len(chunks)
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

def process_pdf_and_upload(pdf_path, gemini_api_key, openai_api_key, pinecone_api_key, pinecone_index_name):
    """
    Main pipeline: Extract text from PDF, chunk it, embed, and upload to Pinecone
    """
    try:
        # Get filename for metadata
        pdf_filename = os.path.basename(pdf_path)
        print(f"Processing {pdf_filename}...")
        
        # Step 1: Extract text using Gemini
        print("Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_path, gemini_api_key)
        
        if not text:
            print("Failed to extract text from PDF")
            return False
        
        print(f"Extracted {len(text)} characters")
        
        # Step 2: Chunk the text
        print("Chunking text...")
        try:
            chunks = chunk_text(text, chunk_size=1000, overlap=200)
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
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                print(f"  Embedding chunk {i+1}/{len(chunks)}...")
            
            embedding = embed_text(chunk, openai_api_key)
            if embedding:
                embeddings.append(embedding)
            else:
                print(f"Failed to create embedding for chunk {i}")
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
