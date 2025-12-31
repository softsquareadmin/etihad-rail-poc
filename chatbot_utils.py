import os
from openai import OpenAI, OpenAIError
from google import genai
from google.genai import types
from pinecone import Pinecone
import dotenv
import json
import cohere
import io
import wave
import streamlit as st

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
cohere_api_key = os.getenv("COHERE_API_KEY")
gemini_transcription_model = os.getenv("GEMINI_TRANSCRIPTION_MODEL")
openai_transcription_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL")
gemini_audio_generation_model = os.getenv("GEMINI_AUDIO_GENERATION_MODEL")
openai_audio_generation_model = os.getenv("OPENAI_AUDIO_GENERATION_MODEL")
gemini_api_key = os.getenv("GEMINI_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)
gemini_client = genai.Client(api_key=gemini_api_key)

def transcribe_audio(audio_file):
    """
    Transcribes an audio file using OpenAI's Whisper-1 model.
    Works with file-like objects (e.g., from Streamlit's st.audio_input) 
    or paths to local files.
    """
    try:
        if not audio_file:
            return ""

        if st.session_state.change_transcription_model == False:    
            # Call the Whisper API
            # The 'audio_file' here is a file-like object provided by Streamlit
            transcript = openai_client.audio.transcriptions.create(
                model=openai_transcription_model, 
                file=audio_file
            )
            print("Transcript :::::", transcript.text)
            return transcript.text
        else:
            transcript = gemini_client.models.generate_content(
                model=gemini_transcription_model,
                contents=[
                    types.Part.from_bytes(
                        data=audio_file.getvalue(),
                        mime_type=audio_file.type
                    ),
                    """ Translate this audio accurately in ENGLISH.
                        NO Foreign Language is accepted, Translate everything into ENGLISH
                        
                        Respond ONLY in valid JSON with this exact format:
                        {
                        "lang": "Detected Language of the audio",
                        "translation": "<translated text>",
                        "transcript": "Actual transcript of the audio in its original language"
                        }

                        Rules:
                        - Output must be valid JSON
                        - No markdown
                        - No explanations
                        - No extra keys
                    """,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            data = json.loads(transcript.text)
            print("Data :::::", data)
            print("Audio Translation :::::", data.get("translation"))
            print("Audio Trancription :::::", data.get("transcript"))
            print("Detected Language :::::", data.get("lang"))
            return data   
    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"Error: {str(e)}"

def generate_audio_response(text):
    if st.session_state.change_transcription_model == False:
        audio_bytes = io.BytesIO()
        with openai_client.audio.speech.with_streaming_response.create(
            model=openai_audio_generation_model,
            voice="shimmer",
            input=text,
            instructions="Speak in a cheerful and positive tone.",
        ) as response:
            for chunk in response.iter_bytes():
                audio_bytes.write(chunk)
        return audio_bytes
    else:
        response = gemini_client.models.generate_content(
                model=gemini_audio_generation_model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name='Kore',
                            )
                        )
                    ),
                )
            )

        # Extract the raw PCM data from the response
        audio_data = response.candidates[0].content.parts[0].inline_data.data

        # Wrap the raw PCM data into a WAV container in memory
        audio_bytes = io.BytesIO()
        with wave.open(audio_bytes, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_data)

        # Seek to start so the returned object is ready to be read/played
        audio_bytes.seek(0)
        return audio_bytes

def embed_query(text, model="text-embedding-3-small"):
    """
    Create embedding for user query
    """
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except OpenAIError as e:
        print(f"OpenAI error: {e}")
        return None

def search_pinecone(query_embedding, top_k=5):
    """
    Search Pinecone for similar chunks
    """
    try:
        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        print("response ::::: Pinecone",response)
        return response.matches
    except Exception as e:
        print(f"Pinecone query error: {e}")
        return []

def build_context_from_matches(matches):
    """
    Build context string from Pinecone matches
    """
    if not matches:
        return "No relevant information found."
    
    context_parts = []
    
    for i, match in enumerate(matches, 1):
        text = match.metadata.get("text", "")
        source = match.metadata.get("source", "Unknown")
        page = match.metadata.get("page_number", "Unknown")
        score = match.score
        
        if text:
            context_parts.append(f"[Source: {source}, Page: {page}, Relevance: {score:.2f}]\n{text}\n")
    return "\n---\n".join(context_parts)

def rerank_matches(user_query, matches, top_k=5):
    """
    Rerank Pinecone matches using Cohere
    """
    if not matches:
        return []
    
    co = cohere.ClientV2(api_key=cohere_api_key)
    
    docs = [match.metadata.get("text", "") for match in matches]
    
    try:
        rerank_response = co.rerank(
            model="rerank-v3.5",
            query=user_query,
            documents=docs,
            top_n=top_k
        )
        
        reranked_indices = [item.index for item in rerank_response.results]
        reranked_matches = [matches[i] for i in reranked_indices]
        
        return reranked_matches
    except Exception as e:
        print(f"Cohere rerank error: {e}")
        return matches[:top_k]

def generate_response(chat_history, context, user_input, language, query_context=None):
    """
    Generate response using GPT with retrieved context
    """
    system_prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context.

INSTRUCTIONS:
1. Answer questions accurately using ONLY the information from the provided 'Context from documents'.
2. PRIORITIZE 'Context from documents' over 'Conversation History' for all factual information.
3. 'Conversation History' should ONLY be used to understand the user's intent (e.g., resolving pronouns like "it" or "that unit") and context of the conversation.
4. If the 'Context from documents' does not contain the answer, state that you don't have that information, EVEN IF the answer was mentioned in a previous turn of the 'Conversation History'. Do NOT leak information from history into the current answer if it's not in the current context.
5. Be concise but comprehensive in your responses.
6. Cite the source document and page number ONLY if the information is present in the current 'Context from documents'.
7. If the query is general conversation like "Hello" or "How are you?", respond appropriately without using the context and DO NOT cite any sources, add empty metadata like {{"source": "", "page": ""}}.
8. If the query is irrelevant to the context or outside the scope of the documents, politely inform the user that you can only answer questions related to the provided context and DO NOT cite any sources, add empty metadata like {{"source": "", "page": ""}}.
9. If the content in 'Context from Document' is irrelevant or insufficient to answer the User question, respond accordingly and DO NOT cite any sources, add empty metadata like {{"source": "", "page": ""}}.
10. If the query is incomplete or unclear, ask for clarification without using the context and DO NOT cite any sources, add empty metadata like {{"source": "", "page": ""}}.
11. Maintain a helpful and professional tone.

Remember: Only use information from the provided context to answer questions.

RESPONSE RULES (MANDATORY):
1. Respond ONLY in Language: {language}, NOTE: If Language is None then respond in the language used in the User Query.
2. Do NOT translate unless the user explicitly asks.
3. If retrieved documents are in another language, still respond in the Prefered Language above.
4. Preserve technical terms when appropriate.

IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanations, just the JSON object:
{{
  "answer": "your detailed answer here",
  "metadata": {{"source": "Source Name", "page": X}}
}}

Formatting rules for "answer":
- Format the content with HTML tags where appropriate (e.g., <b>, <i>, <ul>, <li>, <p>, etc.)
- Use <br> for line breaks
- Make use of the HTML formatting to enhance readability and structure of the answer

If you use multiple sources, pick the chunk from which more information is used to form the response. Extract the source, page number from the [Source: ..., Page: X, ...] markers in the context."""

    # Filter chat history to only include 'role' and 'content' for OpenAI API
    # Force 'content' to be a string to avoid JSON serialization errors
    filtered_history = []
    for msg in chat_history:
        content = msg.get("content", "")
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        filtered_history.append({
            "role": msg["role"],
            "content": str(content)
        })
    
    messages = [{"role": "system", "content": system_prompt}]
    messages += filtered_history
    
    query_context_str = f"Query Context (Equipment Details): {query_context}\n\n" if query_context else ""
    print("Query Context received :::::", query_context_str)
    user_message_content = f"{query_context_str}Context from documents:\n\n{context}\n\nUser question: {user_input}"
    messages.append({"role": "user", "content": user_message_content})
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content.strip()
        parsed = json.loads(result)
        answer = parsed.get("answer", "")
        source = parsed.get("metadata", {})
        print("Open AI Source :::::", source)
        return answer, source
    except OpenAIError as e:
        print(f"OpenAI error: {e}")
        return "Sorry, there was an error generating a response."

def check_query(user_input, lang):
    """
    Generate response using GPT with retrieved context
    """
    system_prompt = f"""You are a helpful AI assistant whose ONLY task is to determine whether a user query is a
general conversational message or not.

GENERAL CONVERSATION includes (but is not limited to):
- Greetings and farewells (e.g., "hi", "hello", "hey", "good morning", "bye")
- Polite expressions (e.g., "thank you", "thanks", "sorry", "appreciate it")
- Small talk (e.g., "how are you?", "what's up?", "how's your day?")
- Casual acknowledgements (e.g., "ok", "okay", "cool", "got it", "fine")
- Emotional expressions (e.g., "I'm tired", "I'm happy", "feeling bored")
- Meta questions about you (e.g., "who are you?", "what can you do?", "are you real?")
- Social or conversational fillers that do NOT require external knowledge or documents

INSTRUCTIONS:
1. If the user query is general conversation, respond with:
   {{
     "is_greeting": true,
     "response": "<polite, friendly conversational reply>"
   }}

2. If the user query is NOT general conversation, respond with:
   {{
     "is_greeting": false,
     "response": ""
   }}

STRICT RULES:
- Respond ONLY with valid JSON
- Generate response ONLY in Language: {lang}, NOTE: If Language is None then respond in the language used in User Query.
- Do NOT add explanations, comments, or extra text
- The "response" field must contain a friendly, short reply ONLY when is_greeting = true
- Use simple HTML formatting (<p>, <b>, <i>, <br>) inside "response" if helpful
- If there is ANY doubt, treat the query as NOT general conversation
"""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Ensure user_input is a string
    clean_input = str(user_input)
    if isinstance(user_input, bytes):
        clean_input = user_input.decode('utf-8', errors='ignore')
        
    messages.append({"role": "user", "content": clean_input})
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content.strip()
        parsed = json.loads(result)
        answer = parsed.get("response", "")
        is_greeting = parsed.get("is_greeting", False)
        
        return answer, is_greeting
    except OpenAIError as e:
        print(f"OpenAI error: {e}")
        return "Sorry, there was an error generating a response."

def process_user_query(user_query, chat_history=None, rerank=False, category=None, type=None, brand=None, model_series=None, is_side = False):
    """
    Main function to process user queries with RAG pipeline
    """
    if isinstance(user_query, dict):
        translation = user_query.get("translation", "")
        lang = user_query.get("lang", "None")
        query = user_query.get("transcript", "")
    else:
        query = user_query
        translation = user_query
        lang = "None"
    
    if chat_history is None:
        chat_history = []
    
    if query.strip() == "":
        return ("Looks like there’s nothing to process — please enter a valid message", [])

    # Checking whether query is a simple greeting or irrelevant 
    response, is_greeting = check_query(query, lang)
    source = {"source": "", "page": ""}

    if not is_greeting:
        print("Processing RAG for query ::::::")
        
        # Step 1: Embed the query
        query_embedding = embed_query(translation)
        if query_embedding is None:
            return ("Sorry, I couldn't process your query at the moment. Please try again.", [])
        
        # Step 2: Search Pinecone for relevant chunks
        matches = search_pinecone(query_embedding, top_k=5 if not rerank else 15)
        
        if not matches:
            return ("I don't have any information about that in my knowledge base. Please make sure you've uploaded relevant PDF documents.", [])
        
        # Optional Step: Rerank matches using Cohere
        if rerank:
            matches = rerank_matches(translation, matches, top_k=5)
            print("Reranked matches ::::::", matches)
        
        # Step 3: Build context from matches
        context = build_context_from_matches(matches)
        
        # Step 4: Generate response
        # Pass filters as query context to the LLM
        query_context = f"Category: {category}, Type: {type}, Brand: {brand}, Model Series: {model_series}"
        response, source = generate_response(chat_history, context, query, query_context=query_context, language = lang)

    # Return both the generated response and the raw matches (so callers can show grounding)
    return (response, source)