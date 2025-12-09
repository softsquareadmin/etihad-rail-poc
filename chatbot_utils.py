import os
from openai import OpenAI, OpenAIError
from pinecone import Pinecone
import dotenv
import json

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

openai_client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)


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

def generate_response(chat_history, context, user_input):
    """
    Generate response using GPT with retrieved context
    """
    system_prompt = """You are a helpful AI assistant that answers questions based on the provided document context.

INSTRUCTIONS:
1. Answer questions accurately using only the information from the provided context
2. If the context doesn't contain enough information to answer the question, say so
3. Be concise but comprehensive in your responses
4. If asked about specific details, cite the source document when relevant
5. If the query is general conversation like "Hello" or "How are you?", respond appropriately without using the context and DO NOT cite any sources, add empty metadata like {"source": "", "page": ""}
6. Maintain a helpful and professional tone

Remember: Only use information from the provided context to answer questions.

IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanations, just the JSON object:
{
  "answer": "your detailed answer here",
  "metadata": {"source": "Source Name", "page": X}
}

Formatting rules for "answer":
- Format the content with HTML tags where appropriate (e.g., <b>, <i>, <ul>, <li>, <p>, etc.)
- Use <br> for line breaks
- Make use of the HTML formatting to enhance readability and structure of the answer

If you use multiple sources, pick the PRIMARY one. Extract the source, page number from the [Source: ..., Page: X, ...] markers in the context."""

    messages = [{"role": "system", "content": system_prompt}]
    messages += chat_history
    
    user_message_content = f"Context from documents:\n\n{context}\n\nUser question: {user_input}"
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
        
        return answer, source
    except OpenAIError as e:
        print(f"OpenAI error: {e}")
        return "Sorry, there was an error generating a response."

def process_user_query(user_query, chat_history=None):
    """
    Main function to process user queries with RAG pipeline
    """
    if chat_history is None:
        chat_history = []
    
    # Step 1: Embed the query
    query_embedding = embed_query(user_query)
    if query_embedding is None:
        return ("Sorry, I couldn't process your query at the moment. Please try again.", [])
    
    # Step 2: Search Pinecone for relevant chunks
    matches = search_pinecone(query_embedding, top_k=5)
    
    if not matches:
        return ("I don't have any information about that in my knowledge base. Please make sure you've uploaded relevant PDF documents.", [])
    
    # Step 3: Build context from matches
    context = build_context_from_matches(matches)
    
    # Step 4: Generate response
    response, source = generate_response(chat_history, context, user_query)

    # Return both the generated response and the raw matches (so callers can show grounding)
    return (response, source)