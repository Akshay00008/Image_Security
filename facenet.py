import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from google.cloud import vision
from deepface import DeepFace
import faiss
from pymongo import MongoClient
from datetime import datetime
import streamlit as st
import concurrent.futures

# Setup Google Vision API credentials path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "pt-generative-bot-2368a761c8d5.json"
vision_client = vision.ImageAnnotatorClient()

# MongoDB Configuration
MONGO_URI = 'mongodb://dev:N47309HxFWE2Ehc@34.121.45.29:27017/ptchatbotdb?authSource=admin'
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['ptchatbotdb']
results_collection = db['face_search_results']

def get_google_vision_candidate_urls(image_path, max_images=30):
    with open(image_path, "rb") as img_file:
        content = img_file.read()

    image = vision.Image(content=content)
    response = vision_client.web_detection(image=image)
    web_detection = response.web_detection

    urls = set()
    if web_detection.pages_with_matching_images:
        for page in web_detection.pages_with_matching_images:
            urls.add(page.url)

    if web_detection.partial_matching_images:
        for img in web_detection.partial_matching_images:
            urls.add(img.url)

    return list(urls)[:max_images]

def download_image(url):
    try:
        response = requests.get(url, timeout=5, allow_redirects=True)
        if "image" not in response.headers.get("Content-Type", ""):
            return None

        img = Image.open(BytesIO(response.content))
        img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"Failed to download or open image {url}: {e}")
        return None

def extract_face_embedding_from_pil(img):
    try:
        temp_path = "temp.jpg"
        img.save(temp_path)
        embeddings = DeepFace.represent(img_path=temp_path, model_name='Facenet', enforce_detection=True)
        if embeddings and len(embeddings) > 0:
            return np.array(embeddings[0]['embedding'], dtype='float32')
        else:
            return None
    except Exception as e:
        print(f"Error extracting embedding: {e}")
    return None

def build_faiss_index(embeddings):
    if not embeddings:
        return None
    
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatIP(dim)
    embeddings_np = np.array(embeddings)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)
    return index

def search_similar_faces(index, query_embedding, top_k=5):
    query_vec = np.array([query_embedding])
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, top_k)
    return distances[0], indices[0]

def save_results_to_mongo(profile_id, input_image_path, matches):
    document = {
        "profile_id": profile_id,
        "input_image_path": input_image_path,
        "search_time": datetime.utcnow(),
        "matches": matches
    }
    results_collection.insert_one(document)

def face_search_pipeline(input_image_path, profile_id="default_user"):
    print("Getting candidate image URLs from Google Vision API...")
    candidate_urls = get_google_vision_candidate_urls(input_image_path)

    candidate_embeddings = []
    valid_urls = []

    # Using ThreadPoolExecutor for concurrent image downloading and processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(download_image, url): url for url in candidate_urls}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            img = future.result()
            if img:
                embedding = extract_face_embedding_from_pil(img)
                if embedding is not None:
                    candidate_embeddings.append(embedding)
                    valid_urls.append(url)

    if not candidate_embeddings:
        print("No candidate face embeddings extracted.")
        return []

    print("Building FAISS index...")
    index = build_faiss_index(candidate_embeddings)

    if index is None:
        print("FAISS index creation failed.")
        return []

    print("Extracting embedding for input image...")
    input_embedding = extract_face_embedding_from_pil(Image.open(input_image_path).convert("RGB"))
    if input_embedding is None:
        print("Failed to extract embedding from input image.")
        return []

    print("Searching for similar faces...")
    distances, indices = search_similar_faces(index, input_embedding, top_k=5)

    results = []
    for idx in indices:
        results.append({
            "matched_image_url": valid_urls[idx]
        })

    save_results_to_mongo(profile_id, input_image_path, results)

    return results

# Streamlit app setup
def main():
    st.title("Face Search Using DeepFace")
    
    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded image to a temporary file
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Perform face search
        st.write("Processing the image...")
        profile_id = "user123"  # You can use a dynamic user ID
        matches = face_search_pipeline("uploaded_image.jpg", profile_id=profile_id)

        if matches:
            st.write("Top matched similar faces:")
            for match in matches:
                st.write(f"Matched Image URL: {match['matched_image_url']}")
        else:
            st.write("No similar faces found.")

if __name__ == "__main__":
    main()
