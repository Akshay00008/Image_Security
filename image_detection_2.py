import os
import streamlit as st
from google.cloud import vision
from pymongo import MongoClient
from PIL import Image, ImageDraw

# --- Google Cloud Vision setup ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\hp\\Desktop\\Security\\Image_Security\\pt-generative-bot-2368a761c8d5.json"
vision_client = vision.ImageAnnotatorClient()

# --- MongoDB setup ---
mongo_uri = 'mongodb://dev:N47309HxFWE2Ehc@34.121.45.29:27017/ptchatbotdb?authSource=admin'
mongo_client = MongoClient(mongo_uri)
db = mongo_client['ptchatbotdb']
results_collection = db['profile analysis']

def draw_face_boxes(image, faces):
    draw = ImageDraw.Draw(image)
    for face in faces:
        vertices = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        draw.line(vertices + [vertices[0]], width=5, fill='red')
    return image

def analyze_image_bytes(image_bytes, profile_id=None):
    # Prepare Vision image
    image = vision.Image(content=image_bytes)
    
    # Face detection
    face_response = vision_client.face_detection(image=image)
    faces = face_response.face_annotations
    
    # Web detection
    web_response = vision_client.web_detection(image=image)
    web_detection = web_response.web_detection

    # Prepare result data structure
    results = {
        "profile_id": profile_id,
        "entities": [],
        "faces_detected": len(faces),
        "partial_matching_images": [],
        "pages_with_matching_images": []
    }
    
    # Web detection result extraction
    if web_detection.partial_matching_images:
        results["partial_matching_images"] = [img.url for img in web_detection.partial_matching_images]
    if web_detection.pages_with_matching_images:
        results["pages_with_matching_images"] = [page.url for page in web_detection.pages_with_matching_images]
    if web_detection.web_entities:
        # Associate all pages URLs with each entity as proxy
        page_urls = results["pages_with_matching_images"]
        results["entities"] = [
            {"description": ent.description, "score": ent.score, "associated_urls": page_urls}
            for ent in web_detection.web_entities if ent.description
        ]

    # Insert results into MongoDB
    results_collection.insert_one(results)

    # Return for display and further usage
    return results, faces

# Streamlit UI
st.title("Face-focused Image Upload and Analysis")

profile_id = st.text_input("Enter Profile ID (optional):")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    pil_image = Image.open(uploaded_file)
    # Perform analysis
    with st.spinner("Analyzing image..."):
        results, faces = analyze_image_bytes(image_bytes, profile_id)
    
    # Draw bounding boxes on faces
    image_with_boxes = draw_face_boxes(pil_image.copy(), faces)
    
    st.image(image_with_boxes, caption=f"Detected {results['faces_detected']} face(s)", use_column_width=True)
    
    # Show URLs of partial matching images
    st.subheader("Partial Matching Images (likely faces)")
    if results["partial_matching_images"]:
        for url in results["partial_matching_images"]:
            st.markdown(f"[{url}]({url})")
    else:
        st.write("No partial matching images found.")
    
    # Show pages with matching images
    st.subheader("Pages with Matching Images")
    if results["pages_with_matching_images"]:
        for url in results["pages_with_matching_images"]:
            st.markdown(f"[{url}]({url})")
    else:
        st.write("No pages with matching images found.")
    
    st.success("Analysis saved to MongoDB.")
else:
    st.info("Please upload an image to analyze.")
