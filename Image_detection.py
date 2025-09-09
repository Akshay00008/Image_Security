import os
from google.cloud import vision
from pymongo import MongoClient


# --- Google Cloud Vision setup ---
# Set your Google Application credentials with your service account JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/bramhesh_srivastav/Akshay/computervision/Image_Security/pt-generative-bot-2368a761c8d5.json"


# Initialize Vision client
vision_client = vision.ImageAnnotatorClient()


# --- MongoDB setup ---
mongo_uri = 'mongodb://dev:N47309HxFWE2Ehc@34.121.45.29:27017/ptchatbotdb?authSource=admin'
mongo_client = MongoClient(mongo_uri)
db = mongo_client['ptchatbotdb']  # Use the database name from the connection string
results_collection = db['profile analysis']


def analyze_image(image_path, profile_id):
    # Load the image
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform web detection (reverse image search, entities, matches)
    response = vision_client.web_detection(image=image)
    web_detection = response.web_detection

    # Prepare results
    results = {
        "profile_id": profile_id,
        "image_path": image_path,
        "entities": [],
        "full_matching_images": [],
        "partial_matching_images": [],
        "pages_with_matching_images": []
    }

    # Get detected web entities (tags/concepts associated)
    if web_detection.web_entities:
        results["entities"] = [
            {"description": entity.description, "score": entity.score}
            for entity in web_detection.web_entities
            if entity.description
        ]

    # Get URLs to images that fully match the user's image
    if web_detection.full_matching_images:
        results["full_matching_images"] = [
            image.url for image in web_detection.full_matching_images
        ]

    # Get URLs to images that partially match
    if web_detection.partial_matching_images:
        results["partial_matching_images"] = [
            image.url for image in web_detection.partial_matching_images
        ]

    # Get pages containing matching images
    if web_detection.pages_with_matching_images:
        results["pages_with_matching_images"] = [
            page.url for page in web_detection.pages_with_matching_images
        ]

    # Insert to MongoDB
    results_collection.insert_one(results)

    return results


# --- Example Usage ---
if __name__ == "__main__":
    analysis_result = analyze_image("download (4).jpeg", profile_id="user123")
    print("Analysis saved in MongoDB:", analysis_result)
