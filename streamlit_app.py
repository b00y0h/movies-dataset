import streamlit as st
from PIL import Image
import boto3
from io import BytesIO
from openai import OpenAI

# Initialize Boto3 client for Amazon Rekognition
rekognition_client = boto3.client(
    'rekognition',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_DEFAULT_REGION"]
)

def analyze_image_with_rekognition(image):
    """Uses Amazon Rekognition to analyze an image and generate alt text."""
    # Convert image to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    
    # Detect labels with Amazon Rekognition
    response = rekognition_client.detect_labels(
        Image={'Bytes': image_bytes},
        MaxLabels=10,
        MinConfidence=75
    )
    
    # Extract labels
    labels = [label['Name'] for label in response['Labels']]
    return labels

def generate_descriptive_text(labels, openai_api_key):
    """Generates descriptive text using OpenAI's GPT model."""
    if not labels:
        return "No descriptive labels found."
    
    # Initialize OpenAI client with the API key
    client = OpenAI(api_key=openai_api_key)
    
    # Construct a prompt for the GPT model
    prompt = f"Create a descriptive sentence for an image containing the following elements: {', '.join(labels)}."
    
    # Call OpenAI's API to generate text using the new client interface
    response = client.chat.completions.create(
        model="gpt-4o",  # Use a suitable model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates descriptive text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=60
    )
    
    return response.choices[0].message.content.strip()

st.title("Enhanced Alt Text Generator with NLP")

# Check if OpenAI API key is available in secrets
default_api_key = st.secrets.get("OPENAI_API_KEY", "")

# Input for OpenAI API key if not provided in secrets
if not default_api_key:
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
else:
    openai_api_key = default_api_key

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and openai_api_key:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    try:
        # Get labels from Rekognition
        labels = analyze_image_with_rekognition(image)
        
        # Generate descriptive text using NLP
        alt_text = generate_descriptive_text(labels, openai_api_key)
        
        st.write("Generated Alt Text:")
        st.write(alt_text)
    except Exception as e:
        st.error(f"Error generating alt text: {str(e)}")
else:
    st.warning("Please upload an image and ensure an OpenAI API key is provided.")
