import os
import glob
import configparser
import logging
import openai as OpenAI
import requests
import base64
import aiohttp
import random
import asyncio
import shutil
from aiohttp import client_exceptions
from pdf2image import convert_from_path
from PIL import Image
from multiprocessing import Pool

# Set up logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read configuration
config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('OpenAI', 'API_KEY')
SOURCE_DIRECTORY = config.get('Directories', 'SOURCE_DIRECTORY')

IMAGE_LABEL_PROMPT = """
            I have given you an image. Your goal is to simply convert the text of the image to markdown:
            * Transcribe the recognized text into Markdown syntax.
            * Preserve basic formatting elements such as headers, lists, bold, and italics.
            * Convert image captions and annotations into appropriate Markdown or alt text formats.
            * If there's a chart or table try your best to preserve that in markdown.
            OUTPUT NOTHING ELSE BESIDES THE TEXT FROM THE IMAGE
                """

def save_image(image, i, pdf_path):
    png_dir = os.path.join(os.path.dirname(pdf_path), 'pngs')
    os.makedirs(png_dir, exist_ok=True)
    image_path = os.path.join(png_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_{i}.png")
    image.save(image_path, 'PNG')
    return image_path

def convert_pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    with Pool() as p:
        image_files = p.starmap(save_image, [(image, i, pdf_path) for i, image in enumerate(images)])
    return image_files

def encode_image(image):
    image_bytes = image.save(fp=None, format='JPEG')
    return base64.b64encode(image_bytes).decode('utf-8')

def find_images(directory):
    extensions = ['png', 'jpg', 'jpeg', 'gif', 'webp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(f"{directory}/**/*.{ext}", recursive=True))
    return files

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None

async def label_image_async(session, image_path, openai_api_key, max_retries=5, initial_delay=1.0):

    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": IMAGE_LABEL_PROMPT,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low",
                }
            }
            ]
        }
        ],
        "max_tokens": 400,
    }

    # Use GPT-4 Turbo to label the image

    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                elif response.status == 429:
                    logger.warning("Rate limit error: backing off and retrying")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay + random.uniform(0, 1))
                        delay *= 2
                    else:
                        return f"Error: {response.status}"
        except client_exceptions.ServerDisconnectedError:
            logger.warning("Server disconnected: backing off and retrying")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay + random.uniform(0, 1))
                delay *= 2
            else:
                return f"Error: Server disconnected"

async def get_labels(image_files):
    labels = []
    async with aiohttp.ClientSession() as session:
        tasks = [label_image_async(session, image_file, openai_api_key) for image_file in image_files]
        labels = await asyncio.gather(*tasks)
    return labels

def validate_directory(directory):
    """Validate if the provided directory path exists and is a directory."""
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return False
    if not os.path.isdir(directory):
        logger.error(f"Provided path is not a directory: {directory}")
        return False
    return True

def sanitize_label(label):
    invalid_chars = ['/', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        label = label.replace(char, '_')
    return label

def label_and_move_images(src_path, ask_to_proceed=True, debug_output=False, keep_originals=False):
    logger.info(f"Searching for images in {src_path}")
    print(f"Searching for images in {src_path}")

    image_files = find_images(src_path)
    if debug_output:
        for file in image_files:
            print(f"Found file: {file}")

    num_images = len(image_files)
    num_input_tokens = num_images * 350  # Example calculation
    num_output_tokens = num_images * 10
    openai_price = num_input_tokens * 0.00001 + num_output_tokens * 0.00003
    print(f"Calculated cost: ${openai_price}")

    if ask_to_proceed:
        proceed = input(f"Found {num_images} images. Proceed with classification? (y/n) ")
        if proceed.lower() != 'y':
            return

    print("Labelling image files. This may take a while.")
    labels = asyncio.run(get_labels(image_files))

    print(f"Retrieved labels from OpenAI. Moving to sorted folder within {src_path}")
    dst_path = os.path.join(src_path, "sorted")
    os.makedirs(dst_path, exist_ok=True)

    for image, label in zip(image_files, labels):
        folder_name = label.split("_")[0]
        folder_path = os.path.join(dst_path, folder_name + "s")
        os.makedirs(folder_path, exist_ok=True)
        sanitized_label = sanitize_label(label[len(folder_name)+1:])
        shutil.copy(image, os.path.join(folder_path, sanitized_label) + "." + image.split(".")[-1])
        if not keep_originals:
            os.remove(image)  # Delete the original image
            
HOME = os.path.expanduser('~') # Home path

async def get_label(image_file):
    async with aiohttp.ClientSession() as session:
        return await label_image_async(session, image_file, openai_api_key)

def write_to_markdown(label, markdown_file):
    md_dir = os.path.join(os.path.dirname(markdown_file), 'markdown')
    os.makedirs(md_dir, exist_ok=True)
    markdown_path = os.path.join(md_dir, os.path.basename(markdown_file))
    with open(markdown_path, 'a') as f:
        f.write(f"{label}\n\n")

def process_file(file_path):
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    # List of supported image extensions
    supported_image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp']

    # Check if the file is a PDF or an image
    if file_extension == '.pdf':
        # Convert PDF to images
        image_files = convert_pdf_to_images(file_path)
    elif file_extension in supported_image_extensions:
        # If it's an image, no need for conversion
        image_files = [file_path]
    else:
        print(f"Unsupported file type: {file_extension}")
        return

    # Convert images to markdown
    markdown_file = f"{os.path.splitext(file_path)[0]}.md"
    for image_file in image_files:
        label = asyncio.run(get_label(image_file))
        write_to_markdown(label, markdown_file)

if __name__ == "__main__":
    if validate_directory(SOURCE_DIRECTORY):
        # Find all PDF and image files
        files = []
        for ext in ['pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp']:
            files.extend(glob.glob(f"{SOURCE_DIRECTORY}/**/*.{ext}", recursive=True))
        for file in files:
            process_file(file)
    else:
        print("Invalid source directory. Please check the path and try again.")