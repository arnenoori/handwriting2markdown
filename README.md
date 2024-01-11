# PDF to Markdown Converter

This script converts PDFs and images into markdown text using OpenAI's GPT-4 model.

## How it works

The script first checks if the input file is a PDF or an image (PNG, JPEG, WEBP, or non-animated GIF). If it's a PDF, it converts the PDF into images. Then, it sends each image to OpenAI's GPT-4 model, which returns the text in the image converted into markdown. The markdown text is then written to a markdown file.

## Usage

1. Install the required Python packages by running `pip install -r requirements.txt`.
2. Set your OpenAI API key in the `config.ini` file under the `OpenAI` section.
3. Set the source directory (the directory containing the PDFs and images you want to convert) in the `config.ini` file under the `Directories` section.
4. Run the script with `python pdf-to-markdown.py`.

## Note

The script supports PNG (.png), JPEG (.jpeg and .jpg), WEBP (.webp), and non-animated GIF (.gif) image formats. If the input file is not a PDF or one of these image formats, the script will print an error message and skip the file.