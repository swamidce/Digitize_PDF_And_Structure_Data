## Digitize a pdf document and structure the data

### Approach 1 : When pages in the Pdf are text-based

# Install necessary packages
!pip install pdfminer.six
!pip install pdfminer.six spacy
!pip install pdfminer.six nltk
!pip install pdfminer.six nltk regex
!pip install spacy download en_core_web_sm

# Import necessary libraries
import re
import json
import nltk
from pdfminer.high_level import extract_text
from nltk import sent_tokenize, word_tokenize, pos_tag

# Download the punkt and averaged_perceptron_tagger resources
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# This function takes a text as input, tokenizes it into sentences and words, performs part-of-speech tagging
# and extracts entities (noun phrases) composed of consecutive nouns (NN).
# The entities are then returned as a list.
def extract_entities(text):
    entities = []
    sentences = sent_tokenize(text)

    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)

        for i in range(len(tagged_words)):
            if tagged_words[i][1] == 'NN' and i + 1 < len(tagged_words) and tagged_words[i + 1][1] == 'NN':
                entities.append(tagged_words[i][0] + " " + tagged_words[i + 1][0])

    return entities

# This function takes PDF text, a list of entities and a dictionary of patterns as input.
# It iterates through each line of the PDF text, searching for entities and patterns.
# When a match is found, it creates a dictionary containing the entity, key and matched value and appends it to the 'data' list.
# The final list of dictionaries is returned.
def extract_bill_data(pdf_text, entities, patterns):
    data = []
    current_entity = None

    for line in pdf_text.split('\n'):
        for entity in entities:
            if entity.lower() in line.lower():
                current_entity = entity
                break

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                data.append({
                    "entity": current_entity,
                    "key": key,
                    "value": match.group()
                })

    return data

def main():
    pdf_path = "Exercise2_data.pdf"
    pdf_text = extract_text(pdf_path)

    # Entities and patterns for PDF File
    entities = ["CARDIOLOGY", "DOCTOR FEES", "LABORATORY", "SEROLOGY", "RADIOLOGY", "MEDICAL APPLIANCES",
                  "NUTRITION OPTIMIZATION AND DIETARY SERVICES", "PATIENT DIET", "SEDATION & MAC", "ANAESTHESIA CHARGES", "DENTAL", "CHEMOPORT INSERTION",
                  "ENDO ROOT CANAL TREATMENT", "BLOWOUT FRACTURE ORBITAL FLOOR", "LAPROSCOPIC SURGICAL PACKAGES"]
    patterns = {
        "description": r"^[A-Z\s]+[A-Z]$",  # Uppercase words
        "quantity_amount": r"(\d+)\s*\.\s*(\d{2,3}(,\d{3})*(.\d{2})?)",  # Quantity and Amount
        "tariff": r"\d+",  # Tariff amount
    }

    bill_data = extract_bill_data(pdf_text, entities, patterns)

    # Print the structured data in JSON format for PDF File
    print("Data for PDF File:")
    print(json.dumps(bill_data, indent=2))

# It helps in preventing the execution of the main function if the script is imported elsewhere
if __name__ == "__main__":
    main()


### Approach 2 : When pages in the Pdf are image-based (Used OCR)

# Install necessary packages
!apt install tesseract-ocr
!apt-get install -y poppler-utils
!pip install pytesseract pdfplumber pdf2image
!pip install numpy

# Import necessary libraries
import pytesseract
from pdf2image import convert_from_path
import regex
import spacy
import json

def extract_ocr_data(pdf_path):
    # Load spaCy NLP model for English
    nlp = spacy.load("en_core_web_sm")

    # Convert PDF to images using pdf2image library
    images = convert_from_path(pdf_path, dpi=300)

    # Initialize an empty list to store extracted data
    data = []
    current_entity = None

    # Iterate through each image extracted from the PDF
    for i, image in enumerate(images):
        # Apply OCR (Optical Character Recognition) to convert image to text
        page_text = pytesseract.image_to_string(image, config='--psm 6')

        # Split text into lines and remove empty lines
        lines = [line.strip() for line in page_text.split('\n') if line.strip()]

        # Iterate through each line of text
        for line in lines:
            # Use spaCy NER (Named Entity Recognition) to identify entities
            doc = nlp(line)
            for ent in doc.ents:
                current_entity = ent.text.strip()

            # Use regex to identify relevant patterns (assuming they are uppercase letters, digits, spaces, underscores, and hyphens)
            match = regex.match(r'^([A-Z\s\d_-]+)\s*$', line)
            if match:
                current_entity = match.group(1).strip()

            # Extract numerical values using regex
            if current_entity and regex.search(r'\d+', line):
                # Append extracted data to the list
                data.append({
                    "entity": current_entity,
                    "text": line.strip()
                })

    # Return the extracted data
    return data

def main():
    # Define the path to the PDF file
    pdf_path = "Exercise2_data.pdf"

    # Call the extract_ocr_data function to extract data from the PDF
    ocr_data = extract_ocr_data(pdf_path)

    # Print the structured data in JSON format for the PDF file
    print("Data for PDF File:")
    print(json.dumps(ocr_data, indent=2))

# It helps in preventing the execution of the main function if the script is imported elsewhere
if __name__ == "__main__":
    main()
