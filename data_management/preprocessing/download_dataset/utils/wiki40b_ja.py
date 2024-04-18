import argparse
import pandas as pd
import json

def main():
    parser = argparse.ArgumentParser()
    # Specify the input file path for the parquet file
    parser.add_argument('--input_file', type=str, required = True, help='Path to the input parquet file')
    # Specify the output file path for the JSON Lines text file
    parser.add_argument('--output_file', type=str, required = True, help='Path to the output JSON Lines file')
    args = parser.parse_args()

    process_dataset(args.input_file, args.output_file)

def process_dataset(input_file, output_file):
    # Read the dataset from the specified parquet file
    train_dataset = pd.read_parquet(input_file)
    train_text = train_dataset["text"]

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for text in train_text:
            articles = text.split('_START_ARTICLE_')[1:]
            for article in articles:
                sections = article.split('_START_SECTION_')[1:]
                for section in sections:
                    paragraphs = section.split('_START_PARAGRAPH_')[1:]
                    for paragraph in paragraphs:
                        cleaned_paragraph = paragraph.replace('_NEWLINE_', '').strip()
                        # Write each paragraph as a separate JSON object
                        json.dump({"text": cleaned_paragraph}, outfile)
                        outfile.write('\n')  # Write a newline character after each JSON object to conform to JSON Lines format

if __name__ == '__main__':
    main()