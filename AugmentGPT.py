import openai
import csv
import ast
import time
import sys

OPENAI_API_KEY = "OPEN_API_KEY"
train_path = "/content/drive/MyDrive/582Final-main/data/train.csv"
augmented_train_path = "/content/drive/MyDrive/582Final-main/data/augmented_train_with_GPT.csv"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def generate_text(original_text):
    """Generate text using GPT-3.5 that retains the original meaning and style."""
    # try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate text with the same meaning and similar language style."},
            {"role": "user", "content": f"Given sample: {original_text}"}
        ]
    )
    return response.choices[0].message.content
    # except openai.Error as e:
    #     if e.__class__.__name__ == 'RateLimitError':
    #         print("Rate limit exceeded, waiting...")
    #         time.sleep(60)  # Wait for 60 seconds before retrying
    #         return generate_text(original_text)
    #     else:
    #         print(f"Failed to generate text: {str(e)}")
    #         sys.exit(1)

with open(train_path, mode='r', newline='', encoding='utf-8') as source_file, \
     open(augmented_train_path, mode='w', newline='', encoding='utf-8') as destination_file:
    reader = csv.DictReader(source_file)
    writer = csv.DictWriter(destination_file, fieldnames=reader.fieldnames)
    writer.writeheader()

    for index, row in enumerate(reader):
        if index < 5000:
            for key in ['utterance1', 'utterance2']:
                # try:
                    original_data = ast.literal_eval(row[key])
                    generated_text = generate_text(original_data['text'])
                    original_data['text'] = generated_text
                    row[key] = str(original_data)
                    print(f"Index: {index}, Key: {key}, Generated Text: {generated_text}")
                # except Exception as e:
                #     print(f"Error processing row {index}: {str(e)}")
                #     continue

            writer.writerow(row)
            time.sleep(1)  # Increase sleep time to reduce rate of API calls
