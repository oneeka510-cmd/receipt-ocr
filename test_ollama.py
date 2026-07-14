import ollama

# Update this path to point at an actual image file on your computer
image_path = r"D:\receipt_ocr\sample_receipts\receipt.jpg"

response = ollama.chat(
    model="gemma3:4b",
    messages=[{
        "role": "user",
        "content": "What do you see in this image?",
        "images": [image_path]
    }]
)
print(response["message"]["content"])
