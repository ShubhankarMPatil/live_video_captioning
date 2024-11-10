from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize T5 model and tokenizer for summarization
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def deduplicate_captions(captions):
    """
    Removes repetitive or near-duplicate captions using TF-IDF similarity.
    """
    unique_captions = []
    if len(captions) == 0:
        return unique_captions
    
    # Use TF-IDF to calculate the uniqueness of each caption
    vectorizer = TfidfVectorizer().fit_transform(captions)
    vectors = vectorizer.toarray()
    
    for i, caption in enumerate(captions):
        if all(np.dot(vectors[i], vectors[j]) < 0.85 for j in range(i)):
            unique_captions.append(caption)
    
    return unique_captions

def generate_paragraph(captions):
    """
    Summarizes and rephrases the captions into a coherent paragraph.
    """
    # Deduplicate captions to avoid repetitive descriptions
    unique_captions = deduplicate_captions(captions)
    
    # Join captions into a single input text
    text = " ".join(unique_captions)
    input_text = "summarize: " + text

    # Encode the input text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
