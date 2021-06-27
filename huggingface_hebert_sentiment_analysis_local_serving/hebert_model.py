from transformers import pipeline

print('Initializing Pipeline')
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="avichr/heBERT_sentiment_analysis",
    tokenizer="avichr/heBERT_sentiment_analysis",
    return_all_scores = True
)

print('Saving model and tokenizers files')
sentiment_analysis.save_pretrained("./model")
