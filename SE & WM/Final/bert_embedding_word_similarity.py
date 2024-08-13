from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import heapq

# BERT model & tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

#  similar words - BERT embeddings
def similar_words_bert(word, tokenizer, model, dataset_words):
    # Tokenizing a given word
    tokens = tokenizer.tokenize(word)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

    # Convert tokens to tensor
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        outputs = model(tokens_tensor)
        word_embedding = torch.mean(outputs[0], dim=1).squeeze()

    top_similar_words = []
    for compare_word in dataset_words:
        # double-check each comparison word is a string
        compare_word = str(compare_word)
        
        # Tokenize comparison word
        compare_tokens = tokenizer.tokenize(compare_word)
        compare_indexed_tokens = tokenizer.convert_tokens_to_ids(compare_tokens)
        compare_tokens_tensor = torch.tensor([compare_indexed_tokens])

        with torch.no_grad():
            compare_outputs = model(compare_tokens_tensor)
            compare_embedding = torch.mean(compare_outputs[0], dim=1).squeeze()

        similarity = cosine_similarity(word_embedding.numpy().reshape(1, -1), compare_embedding.numpy().reshape(1, -1))[0][0]
        
        # sort/track 20 most similar words
        if len(top_similar_words) < 20:
            heapq.heappush(top_similar_words, (similarity, compare_word))
        else:
            heapq.heappushpop(top_similar_words, (similarity, compare_word))
    
    # most similar words in descending order
    top_similar_words = sorted(top_similar_words, reverse=True)
    
    return top_similar_words

# words from 'cleaned_content' column 
words_to_compare = reviews_df['cleaned_content'].astype(str).str.split().explode().unique().tolist()

word = "camera"
similar_words_list = similar_words_bert(word, tokenizer, model, words_to_compare)[:20]

print(f"Top 20 most similar words to '{word}':")
for similarity, similar_word in similar_words_list:
    print(f"'{similar_word}': {similarity}")