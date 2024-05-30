import torch
from transformers import BertTokenizer, BertForNextSentencePrediction
from load_data import DataPrecessForSentence
from model import BertModelTest
import pandas as pd
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity
from gensim.parsing.preprocessing import preprocess_string

class TextSimilarityEvaluator:
    def __init__(self, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModelTest()
        checkpoint = torch.load("model/best.pth.tar", map_location='cpu')
        self.model.load_state_dict(checkpoint['model'],strict=False)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_seq_len = 102
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    
    def read_text_data(self,file_path):
        texts = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                text = line.split('\t')[0]  # 假设每行文本以制表符分隔，并且我们只取第一列
                texts.append(text)
        return texts

    def calculate_similarity(self,query, texts):
        # 预处理文本数据
        processed_texts = [preprocess_string(text) for text in texts]
        processed_query = preprocess_string(query)

        # 创建词典
        dictionary = corpora.Dictionary(processed_texts)

        # 使用词典和文本数据创建TF-IDF模型
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        tfidf_model = TfidfModel(corpus)

        # 使用TF-IDF模型将查询文本转换为向量
        query_vector = tfidf_model[dictionary.doc2bow(processed_query)]

        # 计算相似度
        index = MatrixSimilarity(tfidf_model[corpus])
        sims = index[query_vector]

        return sims

    def calculate_similarity_score(self, tokens_seq_1, tokens_seq_2_list):
        with torch.no_grad():
            max_score = -float('inf')
            best_tokens_seq_2 = None
            
            for tokens_seq_2 in tokens_seq_2_list:

                
                tokens_seq_1_ids = self.bert_tokenizer.tokenize(tokens_seq_1)
                tokens_seq_2_ids = self.bert_tokenizer.tokenize(tokens_seq_2)
                data = DataPrecessForSentence.trunate_and_pad(self,tokens_seq_1_ids,tokens_seq_2_ids)
                seq, seq_mask, seq_segment = data
                seq = torch.tensor([seq], dtype=torch.long)  # Specify the data type as Long
                seq_mask = torch.tensor([seq_mask], dtype=torch.long)  # Specify the data type as Long
                seq_segment = torch.tensor([seq_segment], dtype=torch.long)
                logits, probabilities = self.model(seq, seq_mask, seq_segment)
                score = probabilities[:, 1].cpu().numpy()
                print(score)
                
                if score > max_score:
                    max_score = score
                    best_tokens_seq_2 = tokens_seq_2
            
            return max_score, best_tokens_seq_2

# Example usage
if __name__ == '__main__':
    evaluator = TextSimilarityEvaluator()
    tokens_seq_1 = "酒店"
    address = 'data/address.txt'
    texts = evaluator.read_text_data(address)
    sims = evaluator.calculate_similarity(tokens_seq_1, texts)
    top_20_similar_texts = sorted(enumerate(sims), key=lambda item: -item[1])[:20]
    similar_texts = [texts[index] for index, _ in top_20_similar_texts]

    max_score, best_tokens_seq_2 = evaluator.calculate_similarity_score(tokens_seq_1, similar_texts)
    
    print(f"最高得分: {max_score}")
    print(f"对应的tokens_seq_2: {best_tokens_seq_2}")
