import torch
import math
from transformers import BertTokenizer
from load_data import DataPrecessForSentence
from model import BertModelTest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba

class TextSimilarityEvaluator:
    def __init__(self, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModelTest()
        checkpoint = torch.load("/kaggle/input/model-similar/best.pth.tar", map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_seq_len = 102
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.R = 6371  # 地球半径，单位为公里

    def read_text_data(self, file_path):
        texts = []
        lon_lat = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split('\t')
                text = parts[0]
                longitude, latitude = float(parts[1]), float(parts[2])
                texts.append(text)
                lon_lat.append((longitude, latitude))
        return texts, lon_lat

    def read_text(self,file_path):
        texts = []
        Lon = []
        lat = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                text = line.split('\t')[0]  # 假设每行文本以制表符分隔，并且我们只取第一列
                Longitude = line.split('\t')[1]
                Latitude = line.split('\t')[2]
                texts.append(text)
                Lon.append(Longitude)
                lat.append(Latitude)

        return texts, Lon, lat

    def calculate_similarity(self, query, texts):
        all_texts = [" ".join(jieba.lcut(text)) for text in [query] + texts]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        query_similarity = similarity_matrix.flatten()
        return query_similarity

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = math.sin(d_lat/2) * math.sin(d_lat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon/2) * math.sin(d_lon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = self.R * c
        return distance

    def calculate_score(self, lat, lon, points):
        scores = []
        total_distance = 0
        for point in points:
            distance = self.calculate_distance(lat, lon, float(point[0]), float(point[1]))
            total_distance += 1 / (1 + distance)  # 根据距离计算得分，并累加总距离
        for point in points:
            distance = self.calculate_distance(lat, lon, float(point[0]), float(point[1]))
            score = (1 / (1 + distance)) / total_distance  # 计算得分，确保得分的和为1
            scores.append(score)
        return scores

    def calculate_similarity_score(self, tokens_seq_1, tokens_seq_2_list, score_list):
        with torch.no_grad():
            max_score = -float('inf')
            best_tokens_seq_2 = None

            for tokens_seq_2, similar_score in zip(tokens_seq_2_list, score_list):
                tokens_seq_1_ids = self.bert_tokenizer.tokenize(tokens_seq_1)
                tokens_seq_2_ids = self.bert_tokenizer.tokenize(tokens_seq_2)
                data = DataPrecessForSentence.trunate_and_pad(self, tokens_seq_1_ids, tokens_seq_2_ids)
                seq, seq_mask, seq_segment = data
                seq = torch.tensor([seq], dtype=torch.long).to(self.device)
                seq_mask = torch.tensor([seq_mask], dtype=torch.long).to(self.device)
                seq_segment = torch.tensor([seq_segment], dtype=torch.long).to(self.device)
                logits, probabilities = self.model(seq, seq_mask, seq_segment)
                score = probabilities[:, 1].cpu().numpy()
                score = score + similar_score * 60

                if score > max_score:
                    max_score = score
                    best_tokens_seq_2 = tokens_seq_2

            return max_score, best_tokens_seq_2

from tqdm import tqdm  # Import tqdm library

if __name__ == '__main__':
    evaluator = TextSimilarityEvaluator()
    text_file = '/kaggle/input/test0531/test.txt'
    tokens_seq_1_list, input_lon_list, input_lat_list = evaluator.read_text(text_file)
    best_tokens_seq_2_list = []
    for tokens_seq_1, input_lon, input_lat in tqdm(zip(tokens_seq_1_list, input_lon_list, input_lat_list), total=len(tokens_seq_1_list), desc="Processing"):  # Wrap the loop with tqdm
        address = '/kaggle/input/address/address.txt'
        texts, lon_lat = evaluator.read_text_data(address)
        sims = evaluator.calculate_similarity(tokens_seq_1, texts)

        top_60_similar_texts = sorted(enumerate(sims), key=lambda item: -item[1])[:60]
        similar_texts = [texts[index] for index, _ in top_60_similar_texts]
        similar_lon_lat = [lon_lat[index] for index, _ in top_60_similar_texts]
        score_list = evaluator.calculate_score(float(input_lon), float(input_lat), similar_lon_lat)

        max_score, best_tokens_seq_2 = evaluator.calculate_similarity_score(tokens_seq_1, similar_texts, score_list)
        best_tokens_seq_2_list.append(best_tokens_seq_2) 

    output_file = '/kaggle/working/output.txt'
    with open(output_file, 'w') as file:
        for item in best_tokens_seq_2_list:
            file.write("%s\n" % item)
