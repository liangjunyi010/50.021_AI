from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


class SvdFeatureGenerator(object):
    def __init__(self, n_components=100):
        """
        初始化SVD特征生成器。
        :param n_components: SVD降维后的维度数。
        """
        self.vectorizer = TfidfVectorizer()
        self.svd = TruncatedSVD(n_components=n_components)

    def fit_transform(self, headlines, bodies, enable):
        """
        对文本数据进行TF-IDF转换和SVD降维，然后计算每对标题和正文之间的余弦相似度。
        :param headlines: 标题列表。
        :param bodies: 正文列表。
        :return: 标题和正文的SVD特征的余弦相似度。
        """
        # 合并文本用于TF-IDF
        texts = headlines + bodies
        tfidf_features = self.vectorizer.fit_transform(texts)

        # 应用SVD降维
        svd_features = self.svd.fit_transform(tfidf_features)

        # 分离标题和正文的SVD特征
        svd_headlines = svd_features[:len(headlines)]
        svd_bodies = svd_features[len(headlines):]

        # 计算余弦相似度
        cos_similarities = []
        for i in tqdm(range(len(headlines)), desc="Computing SVD cosine similarities"):
            cos_similarity = cosine_similarity([svd_headlines[i]], [svd_bodies[i]])[0][0]
            cos_similarities.append(cos_similarity)

        return np.array(cos_similarities).reshape(-1, 1)