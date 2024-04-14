from csv import DictReader
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

from utils.score import LABELS


class DataSet():
    def __init__(self, name="train", path="fnc-1"):
        self.path = path

        print("Reading dataset")
        bodies = name+"_bodies.csv"
        stances = name+"_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))

    def get_synonyms(self,word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    def synonym_replacement(self,sentence, n):
        words = word_tokenize(sentence)
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalpha()]))
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = synonyms[0]
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:  # Only replace up to n words
                break

        sentence = ' '.join(new_words)
        return sentence

    def print_stance_counts(self, stances):
        """Print the counts of each type of stance in the dataset."""
        stance_counts = {stance_type: 0 for stance_type in LABELS}
        for stance in stances:
            stance_counts[stance['Stance']] += 1
        print("Stance counts:")
        for stance, count in stance_counts.items():
            print(f"{stance}: {count}")
        return stance_counts

    def augment_data(self, stances, n_augment=1):
        augmented_stances = stances[:]  # 先拷贝原始列表
        for stance in stances:
            if stance['Stance'] in ['disagree', 'agree']:
                original_headline = stance['Headline']
                for _ in range(n_augment):
                    augmented_headline = self.synonym_replacement(original_headline, n=1)
                    new_stance = stance.copy()
                    new_stance['Headline'] = augmented_headline
                    augmented_stances.append(new_stance)
        return augmented_stances

    def read(self,filename):
        rows = []
        with open(self.path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows
