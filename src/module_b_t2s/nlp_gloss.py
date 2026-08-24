import string

class NLPGlossPipeline:
    def __init__(self):
        # Basic stop word list
        self.stop_words = {"bir", "ve", "ile", "de", "da", "ki", "mi", "mı", "mu", "mü"}
        
        # Mock stemmer/lemmatizer map to convert to root words
        self.lemma_map = {
            "gidiyorum": "GİTMEK",
            "geliyorum": "GELMEK",
            "okula": "OKUL",
            "ben": "BEN",
            "sen": "SEN",
            "seviyorum": "SEVMEK",
            "seni": "SEN"
        }

    def process(self, text):
        """
        Converts Turkish/Azerbaijani text into a capitalized Gloss array.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        words = text.split()
        gloss_array = []
        
        for word in words:
            if word in self.stop_words:
                continue
            
            mapped_word = self.lemma_map.get(word, word.upper())
            gloss_array.append(mapped_word)
            
        return gloss_array