import re

class TextToGlossConverter:
    """Yazılı Türkçeyi/Azerbaycancayı İşaret Dili gramerine (Gloss) çevirir."""
    
    def __init__(self):
        # Basit NLP kuralları (Örn: Soru eklerini sona at, zaman kiplerini ayır)
        self.stop_words = ["bir", "şu", "bu", "ve", "ile"]
        
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return [w for w in words if w not in self.stop_words]
        
    def translate_to_gloss(self, text):
        """
        Gelecekte LLM eklenecek olan kural tabanlı prototip katmanı.
        Örn: 'Ben okula gidiyorum' -> 'BEN OKUL GİTMEK'
        """
        words = self.preprocess_text(text)
        # TODO: NLP Modeli (Stemming & Lemmatization) eklenecek
        gloss_sequence = [word.upper() for word in words]
        return gloss_sequence

class PoseGenerator:
    """Gloss dizilimini 3B Avatar sürmek için anahtar noktalara (keypoints) dönüştürür."""
    def __init__(self, dictionary_path):
        self.dictionary_path = dictionary_path
        # Önceden kaydedilmiş gloss-poz eşleşmeleri
        self.pose_dict = {} 
        
    def generate_animation(self, gloss_sequence):
        animation_frames = []
        for gloss in gloss_sequence:
            if gloss in self.pose_dict:
                animation_frames.append(self.pose_dict[gloss])
            else:
                # Bilinmeyen kelimeler için parmak alfabesi (fingerspelling) tetikle
                pass 
        return animation_frames