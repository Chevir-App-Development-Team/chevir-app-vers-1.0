import cv2
from src.utils.streamer import VideoStreamer
from src.module_a_s2t.extractor import HolisticExtractor
from src.module_b_t2s.nlp_gloss import TextToGlossConverter

def run_module_a(source=0):
    """Sağır Üretici Akışı: Kameradan işaret dilini metne çevirir."""
    streamer = VideoStreamer(source)
    extractor = HolisticExtractor()
    
    print("Modül A Başlatıldı... (Kapatmak için 'q' tuşuna basın)")
    while True:
        frame, frame_rgb = streamer.get_frame()
        if frame is None:
            break
            
        keypoints = extractor.extract_keypoints(frame_rgb)
        # TODO: keypoints -> I3D Model Inference -> Metin Çıktısı
        
        cv2.putText(frame, "Cevrilen Metin: (Model Ciktisi)", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Chevir - S2T", frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    streamer.release()

def run_module_b(text):
    """Sağır Tüketici Akışı: Yazılı metni Gloss'a çevirir (Avatar render öncesi)."""
    print(f"\nGirdi Metni: '{text}'")
    converter = TextToGlossConverter()
    glosses = converter.translate_to_gloss(text)
    print(f"Ara Dil (Gloss) Çıktısı: {glosses}")
    print("Avatar animasyonu için anahtar noktalar oluşturuluyor...")
    # TODO: PoseGenerator tetiklemesi

if __name__ == "__main__":
    print("Chevir: NSosyal Erişilebilirlik Katmanı")
    print("1. Modül A (İşaret Dili -> Metin) Testi")
    print("2. Modül B (Metin -> İşaret Dili) Testi")
    
    choice = input("Seçiminiz (1/2): ")
    if choice == '1':
        run_module_a()
    elif choice == '2':
        sample_text = input("Çevrilecek metni girin: ")
        run_module_b(sample_text)
    else:
        print("Geçersiz seçim.")