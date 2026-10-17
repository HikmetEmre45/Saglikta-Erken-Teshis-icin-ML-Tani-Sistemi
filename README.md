# 🩺 Sağlıkta Erken Teşhis İçin Makine Öğrenmesi Destekli Tanı Sistemi
> **Pima Indians Veri Seti ile Tip 2 Diyabet Risk Tahmini, Makale Karşılaştırması ve Açıklanabilir Yapay Zekâ (XAI)**

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Model](https://img.shields.io/badge/En%20%C4%B0yi%20Model-Gradient%20Boosting%20(%2588.31)-green?style=flat-square)
![AUC](https://img.shields.io/badge/ROC%20AUC-0.96-brightgreen?style=flat-square)

Bu çalışma; literatürdeki mevcut temel makine öğrenmesi yaklaşımlarını inceleyerek tıp dünyasında hayati öneme sahip **Recall (Duyarlılık / Tip II Hata Minimizasyonu)** odaklı gelişmiş bir **Klinik Karar Destek Sistemi (CDSS)** geliştirmeyi ve model kararlarını **Açıklanabilir Yapay Zekâ (XAI)** ile klinik teoriye dayandırmayı amaçlamaktadır.

---

## 📌 1. Klinik Problem ve Projenin Amacı

Tip 2 diyabet, erken evrelerde belirgin klinik semptom göstermeyen; ancak kardiyovasküler sistem, böbrek ve görme fonksiyonlarında geri döndürülemez hasarlara yol açan kronik bir hastalıktır. 

* **Problem:** Hekimler yoğun poliklinik şartlarında glikoz, insülin, tansiyon ve BMI gibi çok değişkenli parametreleri aynı anda değerlendirirken sınırda kalan vakaları gözden kaçırabilmektedir.
* **Proje Amacı:** Hekimin yerine teşhis koymak değil; hastaların rutin kan değerlerindeki örüntüleri tarayarak hekime **"Yüksek Riskli Vaka"** uyarısı veren bir ikinci görüş mekanizması oluşturmaktır.

> 🎤 **Sunum Notu / Konuşma Cümlesi:**  
> *"Hocam, bu projede amacımız doktora iş öğretmek ya da doktorun yerine geçmek değil. Tıpta Tip 2 diyabet sinsi ilerler; yoğun poliklinik ortamında doktorun gözünden kaçabilecek sınır vakaları yakalayan ve hekime 'Bu hastaya dikkat et' uyarısı veren bir Klinik Karar Destek Sistemi kurmaktır."*

---

## 📚 2. Literatür Taraması & İncelenen Referans Makale

Bu çalışmada literatür kıyaslaması olarak şu araştırma temel alınmıştır:
* **Referans:** *Alzboon, M. S., Al-Batah, M., Alqaraleh, M., Abuashour, A., & Bader, A. F. (2023). A Comparative Study of Machine Learning Techniques for Early Prediction of Diabetes.* IEEE ComNet.

### Makalenin Yaklaşımı ve Eksik Yönleri:
1. İlgili makalede temel sınıflandırma modelleri kullanılmış ve en yüksek **%76 genel doğruluk (Accuracy)** oranına ulaşılmıştır.
2. Makale yalnızca genel doğruluğa odaklanmış; tıp etiği ve hasta güvenliği açısından kritik olan **hastayı kaçırmama (Recall / Tip II Hata)** metriğini arka planda bırakmıştır.
3. Hatalı ölçülen 0 değerleri (biyolojik olarak imkansız olan tansiyon, glikoz vb.) için derinlemesine bir veri kurtarma stratejisi izlenmemiştir.

> 🎤 **Sunum Notu / Konuşma Cümlesi:**  
> *"2023 IEEE ComNet konferansında yayımlanan Alzboon ve arkadaşlarının makalesini incelediğimde, temel modellerle tavan başarının %76 doğrulukta kaldığını gördüm. Üstelik makale sadece 'Doğruluk' metriğine odaklanmıştı. Ancak bir hastanede en büyük felaket, hasta birine 'sen sağlıklısın' deyip eve göndermektir. Ben bu projede bu eksikliği gidermek için yola çıktım."*

---

## 🔬 3. Veri Seti ve Veri Kurtarma (Ön İşleme)

Çalışmada **National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)** tarafından paylaşılan **Pima Indians Diabetes** veri seti kullanılmıştır.

* **Toplam Kayıt:** 768 Kadın Hasta
* **Sınıf Dağılımı:** 500 Sağlıklı (%65.1) / 268 Diyabetik (%34.9)
* **Klinik Değişkenler:** Gebelik Sayısı, Glikoz, Kan Basıncı (Tansiyon), Deri Kalınlığı, İnsülin, Vücut Kitle İndeksi (BMI), Genetik Soyağacı Skoru, Yaş.

### Ön İşleme Farkımız:
1. **Veri Kurtarma (İmputasyon):** Canlı bir insanda Kan Basıncı, Glikoz, BMI ve İnsülin değerlerinin "0" olması fizyolojik olarak imkânsızdır. Bu kayıtları doğrudan silmek yerine, ait oldukları sınıfların (Sağlıklı/Diyabetik) medyan değerleriyle doldurularak veri kaybı önlenmiştir.
2. **Katmanlı Bölümleme (Stratified Split):** Veri dengesizliğini korumak için veri seti %80 Eğitim, %20 Test olarak katmanlı bölünmüştür.
3. **Standartlaştırma:** Modellerin değişken ölçeklerinden etkilenmemesi için özellikler standartlaştırılmıştır.

> 🎤 **Sunum Notu / Konuşma Cümlesi:**  
> *"Veri setinde yaşayan bir insanda asla sıfır olamayacak glikoz, tansiyon, insülin gibi değerlere 0 girilmişti. Bunları silseydik hasta verisi kaybedecektik. Ben bu değerleri sağlıklı ve diyabetik sınıfların kendi medyanlarıyla doldurarak veri kaybını engelledim ve %80-%20 katmanlı olarak test setimi ayırdım."*
