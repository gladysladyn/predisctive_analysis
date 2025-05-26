# Laporan Proyek Machine Learning - Gladys Lady Nathasha

## Domain Proyek

Pendidikan adalah komponen penting dalam pembangunan negara. Kemampuan siswa untuk menyelesaikan studi mereka pada waktunya adalah indikator utama keberhasilan sistem pendidikan. Namun, masalah seperti kesulitan akademik, kekurangan dukungan belajar, dan masalah sosial dan ekonomi seringkali menyebabkan siswa terlambat lulus atau bahkan putus sekolah. Permasalahan ini tidak hanya berdampak pada individu, tetapi juga dapat mengganggu efisiensi sistem pendidikan secara keseluruhan.   

Dalam upaya mengatasi persoalan tersebut, pendekatan berbasis teknologi mulai dilirik untuk mendeteksi potensi keterlambatan kelulusan sedini mungkin. Salah satu pendekatan yang menjanjikan adalah penerapan machine learning dalam pendidikan, khususnya untuk memprediksi kelulusan siswa berdasarkan data historis akademik dan faktor-faktor pendukung lainnya.   

Naibaho [1] menunjukkan bahwa algoritma pembelajaran mesin dapat digunakan untuk memprediksi kelulusan siswa Sekolah Menengah Pertama (SMP) dengan akurat dengan memperhitungkan nilai akademik, kehadiran, dan latar belakang siswa. Teknik ini terbukti dapat membantu sekolah mengatasi siswa yang berisiko lebih awal.   

Selain itu, Fatunnisa dan Marcos [2] menggunakan algoritma Random Forest untuk memprediksi kelulusan tepat waktu siswa yang mengambil kelas Teknik Komputer di SMK. Studi mereka menunjukkan bahwa penggunaan model prediktif tidak hanya meningkatkan efisiensi pengelolaan pendidikan, tetapi juga memberi guru dan wali kelas kesempatan untuk memberikan perhatian lebih pada siswa yang kurang beruntung.   

Secara lebih luas, tinjauan sistematis oleh Almarzouqi et al. [3] menunjukkan bahwa penggunaan pembelajaran mesin dalam memprediksi kelulusan siswa telah membantu lembaga pendidikan tinggi dalam perencanaan, pengambilan keputusan, dan pembuatan strategi akademik.   

Dengan mengacu pada studi-studi tersebut, proyek ini bertujuan untuk membangun model prediksi kelulusan siswa menggunakan algoritma machine learning. Diharapkan model ini dapat membantu institusi pendidikan, khususnya pada jenjang pendidikan menengah dan kejuruan, dalam mengidentifikasi siswa yang berpotensi mengalami keterlambatan kelulusan dan memberikan intervensi yang lebih tepat sasaran.      

Referensi:   
[1] A. Naibaho, "Prediksi Kelulusan Siswa Sekolah Menengah Pertama Menggunakan Machine Learning," Jurnal Informatika dan Teknik Elektro Terapan, vol. 11, no. 3, pp. 45–52, Jul. 2023. [Online]. Tersedia: https://journal.eng.unila.ac.id/index.php/jitet/article/view/3056   

[2] A. Fatunnisa dan H. Marcos, "Prediksi Kelulusan Tepat Waktu Siswa SMK Teknik Komputer Menggunakan Algoritma Random Forest," Jurnal Manajemen Informatika (JAMIKA), vol. 14, no. 1, pp. 101–110, Apr. 2024. [Online]. Tersedia: https://ojs.unikom.ac.id/index.php/jamika/article/download/12114/4255/   

[3] A. Almarzouqi, S. Alketbi, and M. Alneyadi, “Predicting University Student Graduation Using Academic Performance and Machine Learning: A Systematic Literature Review,” ResearchGate, Dec. 2023. [Online]. Available: https://www.researchgate.net/publication/377942662_Predicting_University_Student_Graduation_Using_Academic_Performance_and_Machine_Learning_A_Systematic_Literature_Review   

## Business Understanding

Kelulusan siswa merupakan indikator utama dalam mengevaluasi efektivitas sistem pendidikan. Kegagalan siswa menyelesaikan pendidikan tepat waktu memengaruhi siswa itu sendiri dan kualitas institusi pendidikan secara keseluruhan. Oleh karena itu, menjadi sangat penting untuk mengetahui kapan siswa akan lulus agar intervensi yang tepat dapat diberikan kepada siswa yang berisiko.   
Berbagai informasi tentang siswa, seperti nilai ujian, tingkat kehadiran, demografi, dan faktor sosial lainnya, biasanya disimpan oleh institusi pendidikan. Meskipun demikian, data tersebut seringkali tidak digunakan dengan benar untuk analisis prediktif. Dengan menggunakan pendekatan machine learning, data-data tersebut dapat diolah menjadi sistem prediktif yang membantu pengambilan keputusan yang lebih objektif dan berbasis data (data-driven decision making).   

### Problem Statements
  
1. Bagaimana cara memanfaatkan data akademik siswa untuk memprediksi kemungkinan kelulusan mereka?   
2. Fitur-fitur apa saja yang paling berpengaruh terhadap kelulusan siswa?   
3. Algoritma machine learning mana yang memberikan performa terbaik dalam memprediksi kelulusan siswa?   

### Goals

1. Membangun model klasifikasi untuk memprediksi kelulusan siswa berdasarkan data ujian dan faktor-faktor pendukung lainnya.   
2. Mengidentifikasi fitur-fitur dominan yang memengaruhi kelulusan.   
3. Membandingkan performa beberapa algoritma klasifikasi untuk mendapatkan model terbaik.   

### Solution statements
Untuk mencapai tujuan tersebut, solusi yang diusulkan melibatkan:   
1. Eksplorasi dan pra-pemrosesan data, termasuk data nilai ujian, jenis kelamin, status persiapan ujian, dan tingkat pendidikan orang tua.   
2. Implementasi beberapa algoritma klasifikasi, seperti:   
* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* Gradient Boosting (XGBoost)   
3. Evaluasi dan pemilihan model terbaik berdasarkan metrik klasifikasi:      
* Accuracy: Persentase prediksi yang benar.
* Precision: Proporsi prediksi lulus yang benar-benar lulus.
* Recall: Kemampuan model mendeteksi semua siswa yang lulus.
* F1-score: Harmoni antara precision dan recall.   
4. Melakukan hyperparameter tuning pada model terbaik (misalnya Random Forest atau XGBoost) untuk meningkatkan performa secara signifikan.   
5. Visualisasi hasil penting seperti confusion matrix dan feature importance untuk memperjelas hasil analisis dan mendukung proses interpretasi oleh pihak non-teknis.   

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Students Performance in Exams, yang dapat diunduh melalui tautan berikut:   

https://www.kaggle.com/datasets/spscientist/students-performance-in-exams   

Dataset ini berisi informasi tentang hasil ujian siswa dari berbagai latar belakang demografis dan akademik. Tujuan penggunaannya adalah untuk membangun model prediksi kelulusan siswa berdasarkan karakteristik dan skor mereka.     

###  Deskripsi Variabel
Dataset terdiri dari 1000 baris dan 8 kolom, dengan detail sebagai berikut:   
* gender : Jenis kelamin siswa (male atau female)
* race/ethnicity : Kelompok etnis siswa (grup A hingga grup E)
* parental level of education : Tingkat pendidikan tertinggi dari orang tua siswa
* lunch : Jenis makan siang yang diterima siswa (standard atau free/reduced)
* test preparation course : Status mengikuti kursus persiapan ujian (none atau completed)
* math score : Nilai ujian matematika (0–100)
* reading score : Nilai ujian membaca (0–100)
* writing score : Nilai ujian menulis (0–100)   

### Eksplorasi Data (EDA)
1. Struktur dan Tipe Data   
`print(df.info())`   
Semua fitur terisi penuh tanpa nilai kosong. Tiga kolom bertipe numerik (math score, reading score, writing score), sementara lima kolom lainnya bertipe kategorikal.   

2. Statistik Deskriptif   
`print(df.describe())`   
Rata-rata nilai matematika adalah 66.09, membaca 69.17, dan menulis 68.05. Rentang skor dari 0–100 menunjukkan variasi tingkat performa akademik siswa.   

3. Pengecekan Nilai Hilang dan Duplikat   
```
print(df.isnull().sum())   
print(df.duplicated().sum())   
```   
Tidak ditemukan data yang hilang dan data duplikat.   

4. Korelasi Antar Skor   
Visualisasi berikut menunjukkan korelasi antara nilai matematika, membaca, dan menulis yaitu korelasi tertinggi terjadi antara reading score dan writing score (0.95), menunjukkan bahwa siswa dengan kemampuan membaca tinggi cenderung juga unggul dalam menulis.   

Berikut adalah visualisasi korelasi antar skor menggunakan heatmap:
![Grafik Heatmap Korelasi Antar Skor](images/output%201.png "Heatmap Korelasi Antar Skor")

5. Distribusi Variabel Kategorikal   
Distribusi data pada beberapa fitur penting:
* Gender: 518 siswa perempuan dan 482 laki-laki.
* Race/Ethnicity: Grup C mendominasi populasi.
* Parental Level of Education: Sebagian besar orang tua memiliki pendidikan "some college" atau "associate's degree".
* Lunch: 645 siswa menerima makan siang standard, sisanya free/reduced.
* Test Preparation Course: 642 siswa tidak mengikuti kursus, 358 mengikuti kursus.   
Visualisasi distribusi juga disertakan dengan countplot untuk tiap fitur sebagai berikut:   

![Grafik Distribusi Gender](images/output%202.png "Distribusi Gender")   
![Grafik Distribusi Race/Ethnicity](images/output%203.png "Distribusi Race/Ethnicity")   
![Grafik Distribusi Parental Level of Education](images/output%204.png "Distribusi GenderParental Level of Education")   
![Grafik Distribusi Lunch](images/output%205.png "Distribusi Lunch")   
![Grafik Distribusi Test Preparation Course](images/output%206.png "Distribusi Test Preparation Course")   

6. Perbandingan Skor berdasarkan Gender   
Boxplot menunjukkan bahwa terdapat perbedaan distribusi skor antara gender untuk ketiga mata pelajaran, meskipun tidak terlalu signifikan secara visual.
* Perempuan sedikit unggul di reading dan writing.
* Laki-laki sedikit unggul di math.   

Berikut adalah visualisasi dari perbandingan skor berdasarkan gender:   
![Grafik Distribusi Math Score Antara Gender](images/output%207.png "Distribusi Math Score Antara Gender")   
![Grafik Distribusi Reading Score Antara Gender](images/output%208.png "Distribusi Reading Score Antara Gender")   
![Grafik Distribusi Writing Score Antara Gender](images/output%209.png "Distribusi Writing Score Antara Gender")   

## Data Preparation
Tahapan ini bertujuan untuk mempersiapkan data agar siap digunakan dalam pelatihan model machine learning. Beberapa langkah dilakukan secara sistematis untuk memastikan kualitas dan konsistensi data.   

1. Membuat Kolom Target pass   
Langkah pertama adalah menentukan label target klasifikasi, yaitu apakah siswa lulus atau tidak lulus. Kriteria kelulusan ditentukan berdasarkan rata-rata nilai dari tiga mata pelajaran utama: matematika, membaca, dan menulis.   
```
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['pass'] = df['average_score'].apply(lambda x: 1 if x >= 60 else 0)
```   
* pass = 1 → siswa lulus jika rata-rata skor ≥ 60
* pass = 0 → siswa tidak lulus jika rata-rata skor < 60   
Langkah ini penting untuk mengubah problem menjadi masalah klasifikasi biner.   

2. Menghapus Kolom average_score (Opsional)   
Kolom average_score bersifat turunan (derived feature) yang tidak diperlukan dalam pelatihan model.   
`df.drop('average_score', axis=1, inplace=True)`   

3. Encoding Variabel Kategorikal   
Karena model machine learning tidak dapat bekerja langsung dengan data bertipe teks/kategorikal, maka fitur-fitur kategorikal perlu diubah menjadi bentuk numerik. Di sini digunakan teknik Label Encoding untuk setiap kolom kategorikal.   

4. Memisahkan Fitur dan Target   
Setelah semua fitur disiapkan, dilakukan pemisahan antara fitur independen (X) dan target (y).   
```
X = df.drop('pass', axis=1)
y = df['pass']
```   

5. Split Data: Training dan Testing   
Dataset dibagi menjadi dua bagian, yaitu 80% untuk pelatihan (training set) dan 20% untuk pengujian (testing set).   
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`   
Split ini penting agar model dapat diuji pada data yang belum pernah dilihat sebelumnya.   

6. Standardisasi Data   
Fitur numerik dinormalisasi menggunakan StandardScaler agar memiliki distribusi dengan mean = 0 dan standar deviasi = 1. Ini membantu mempercepat proses pelatihan dan meningkatkan performa beberapa algoritma, seperti Logistic Regression atau SVM.   
```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```  

## Modeling
Tahap ini bertujuan untuk membangun model machine learning yang dapat memprediksi apakah seorang siswa lulus atau tidak berdasarkan skor ujian dan informasi demografis. Proses pemodelan dilakukan dalam beberapa tahapan:   

### Tahapan Proses Modeling:
1. Pemilihan Algoritma   
Empat algoritma digunakan untuk klasifikasi:
* Logistic Regression
* Random Forest Classifier
* Support Vector Machine (SVM)
* XGBoost Classifier   
2. Pelatihan Model (Training)   
Masing-masing algoritma dilatih menggunakan data training (X_train, y_train) yang telah melalui proses standardisasi.   
3. Prediksi (Prediction)   
Setelah dilatih, model digunakan untuk memprediksi data testing (X_test) dan menghasilkan y_pred.   
4. Pencatatan Parameter   
Parameter default digunakan pada semua model kecuali dinyatakan berbeda. Parameter ini dapat dioptimalkan lebih lanjut pada tahap tuning jika diperlukan.   

**Model 1: Logistic Regression**   
Logistic Regression merupakan algoritma klasifikasi yang bersifat sederhana namun cukup efektif sebagai baseline dalam berbagai kasus klasifikasi biner. Algoritma ini bekerja dengan memodelkan probabilitas keluaran sebagai fungsi logistik dari kombinasi linier fitur input.   

Parameter penting:   
max_iter=1000 digunakan untuk memastikan model memiliki cukup iterasi agar proses pelatihan dapat mencapai konvergensi, terutama saat data memiliki banyak fitur atau distribusi kompleks.   

Kelebihan:   
* Mudah dipahami dan diimplementasikan
* Cepat dalam proses pelatihan
* Cocok untuk baseline dalam klasifikasi biner   

Kekurangan:   
* Kurang efektif untuk pola non-linear
* Tidak fleksibel jika data mengandung interaksi kompleks antar fitur   

Berikut adalah visualisasi model logistic regression dengan confussion matrix:
![Grafik Confussion Matrix Model Logistic Regression](images/output%2010.png "Confussion Matrix Model Logistic Regression")

**Model 2: Random Forest Classifier**   
Random Forest adalah algoritma ensemble yang membangun banyak decision tree secara acak dan menggabungkan hasilnya untuk menghasilkan prediksi akhir. Teknik ini efektif dalam meningkatkan akurasi dan mengurangi risiko overfitting.   

Parameter penting:   
random_state=42 digunakan agar hasil pelatihan bersifat reproducible. Parameter lain seperti n_estimators, max_depth, dan min_samples_split menggunakan nilai default.   

Kelebihan:   
* Tahan terhadap overfitting
* Dapat menangani data non-linear dan outlier
* Bekerja baik meskipun fitur tidak memiliki skala yang sama   

Kekurangan:   
* Relatif lambat untuk prediksi real-time
* Model lebih kompleks dan sulit diinterpretasikan dibandingkan Logistic Regression   

Berikut adalah visualisasi model random forest dengan confussion matrix:
![Grafik Confussion Matrix Model Random Forest](images/output%2011.png "Confussion Matrix Model Random Forest")   

**Model 3: Support Vector Machine (SVM)**
Support Vector Machine bekerja dengan mencari hyperplane terbaik yang memisahkan kelas dalam ruang berdimensi tinggi. Cocok untuk kasus klasifikasi yang kompleks dan data dengan jumlah fitur besar.   

Parameter penting:   
Secara default, SVC() menggunakan kernel RBF (Radial Basis Function) yang efektif dalam memetakan data yang tidak dapat dipisahkan secara linear ke ruang berdimensi lebih tinggi.   

Kelebihan:   
* Sangat efektif untuk data berdimensi tinggi
* Mampu menangkap pola non-linear dengan kernel yang tepat
* Akurat bila parameter dituning dengan baik   

Kekurangan:   
* Sensitif terhadap skala fitur (perlu normalisasi)
* Lambat saat digunakan pada dataset berukuran besar
* Memerlukan tuning terhadap parameter seperti C dan gamma   

Berikut adalah visualisasi model SVM dengan confussion matrix:
![Grafik Confussion Matrix SVM](images/output%2012.png "Confussion Matrix Model SVM")   

**Model 4: XGBoost Classifier**
XGBoost (Extreme Gradient Boosting) adalah algoritma boosting berbasis pohon yang sangat populer karena performanya yang tinggi dalam berbagai kompetisi data science. Algoritma ini membangun model secara bertahap dengan fokus pada memperbaiki kesalahan prediksi dari model sebelumnya.   

Parameter penting:   
* use_label_encoder=False: Menonaktifkan label encoder bawaan karena sudah tidak direkomendasikan
* eval_metric='logloss': Menentukan metrik evaluasi selama pelatihan untuk klasifikasi biner
Parameter lainnya seperti n_estimators, max_depth, learning_rate, dan subsample tetap menggunakan nilai default.   

Kelebihan:   
* Sangat akurat dan efisien
* Mendukung regularisasi, sehingga mampu mengurangi overfitting
* Dapat menangani data yang tidak seimbang   

Kekurangan:   
* Struktur parameter cukup kompleks
* Membutuhkan tuning yang cermat untuk performa optimal
* Waktu pelatihan bisa lebih lama dibanding model sederhana   

Keempat model di atas dilatih dengan pipeline yang sama, yaitu:   
* Data hasil preprocessing dan standardisasi
* Penggunaan parameter default (kecuali di Logistic Regression dan XGBoost)
* Prediksi dilakukan terhadap X_test   

Berikut adalah visualisasi model SVM dengan confussion matrix:
![Grafik Confussion Matrix XGBoost Classifier](images/output%2013.png "Confussion Matrix XGBoost Classifier")   

Setiap model memiliki karakteristik yang berbeda. Logistic Regression unggul dalam kesederhanaan dan interpretabilitas, namun kurang efektif untuk data non-linear. Random Forest lebih kompleks tetapi tahan terhadap overfitting. SVM sangat akurat pada kasus tertentu namun memerlukan tuning parameter yang teliti. XGBoost dikenal karena keakuratannya, meskipun pelatihannya lebih kompleks dan memakan waktu lebih lama.   

Berdasarkan hasil evaluasi performa, Logistic Regression menunjukkan akurasi dan stabilitas metrik yang paling tinggi dibandingkan model lainnya, sehingga dijadikan model terbaik untuk kasus ini.    

## Evaluation
Setelah seluruh model dilatih, langkah selanjutnya adalah melakukan evaluasi menggunakan data uji. Evaluasi dilakukan menggunakan metrik utama. Evaluasi model dilakukan menggunakan empat metrik utama, yaitu:   
1. Akurasi   
Mengukur seberapa besar proporsi prediksi yang benar dari seluruh prediksi yang dilakukan. Semakin tinggi akurasi, semakin baik model dalam mengenali pola data.   
2. Precision   
Menunjukkan proporsi prediksi positif yang benar-benar positif. Metrik ini penting saat kesalahan prediksi positif memiliki dampak besar.   
3. Recall   
Mengukur seberapa banyak data positif yang berhasil dikenali oleh model dari seluruh data positif yang ada.   
4. F1 Score   
Merupakan rata-rata harmonis dari precision dan recall, dan digunakan untuk menyeimbangkan keduanya terutama saat terdapat ketidakseimbangan data.   
Selain itu, untuk masing-masing model ditampilkan confusion matrix serta visualisasi.    

Berikut adalah hasil evaluasi dari empat model yang dibandingkan:   

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.995    | 0.990     | 1.000  | 0.995    |
| Random Forest       | 0.980    | 0.990     | 0.980  | 0.985    |
| SVM                 | 0.865    | 0.960     | 0.990  | 0.974    |
| XGBoost             | 0.980    | 0.990     | 0.980  | 0.985    |

**Komparasi Model**   
* Logistic Regression unggul dalam akurasi dan F1 Score, serta hanya membuat 1 kesalahan prediksi (False Negative).
* Random Forest dan XGBoost memberikan performa yang sangat mirip, keduanya membuat 4 kesalahan klasifikasi dengan akurasi 98%.
* SVM memiliki akurasi terendah di antara semua model, yaitu 96.5%, dengan kesalahan klasifikasi terbesar (7 kesalahan).

**Model Terbaik**
Berdasarkan evaluasi menyeluruh, Logistic Regression dipilih sebagai model terbaik karena:
* Memiliki akurasi tertinggi (99.5%).
* Nilai F1 Score paling tinggi (0.995).
* Jumlah kesalahan klasifikasi paling sedikit (1 error).
* Model yang lebih sederhana dan mudah diinterpretasi, cocok untuk kasus klasifikasi linier seperti ini.

Hasil evaluasi menunjukkan bahwa semua model memberikan performa yang sangat baik, namun Logistic Regression memiliki nilai akurasi tertinggi, yaitu sekitar 99,5%, serta precision, recall, dan F1 score yang mendekati sempurna. Dengan pertimbangan tersebut, Logistic Regression dipilih sebagai model akhir yang digunakan untuk memprediksi kelulusan siswa.   

**Feature Importance**
Analisis feature importance dari ketiga model (Logistic Regression, Random Forest, dan XGBoost) mengungkap bahwa fitur math score, reading score, dan writing score merupakan prediktor utama dalam menentukan kelulusan. Hal ini sejalan dengan logika domain bahwa performa akademik siswa adalah penentu utama kelulusan.   

Visualisasi koefisien dan kepentingan fitur telah ditampilkan pada grafik sebelumnya menggunakan pustaka seaborn.     

Berikut adalah visualisasi fitur penting yang paling berpengaruh terhadap prediksi:   
1. Feature Influence Logistic Regression
![Grafik Feature Influence Logistic Regression Coefficients](images/output%2014.png "Feature Influence Logistic Regression Coefficients")   
![Grafik Feature Importance Random Forest](images/output%2015.png "Feature Importance Random Forest")   
![Grafik Feature Importance XGBoost](images/output%2016.png "Feature Importance XGBoost")   

## Pengujian Model
Sebagai langkah tambahan untuk menguji kemampuan model dalam situasi nyata, dilakukan pengujian terhadap data baru berupa dua contoh siswa dengan karakteristik berbeda. Model yang digunakan adalah Logistic Regression, yang sebelumnya telah dilatih pada data pelatihan (training data).   

**Contoh Data Baru**
Data Siswa 1   
Gender : Female   
Race/Ethnicity : Group B   
Parental Level of Education : Bachelor's Degree   
Lunch : Standard   
Test Preparation Course : Completed   
Math Score : 72   
Reading Score : 80   
Writing Score : 78   

**Hasil Prediksi**
Setelah dilakukan encoding terhadap fitur kategorikal dan standardisasi pada fitur numerik sesuai dengan proses pelatihan, model memberikan hasil prediksi sebagai berikut:   
Siswa 1: Diprediksi Lulus, dengan probabilitas sebesar 1.00
Siswa 2: Diprediksi Tidak Lulus, dengan probabilitas sebesar 0.00   

**Interpretasi**
Model berhasil memberikan prediksi kelulusan berdasarkan karakteristik siswa baru. Siswa pertama menunjukkan nilai akademik yang tinggi dan memiliki dukungan seperti makanan standar serta mengikuti kursus persiapan ujian. Sebaliknya, siswa kedua memiliki skor akademik lebih rendah, tidak mengikuti persiapan ujian, serta berasal dari keluarga dengan tingkat pendidikan lebih rendah—faktor-faktor ini berkontribusi terhadap prediksi tidak lulus.

## Penutup
Dalam proyek ini, telah dilakukan eksplorasi dan penerapan berbagai algoritma machine learning untuk menyelesaikan masalah klasifikasi pada dataset Students Performance in Exams. Empat model yang diimplementasikan—Logistic Regression, Random Forest, Support Vector Machine (SVM), dan XGBoost—masing-masing menunjukkan karakteristik dan performa yang berbeda berdasarkan pendekatan, kompleksitas, dan kecocokan terhadap data.   

Logistic Regression digunakan sebagai baseline karena sifatnya yang sederhana dan efisien, meskipun kurang mampu menangkap pola non-linear. Random Forest menawarkan ketangguhan terhadap overfitting dan fleksibilitas dalam menangani data kompleks. SVM memberikan akurasi yang tinggi untuk data berdimensi besar namun menuntut praproses dan tuning parameter yang teliti. XGBoost terbukti unggul dalam akurasi dan efisiensi, terutama karena mekanisme boosting dan regularisasi yang dimilikinya.   

Proses pengembangan model dimulai dari praproses data, eksplorasi fitur, pelatihan model, hingga evaluasi menggunakan metrik yang relevan seperti akurasi, precision, recall, dan F1-score. Evaluasi ini memungkinkan analisis performa yang menyeluruh serta penentuan model terbaik untuk digunakan dalam konteks prediksi kelulusan siswa.   

Secara keseluruhan, proyek ini tidak hanya memberikan gambaran praktis mengenai penerapan algoritma machine learning, tetapi juga memperlihatkan pentingnya pemilihan model yang tepat, pemahaman terhadap parameter, serta proses evaluasi yang komprehensif. Hasil dari proyek ini diharapkan dapat menjadi fondasi yang kuat dalam pengembangan solusi prediktif di bidang pendidikan, sekaligus membuka peluang pengembangan lanjutan dengan teknik optimasi atau perluasan data.   