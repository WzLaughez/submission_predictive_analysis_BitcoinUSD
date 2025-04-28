# Laporan Proyek Machine Learning - Muhammad Fariz Ramadhan

## Domain Proyek
Domain dari proyek ini merupakan ekonomi dan bisnis dalam menunjang 
Prediksi khususnya Time Series menjadi salah satu strategi dalam menentukan apa yang akan terjadi kedepannya sehingga ilmu mengenai machine learning tersebut sangat berguna dalam data yang terus mengalami perubahan bergantung kepada waktu. Contoh penerapan time series adalah menentukan kapan waktu investasi yang terbaik terhadap perubahan harga-harga stock market yang terjadi di dunia nyata. Bitcoin merupakan mata uang digital yang merepresentasikan perubahan penting yang dapat memberikan dampak untuk sektor keuangan telah berkembang sejak beberapa tahun belakangan. Masih banyak pemilihan investasi tanpa mengetahui pengetahuan dan pergerakan selanjutnya agar dapat mempersiapkan dampaknya sedini mungkin. Investasi bitcoin merupakan salah satu mata uang digital yang dimana ilmunya masih belum banyak yang mengetahui sehingga strategi untuk berinvestasi di mata uang masih kurang jika berdasarkan data.
**Rubrik/Kriteria Tambahan (Opsional)**:
Masalah tersebut harus diselesaikan agar kita tidak melakukan investasi dengan cara asal-asalan tanpa hitung-hitungan. Menyelesaikan masalah tersebut menggunakan Machine Learning diharapkan dapat memberikan prediksi yang telah dihitung sebelumnya dan sesuai data.
beberapa riset terkait salah satunya adalah
[Prediksi Harga Bitcoin Menggunakan Metode Random Forest](http://jurnal.pcr.ac.id/index.php/jkt/article/view/4618)

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Kurangnya strategi investasi berdasarkan data
Banyak orang berinvestasi Bitcoin tanpa menggunakan analisis prediktif berbasis data, sehingga keputusan investasi bersifat spekulatif dan pengandai-andaian.
- Kebutuhan akan prediksi pergerakan harga Bitcoin
Fluktuasi harga Bitcoin yang tinggi membutuhkan model prediksi yang mampu membantu investor memperkirakan arah harga berikutnya untuk pengambilan keputusan yang lebih rasional.
- Minimnya penggunaan teknologi Machine Learning di dalam pengambilan keputusan investasi Bitcoin
Masih banyak investor yang belum memanfaatkan teknologi Machine Learning untuk memprediksi pergerakan harga dan meminimalkan risiko investasi

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Memanfaatkan data sebagai alat untuk membangun sistem prediksi harga Bitcoin
Dengan menggunakan data historis kita dapat membangun sistem prediksi yang dapat memperkirakan harga bitcoin ke depan.
- Memberikan alat bantu pengambilan keputusan investasi
Dengan adanya model prediksi, diharapkan dapat membantu calon investor untuk mengambil keputusan investasi Bitcoin secara lebih terinformasi, bukan sekadar spekulasi
- Meningkatkan akurasi dan rasionalitas dalam berinvestasi Bitcoin dengan memanfaatkan Machine Learning


**Rubrik/Kriteria Tambahan (Opsional)**:

    ### Solution statements
    - Menggunakan beberapa algoritma seperti Machine Learning konvensional dam deep learning seperti LSTM dan GRU
    - Mencapai akurasi setidaknya mencapai 0.0n error sekecil"nya dengan menggunakan callback

## Data Understanding
Data yang akan digunakan disini adalah data mengenai data history time series digital currency Bitcoin yang transaksi di  USD  
[Kaggle Dataset (Analyzing and Predicting Bitcoin pricing trend)](https://www.kaggle.com/datasets/surajjha101/analyzing-and-prediction-of-bitcoin-pricing).

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- Date      : Tanggal dimana pasar dibuka
- Open      : Harga BTC saat market dibuka
- High      : Harga tertinggi yang diperoleh saat hari tersebut
- Low       : Harga terendah yang diperoleh saat hari tersebut
- Close     : Harga BTC saat market ditutup
- Adj Close : Penyesuaian harga BTC beberapa faktor pada jam terakhir
- Volume    : Berapa banyak beli/jual terjadi atau berapa banyak perdagangan yang terjadi 

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.
- Disini kita mengeksplorasi data terlebih dahulu dengan mengecek korelasi pada data numerik
terlihat bahwa seluruh kolom sangat kuat korelas
- Kemudian terlihat di visualisasi bahwa kenaikan seluruh fitur terjadi pada tahun 2021 karena dipengaruhi oleh kondisi ekonomi global tepatnya saat terjadi wabah covid-19

## Data Preparation
Teknik yang diperlukan dalam penyelesaian ini adalah normalisasi data. Normalisasi diperlukan agar model deep learning lebih cepat belajar dan lebih akurat. 
Kemudian melakukan sliding window dengan membagi data ke dalam window dan memproses setiap window secara independen.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
1. Menggunakan MinMaxScaler dengan mencari nilai minimum dan maksimum dari seluruh data dan setiap data dikurangi minimum lalu dibagi jarak max-min
2. Sliding window dengan memilih panjang sequence (disini 60 hari) berarti inputnya 60 hari berturut-turut

- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.
Normalisasi dibuat agar Membuat semua fitur dalam skala kecil dan sama, supaya model lebih cepat dan stabil belajar
Sliding Window dibuat agar potongan data time series berukuran tetap supaya bisa dijadikan input ke model

## Modeling
Pada proyek ini, digunakan pendekatan time series forecasting multivariate dengan model berbasis deep learning. Model utama yang diterapkan adalah Long Short-Term Memory (LSTM) dan Gated Recurrent Unit (GRU). LSTM adalah varian dari Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah vanishing gradient pada data sekuensial. 

**Rubrik/Kriteria Tambahan (Opsional)**: 
Kelebihan LSTM:
- Dapat menangkap pola time series jangka panjang.
- Mengatasi masalah short-term memory pada RNN biasa.
- Dapat digunakan untuk prediksi multi-step dan multivariate.
Kekurangan LSTM:
- Memerlukan waktu training lebih lama dibanding model klasik seperti ARIMA.
- Membutuhkan data yang telah dinormalisasi dan dipersiapkan dengan benar.
- Model lebih sulit untuk diinterpretasikan dibanding metode statistik tradisional.

GRU :
- GRU memiliki lebih sedikit parameter daripada LSTM.
- Training lebih cepat tetapi kadang kalah akurat untuk pola sangat kompleks.

- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.
Dipilih: LSTM
Alasan:
- Memberikan loss validasi lebih rendah.
- Lebih stabil saat memprediksi 30 hari ke depan.
- Lebih mampu mempelajari ketergantungan jangka panjang antar fitur (Open, High, Low, Close, Volume).
## Evaluation
Karena ini masalah regresi time series, maka metrik yang digunakan adalah MSE, RMSE, dan MAE

- Mean Squared Error (MSE)
Mengukur rata-rata kesalahan kuadrat antara nilai aktual (y) dan prediksi (ŷ).
Sensitif terhadap kesalahan besar.

- Root Mean Squared Error (RMSE)
Interpretasi lebih intuitif karena satuannya sama dengan target.

- Mean Absolute Error (MAE)
Mengukur rata-rata kesalahan absolut prediksi.
Lebih tahan terhadap outlier dibanding MSE.

LSTM
Training Loss sangat kecil (sekitar 0.0008).
Validation Loss lebih besar (0.0071 - 0.0117).

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.
MSE = (1 / n) × jumlah dari (yᵢ - yᵢ_prediksi) kuadrat, untuk semua i dari 1 sampai n.
- Hitung selisih antara nilai aktual dan prediksi.
- Kuadratkan setiap selisih tersebut.
- Jumlahkan semua hasil kuadrat tadi.
- Bagi dengan jumlah data (n).

Bagaimana MSE bekerja:
MSE menghitung rata-rata dari kuadrat selisih antara prediksi dan nilai sebenarnya.
Karena dikwadratkan, kesalahan besar akan diperbesar dampaknya, sehingga MSE sangat sensitif terhadap outlier (prediksi yang jauh meleset).
Semakin kecil nilai MSE, semakin baik model dalam memprediksi.

Kelebihan:
Memberikan penalti lebih besar pada kesalahan besar.
Cocok untuk menghindari prediksi yang jauh dari kenyataan.

Kekurangan:
Sangat sensitif terhadap outlier.

RMSE = akar kuadrat dari MSE.
Bagaimana RMSE bekerja:
- RMSE adalah akar kuadrat dari MSE.
- RMSE mengembalikan satuan error ke skala yang sama dengan target asli (contoh: harga Bitcoin dalam USD, bukan USD²).
- RMSE lebih mudah untuk diinterpretasikan dibanding MSE karena unitnya sama dengan target.


MAE = (1 / n) × jumlah dari nilai absolut (yᵢ - yᵢ_prediksi), untuk semua i dari 1 sampai n.
Artinya:
- Hitung selisih antara nilai aktual dan prediksi.
- Ambil nilai absolut dari setiap selisih tersebut.
- Jumlahkan semua nilai absolut tadi.
- Bagi dengan jumlah data (n).

Bagaimana MAE bekerja:
- MAE menghitung rata-rata nilai absolut selisih antara prediksi dan nilai aktual.
- Berbeda dengan MSE, MAE tidak mengkuadratkan selisih, sehingga lebih tahan terhadap outlier.
**---Ini adalah bagian akhir laporan---**



