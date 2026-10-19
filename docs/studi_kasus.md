# Studi Kasus: Klasifikasi Driver Deforestasi dari Alert Satelit

Berdasarkan seminar Dr. Bart Slachter (Wageningen University and Research) tentang deforestation alerts dan
klasifikasi direct driver menggunakan citra Sentinel-1/Sentinel-2 dan convolutional neural network (CNN).

## 1. Latar Belakang

Hutan tropis menyimpan karbon dalam jumlah besar, mengatur siklus air, menjadi habitat keanekaragaman hayati,
dan menopang mata pencaharian. Hutan ini hilang dengan cepat akibat alih fungsi lahan, penebangan, dan
pertambangan. Sejak hadirnya satelit Sentinel, sistem **deforestation alert** dapat mendeteksi gangguan hutan
sekitar tiap minggu pada piksel 10–30 m.

Keterbatasan utama alert adalah ia hanya menunjukkan *di mana* hutan berubah, bukan *mengapa*. Gangguan bisa
berasal dari pertanian, jalan, logging selektif, tambang, kebakaran, banjir, atau badai. Materi seminar
mengatasi hal ini dengan CNN yang mengklasifikasikan **direct driver** setiap alert dari tumpukan citra radar dan optik.

## 2. Rumusan Masalah

1. Bagaimana mengklasifikasikan direct driver sebuah alert dari tumpukan citra Sentinel-1 dan Sentinel-2?
2. Apakah deep learning (CNN) lebih baik daripada machine learning konvensional dengan fitur buatan tangan?
3. Sensor mana yang lebih berguna ketika citra optik tertutup awan, dan apakah penggabungan keduanya membantu?
4. Bagaimana menangani prediksi yang tidak meyakinkan agar tidak menyesatkan pengguna?

## 3. Tujuan

Membangun prototipe klasifikasi driver, membandingkannya dengan baseline konvensional, mengukur ketahanannya
terhadap awan, dan menunjukkan trade-off antara cakupan dan akurasi lewat ambang keyakinan.

## 4. Pemetaan Materi ke Implementasi

| Poin di materi | Implementasi di proyek |
|---|---|
| Optik tertutup awan, radar konsisten tetapi sulit dibaca | `apply_clouds()` hanya menutup pita Sentinel-2; Sentinel-1 tidak terpengaruh |
| Input CNN: 2 pita Sentinel-1 + 4 pita Sentinel-2 ditumpuk jadi citra multi-pita | 6 pita + 1 kanal mask alert = 7 kanal pada `SmallCNN` |
| Model belajar sendiri kapan memakai radar atau optik | Satu CNN dengan seluruh kanal; tidak ada aturan eksplisit pemilihan sensor |
| Klasifikasi: pertanian skala kecil/besar, jalan, logging selektif, tambang, kebakaran, banjir | `CLASSES` (7 kelas) |
| Model melihat konteks dan bentuk, bukan hanya ukuran alert | Patch 32x32 piksel di sekitar alert; baseline RF hanya memakai fitur bentuk dan spektral ringkas |
| Ada label "tidak yakin" karena model tidak selalu akurat | Ambang keyakinan τ; di bawahnya diberi `tidak_pasti` (`predict.py`) |
| Kebingungan utama: kebakaran vs banjir (sering terjadi di dataran banjir) | Kelas dibuat sengaja mirip (`CONFUSER`); dilaporkan lewat confusion matrix |
| Trade-off kecepatan dan akurasi, latensi bulanan | Dicatat sebagai keterbatasan pada keluaran `predict.py` |
| Pola antar wilayah (Amazon, Kongo, Kalimantan), musiman kebakaran | Metadata wilayah/bulan sintetis; grafik komposisi driver per wilayah |

Tiga pertanyaan di sesi tanya jawab dijadikan eksperimen:

- **Kapan deep learning layak dipakai?** Pembicara menyarankan selalu memulai dari metode konvensional sebagai
  benchmark, lalu membandingkan apakah peningkatan akurasi sebanding dengan tambahan komputasi. Eksperimen 1 mengikuti saran ini.
- **Kapan memakai Sentinel-1 vs Sentinel-2?** Eksperimen 2 (ablasi sensor di bawah awan).
- **Apakah data satelit saja cukup untuk menentukan penyebab?** Tidak sepenuhnya: kelas hanya berupa direct driver
  yang terlihat dari citra, dan penyebab sebenarnya bisa lebih kompleks (lihat bagian Keterbatasan).

## 5. Data

**Data pada repositori ini sintetis.** Citra Sentinel nyata berlabel tidak tersedia di sini, sehingga
`src/simulate_data.py` membangkitkan patch 32x32 dengan 6 pita + mask alert. Bentuk spasial dan tanda spektral
kelas dirancang mengikuti deskripsi di materi (misalnya jalan = garis tipis, tambang = area sangat terang dengan
kolam, banjir = pita di sepanjang sungai dengan backscatter radar rendah).

Agar masalah tidak trivial, simulator menambahkan:
- campuran spektral antar kelas yang mirip (kebakaran–banjir, pertanian kecil–besar, jalan–logging),
- 20% kejadian dengan bentuk tidak khas (bentuk kelas mirip, spektral kelas asli),
- mask alert yang berderau (tepi meleset, piksel hilang, piksel positif palsu),
- derau speckle pada radar dan awan acak pada optik.

Tingkat kesulitan ini dipilih oleh pengembang. Versi awal simulator terlalu mudah (Random Forest mencapai 99%),
sehingga derau mask dan bentuk tidak khas ditambahkan. **Karena itu angka akurasi bergantung pada simulator dan
tidak dapat dibaca sebagai akurasi pada citra nyata.**

## 6. Metode

- **Pembagian data:** 400 patch/kelas untuk latih+validasi (85/15), 150 patch/kelas untuk uji (total uji 1.050).
- **Augmentasi awan:** set latih/validasi = versi bersih + versi berawan (tutupan acak 20–95%) agar model terbiasa dengan awan.
- **Baseline:** Random Forest (300 pohon) pada 27 fitur: bentuk alert (luas, extent, elongasi, rasio keliling, jumlah
  komponen) dan kontras spektral di dalam vs di luar alert.
- **CNN:** 5 blok konvolusi (16-16-32-32-64 filter), BatchNorm, global average pooling, ±36 ribu parameter.
  Dilatih 15 epoch (AdamW, OneCycle, augmentasi flip/rotasi), bobot terbaik dipilih berdasarkan **validasi**, bukan data uji.
- **Eksperimen 2 (ablasi):** CNN dengan kanal S1 saja, S2 saja, dan S1+S2, diuji pada tutupan awan 0%, 50%, 90%.
- **Eksperimen 3:** ambang keyakinan τ pada keluaran softmax CNN S1+S2 (awan 50%).

## 7. Hasil (data sintetis, satu seed)

Akurasi pada data uji (1.050 patch):

| Model | Awan 0% | Awan 50% | Awan 90% |
|---|---|---|---|
| Random Forest (fitur buatan tangan) | 0,851 | 0,790 | 0,752 |
| CNN S1+S2 | 0,856 | 0,842 | 0,771 |
| CNN S1 (radar saja) | 0,795 | 0,795 | 0,795 |
| CNN S2 (optik saja) | 0,839 | 0,824 | 0,676 |

Ambang keyakinan (CNN S1+S2, awan 50%):

| τ | Cakupan | Akurasi pada alert berlabel |
|---|---|---|
| 0,5 | 97,1% | 85,6% |
| 0,6 | 86,1% | 89,3% |
| 0,7 | 72,2% | 92,9% |
| 0,8 | 55,5% | 95,7% |
| 0,9 | 40,2% | 98,6% |

Pembacaan hasil:

1. **CNN vs Random Forest.** Pada langit cerah selisihnya (0,5 poin) masih dalam rentang ketidakpastian
   (galat baku sekitar 1,1 poin pada n = 1.050, satu seed). Keunggulan CNN lebih jelas saat berawan 50%
   (sekitar 5 poin) dan kecil pada 90% (sekitar 2 poin). Dengan satu seed dan data sintetis, ini indikasi, bukan bukti.
2. **Radar vs optik.** Optik lebih informatif saat langit cerah (0,839 vs 0,795), tetapi runtuh pada awan 90%
   (0,676), sedangkan radar tetap 0,795. Kestabilan radar terhadap awan memang tertanam di simulator (awan hanya
   mengubah pita optik), jadi hasil ini mengonfirmasi desain, bukan menemukan sesuatu yang baru.
3. **Penggabungan.** Model gabungan terbaik pada awan 0% dan 50%, tetapi pada awan 90% (0,771) berada di bawah
   radar saja (0,795). Model gabungan sempat terganggu oleh optik yang hampir seluruhnya tertutup. Perlu diselidiki
   lebih lanjut (lebih banyak seed, lebih banyak contoh awan tebal saat pelatihan).
4. **Ambang keyakinan.** Menaikkan τ meningkatkan akurasi pada alert berlabel dengan mengorbankan cakupan.
   Nilai τ tergantung kebutuhan: penegakan hukum biasanya memilih akurasi tinggi, pemantauan luas memilih cakupan tinggi.
5. **Per kelas.** Tambang sempurna (1,000) karena tanda spektral kolam dan area sangat terang dibuat sangat khas di
   simulator. Kelas yang paling sulit adalah pertanian skala kecil/besar (F1 sekitar 0,79).

## 8. Keterbatasan

- **Data sintetis.** Hasil tidak mewakili citra Sentinel nyata. Langkah berikut yang bermakna adalah memakai citra
  dan label nyata (misalnya dari data terbuka alert dan driver).
- **Satu seed, satu pembagian data.** Belum ada pengulangan untuk mengukur variasi.
- **Hanya direct driver.** Seperti ditegaskan pembicara, penyebab deforestasi lebih kompleks daripada kelas yang
  terlihat dari satelit (misalnya pertanian skala kecil bisa bercampur logging artisanal dan produksi arang);
  validasi lapangan tetap diperlukan.
- **Latensi bulanan** dan trade-off kecepatan vs akurasi: klasifikasi bulanan tidak cocok untuk intervensi lapangan
  yang butuh respons dalam hitungan hari.
- **Driver bisa berubah kemudian.** Alert yang tampak seperti kebakaran bisa berubah menjadi pertanian tahun depan;
  klasifikasi hanya menangkap driver langsung saat kejadian.
- Tidak ada kelas "gangguan alami lain"; kelas tersebut disebut di materi tetapi tidak disimulasikan.

## 9. Pengembangan Lanjut

Memakai data nyata, mengulang dengan beberapa seed dan menambahkan interval kepercayaan, menambah kelas gangguan
alami, mencoba fusi sensor yang sadar-awan (misalnya masukan mask awan), dan kalibrasi probabilitas sebelum
menentukan τ.
