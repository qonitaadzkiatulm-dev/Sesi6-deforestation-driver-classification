"""Konfigurasi bersama: kelas driver, pita citra, dan pemetaan kanal."""

PATCH = 32  # ukuran patch (piksel). Pada resolusi 10 m setara area 320 m x 320 m.

# 7 kelas direct driver (disederhanakan dari materi seminar)
CLASSES = [
    "pertanian_skala_kecil",
    "pertanian_skala_besar",
    "jalan",
    "logging_selektif",
    "tambang",
    "kebakaran",
    "banjir",
]
N_CLASSES = len(CLASSES)

# 6 pita citra seperti pada materi: 2 pita Sentinel-1 (radar) + 4 pita Sentinel-2 (optik)
BANDS = ["S1_VV", "S1_VH", "S2_B2", "S2_B3", "S2_B4", "S2_B8"]
N_BANDS = len(BANDS)
MASK_CH = N_BANDS  # kanal ke-7 = mask area alert (lokasi gangguan hutan yang terdeteksi)

# Subset kanal untuk eksperimen ablasi sensor (selalu menyertakan mask alert)
CHANNEL_SETS = {
    "S1+S2": [0, 1, 2, 3, 4, 5, MASK_CH],
    "S1": [0, 1, MASK_CH],
    "S2": [2, 3, 4, 5, MASK_CH],
}

WILAYAH = ["Amazon", "Cekungan Kongo", "Kalimantan"]
