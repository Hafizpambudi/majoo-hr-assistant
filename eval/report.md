# Evaluation report

Model `openai/gpt-oss-20b`, embeddings `gemini-embedding-2` (1536d), retrieval scope `majo-hr`.

## Deterministic metrics

| metric | value |
|---|---|
| questions | 25 |
| tool_routing_accuracy | 1.0 |
| refusal_accuracy | 1.0 |
| false_answer_rate | 0.0 |
| false_refusal_rate | 0.0 |
| citation_rate | 1.0 |
| documented_answers | 19 |
| latency_p50_ms | 17946.5 |
| latency_p95_ms | 22402.2 |

## Ragas metrics (answered questions only)

| metric | value | scored |
|---|---|---|
| faithfulness | 0.675 | 10/10 |
| context_precision | 0.575 | 10/10 |

## Per question

| id | tools | refusal | answer |
|---|---|---|---|
| Q01 | OK | OK | Sisa cuti tahunan Anda adalah **9 hari**.   Anda telah menggunakan 5 hari cuti dan masih ada 1 hari yang sedan |
| Q02 | OK | OK | Sisa cuti EMP-002 adalah 0 hari.   - Cuti tahunan yang telah digunakan adalah 12 hari.   - Tidak ada cuti tahu |
| Q03 | OK | OK | Bawahan langsung EMP-001 adalah:  - Siti Rahmawati, Backend Engineer, Engineering, permanent   - Budi Santoso, |
| Q04 | OK | OK | Sisa cuti tahunan Anda adalah **13 hari**.   Anda dapat memindahkan **maksimal 5 hari** ke tahun berikutnya, a |
| Q05 | OK | OK | Tidak, cuti tahunan tidak dapat diuangkan kecuali saat pemutusan hubungan kerja sesuai ketentuan perundang-und |
| Q06 | OK | OK | Cuti melahirkan di Majoo adalah **3 bulan (90 hari kalender)** dengan upah penuh.   - Cuti dapat dimulai 1 bul |
| Q07 | OK | OK | Batas carry over cuti maksimal **5 hari**.   Menurut kebijakan Majoo, sisa cuti tahunan maksimal 5 hari dapat  |
| Q08 | OK | OK | Batas pengajuan carry over cuti adalah **sebelum 20 Desember tahun berjalan**.   Karyawan harus mengajukan per |
| Q09 | OK | OK | Minimal 2 hari per minggu.   Kebijakan kerja hybrid Majoo mensyaratkan karyawan hadir di kantor minimal dua ha |
| Q10 | OK | OK | Plafon asuransi kesehatan swasta di Majoo adalah Rp25.000.000 per tahun per karyawan.   Karyawan tetap mempero |
| Q11 | OK | OK | Tidak. Asuransi kesehatan swasta Majoo hanya menanggung karyawan, satu pasangan resmi, dan maksimal tiga anak  |
| Q12 | OK | OK | Proses reimbursement maksimal 14 hari kalender sejak pengeluaran.   Pengajuan harus dilakukan dalam 14 hari ka |
| Q13 | OK | OK | Rp350.000 per hari.   Per diem perjalanan dinas dalam negeri mencakup makan dan transportasi lokal, dan tidak  |
| Q14 | OK | OK | THR dibayarkan satu kali setahun, paling lambat 7 hari sebelum Idul Fitri [1][2].   Besarnya 1 kali gaji bulan |
| Q15 | OK | OK | Tidak, karyawan tidak berhak atas cuti tahunan selama masa percobaan.   Selama masa percobaan 3 bulan, karyawa |
| Q16 | OK | OK | 30 hari kalender.   Karyawan tetap wajib memberikan pemberitahuan tertulis minimal 30 hari kalender sebelum ta |
| Q17 | OK | OK | Sanksinya adalah biaya penggantian yang dipotong dari gaji terakhir atau ditagihkan ke karyawan.   Aset yang t |
| Q18 | OK | OK | Batas hadiah dari vendor yang boleh diterima tanpa laporan adalah **Rp500.000**.   Hadiah di bawah Rp500.000 d |
| Q19 | OK | OK | Investigasi laporan whistleblowing dilakukan maksimal 30 hari kalender sejak laporan diterima.   Laporan dapat |
| Q20 | OK | OK | Gaji dibayarkan setiap tanggal 25.   Jika tanggal 25 jatuh pada hari libur, gaji dibayarkan pada hari kerja se |
| Q21 | OK | OK | Tidak, Majoo tidak menyediakan kasbon maupun pinjaman karyawan.   Majoo tidak menyediakan kasbon, pinjaman kar |
| Q22 | OK | OK | Durasi PIP adalah **60 hari**.   PIP berlaku bila performa berada pada tingkat Below Expectation selama satu s |
| Q23 | OK | OK | Maaf, saya tidak tahu. Informasi itu tidak ada di dokumen HR yang saya akses. |
| Q24 | OK | OK | Maaf, saya tidak tahu. Informasi itu tidak ada di dokumen HR yang saya akses. |
| Q25 | OK | OK | Maaf, saya tidak tahu. Informasi itu tidak ada di dokumen HR yang saya akses. |

