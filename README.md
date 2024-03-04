# TAresnet
Tugas akhir untuk deteksi retinopati diabetik menggunakan resnet dan analisa menggunakan gradcam

Deskripsi file dalam folder "Dataset" \n
FullDataset merupakan dataset yang berasal dari https://drac22.grand-challenge.org/ task 3.
Sedangkan folder test, train, dan validation merupakan dataset yang diturunkan dari FullDataset/a. Training Set.

Deskripsi folder "evaluation" \n
Berisi kode singkat untuk load model yang telah di train, dan mengetes dengan salah satu gambar dari dataset

Progress:\n 
Able to train the model and reached 65% accuracy
Current model when exported to classify an image, give out numerical output (float)

To Do:
1. Interpret the output from the model into readable results
2. Improve the model's accuracy