# 🧠 VAE Neural Network Dashboard

### Interactive Handwritten Digit Generation using Variational Autoencoders

An interactive deep learning application built with **Python, PyTorch, Streamlit and MNIST**. This project demonstrates how Variational Autoencoders (VAEs) learn compressed representations of handwritten digits and generate new digit images using a two-dimensional latent space.

---

## 🚀 Project Overview

The VAE Neural Network Dashboard provides an interactive environment to explore deep generative learning.

The application includes model training, latent space visualization, digit generation and training performance monitoring.

## ✨ Key Features

- 🧠 **Variational Autoencoder:** Encoder-decoder architecture for learning latent representations.
- 🔍 **Latent Space Visualization:** Explore the learned two-dimensional latent space.
- ✍️ **Digit Generation:** Generate handwritten digits by sampling latent variables.
- 📊 **Training Dashboard:** Monitor training loss and model performance.
- 🎛️ **Interactive Interface:** Explore generated digits using Streamlit sliders.
- 🔢 **Digit Classification:** Additional digit classification functionality.

## 🏗️ Model Architecture

The Variational Autoencoder consists of three major components:

1. **Encoder:** Extracts image features and learns the latent distribution.
2. **Latent Space:** Represents input images using mean and log variance.
3. **Decoder:** Reconstructs images from sampled latent vectors.

The reparameterization trick allows the model to learn through stochastic sampling.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | MNIST dataset |
| Streamlit | Interactive dashboard |
| NumPy | Numerical computation |
| Pandas | Data handling |
| Plotly | Interactive visualization |
| Matplotlib | Graph plotting |

## ⚙️ Training Configuration

| Parameter | Value |
|---|---:|
| Dataset | MNIST |
| Latent Dimension | 2 |
| Batch Size | 128 |
| Epochs | 5 |
| Learning Rate | 0.001 |
| Device | CPU |

## 📁 Project Structure

```text
VAE_Project/
│
├── README.md
├── requirements.txt
├── .gitignore
├── VAE_Project_Overview.docx
│
└── VAE_Project/
    ├── app.py
    ├── vae_model.py
    ├── train_vae.py
    ├── test_vae.py
    ├── digit_classifier.py
    ├── data/
    ├── vae_loss_history.csv
    └── model checkpoints
```

## 💻 Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/vae-neural-network-dashboard.git
```

**2. Navigate to the project directory**

```bash
cd vae-neural-network-dashboard
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Navigate to the application folder**

```bash
cd VAE_Project
```

**5. Run the Streamlit application**

```bash
streamlit run app.py
```

If pretrained model checkpoints are not included, train the models using the provided training scripts before launching features that require them.

## 📈 Learning Outcomes

- Understanding generative deep learning.
- Implementing a Variational Autoencoder using PyTorch.
- Exploring latent representations.
- Applying the reparameterization trick.
- Visualizing model training and generated samples.
- Building interactive machine learning applications.

## 🔮 Future Improvements

- Train with more epochs and experiment with hyperparameters.
- Improve generated image quality.
- Explore higher-dimensional latent spaces.
- Add additional model evaluation metrics.
- Deploy the dashboard online.

## 👨‍💻 Author

**Teenu Singh**

Mathematics Enthusiast | Python Developer | AI & Machine Learning

[GitHub](https://github.com/kings244645-gif) | [LinkedIn](https://www.linkedin.com/in/teenu-singh)

---

⭐ If you find this project interesting, consider giving the repository a star!

*Built with curiosity, mathematics and deep learning.*
