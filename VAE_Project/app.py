import streamlit as st
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from vae_model import VAE

from digit_classifier import DigitClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. PAGE SETTINGS
# --------------------------------------------------
st.set_page_config(
    page_title="VAE Neural Network Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Variational Autoencoder — Live Dashboard")
st.write(
    "Explore how a trained Variational Autoencoder "
    "encodes and reconstructs handwritten digits."
)

st.divider()

DEVICE = torch.device("cpu")


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        "vae_mnist.pth",
        map_location=DEVICE,
        weights_only=True
    )

    model = VAE(
        latent_dim=checkpoint["latent_dim"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


try:
    load_model.clear()
    model = load_model()
    st.success("Trained VAE model loaded successfully!")
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# --------------------------------------------------
# 3. LOAD MNIST DATASET
# --------------------------------------------------
@st.cache_resource
def load_mnist():
    transform = transforms.ToTensor()

    dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    return dataset


with st.spinner("Loading MNIST dataset..."):
    dataset = load_mnist()


# --------------------------------------------------
# 4. SELECT AN IMAGE
# --------------------------------------------------
st.header("1. Select a Handwritten Digit")

selected_digit = st.selectbox(
    "Choose a digit:",
    list(range(10)),
    index=5
)

matching_indices = (
    dataset.targets == selected_digit
).nonzero(as_tuple=True)[0].tolist()

example_number = st.slider(
    "Choose an example of this digit:",
    min_value=1,
    max_value=10,
    value=1
)

index = matching_indices[example_number - 1]
image, label = dataset[index]

# Add batch dimension: [1, 1, 28, 28]
input_image = image.unsqueeze(0).to(DEVICE)


# --------------------------------------------------
# 5. REAL VAE FORWARD PASS
# --------------------------------------------------
with torch.no_grad():
    features, mu, logvar = model.encode(input_image)

    # Use the mean for a stable reconstruction.
    # This avoids additional random sampling noise.
    z = mu

    reconstruction = model.decode(z)

    original_flat = input_image.view(1, -1)
    reconstructed_flat = reconstruction.view(1, -1)

    reconstruction_loss = F.binary_cross_entropy(
        reconstructed_flat,
        original_flat,
        reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = reconstruction_loss + kl_loss

    std = torch.exp(0.5 * logvar)


# --------------------------------------------------
# 6. ORIGINAL VS RECONSTRUCTION
# --------------------------------------------------
st.divider()
st.header("2. Original vs. Reconstructed Image")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original MNIST Image")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Original Digit: {label}")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("VAE Reconstruction")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(
        reconstruction[0, 0].cpu().numpy(),
        cmap="gray",
        vmin=0,
        vmax=1
    )
    ax.set_title("Reconstructed Digit")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 7. ACTUAL MODEL VALUES
# --------------------------------------------------
st.divider()
st.header("3. Inside the Neural Network")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Encoder Features", features.shape[1])
    st.caption("128 learned features")

with col2:
    st.metric("Latent Dimensions", z.shape[1])
    st.caption("Two coordinates: z₁ and z₂")

with col3:
    st.metric("Reconstruction Size", "28 × 28")
    st.caption("784 output pixels")


st.subheader("Latent Distribution")

latent_col1, latent_col2 = st.columns(2)

with latent_col1:
    st.write("**Mean (μ)**")
    st.write(mu[0].cpu().tolist())

    st.write("**Standard Deviation (σ)**")
    st.write(std[0].cpu().tolist())

with latent_col2:
    st.write("**Log-Variance**")
    st.write(logvar[0].cpu().tolist())

    st.write("**Latent Coordinates (z)**")
    st.write(z[0].cpu().tolist())


# --------------------------------------------------
# 8. ACTUAL LOSS VALUES
# --------------------------------------------------
st.divider()
st.header("4. VAE Loss Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Reconstruction Loss",
        f"{reconstruction_loss.item():.2f}"
    )

with col2:
    st.metric(
        "KL Divergence",
        f"{kl_loss.item():.2f}"
    )

with col3:
    st.metric(
        "Total Loss",
        f"{total_loss.item():.2f}"
    )

st.caption(
    "These losses are calculated for the selected image, "
    "not averaged across the entire training dataset."
)

# --------------------------------------------------
# 9. ANIMATED VAE MOVEMENT TRAIL
# --------------------------------------------------
import streamlit.components.v1 as components

st.divider()
st.header("5. Animated Neural Network Data Flow")

st.write(
    "Watch the data travel through the five stages of the "
    "trained Variational Autoencoder."
)

animation_html = """
<!DOCTYPE html>
<html>
<head>
<style>
body {
    margin: 0;
    padding: 12px;
    background: #0e1117;
    color: white;
    font-family: Arial, sans-serif;
}

.network {
    position: relative;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    min-height: 180px;
    padding: 20px 5px;
}

.stage {
    position: relative;
    z-index: 2;
    flex: 1;
    min-width: 0;
    padding: 18px 5px;
    border: 1px solid #526078;
    border-radius: 12px;
    text-align: center;
    background: #1b2432;
    color: #d7e2f2;
    font-size: 13px;
    transition: background 0.3s, border 0.3s, transform 0.3s;
}

.stage .icon {
    display: block;
    font-size: 27px;
    margin-bottom: 10px;
}

.stage.active {
    background: #164e63;
    border: 2px solid #38bdf8;
    color: white;
    transform: translateY(-7px);
    box-shadow: 0 0 18px #0ea5e955;
}

.track {
    position: absolute;
    left: 8%;
    right: 8%;
    top: 50%;
    height: 4px;
    background: #38465b;
    transform: translateY(-50%);
    z-index: 1;
}

.marker {
    position: absolute;
    top: 50%;
    left: 8%;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background: #67e8f9;
    box-shadow: 0 0 16px #22d3ee;
    transform: translate(-50%, -50%);
    z-index: 3;
    transition: left 0.7s ease-in-out;
}

.status {
    text-align: center;
    margin: 8px 0 18px;
    font-size: 15px;
    color: #bae6fd;
}

.controls {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
}

button {
    padding: 10px 20px;
    border-radius: 8px;
    border: 1px solid #38bdf8;
    background: #172554;
    color: white;
    cursor: pointer;
    font-size: 14px;
}

button:hover {
    background: #075985;
}

.note {
    color: #9ca3af;
    text-align: center;
    font-size: 12px;
    margin-top: 18px;
}

@media (max-width: 650px) {
    .network {
        gap: 3px;
        padding-left: 0;
        padding-right: 0;
    }

    .stage {
        padding: 12px 2px;
        font-size: 10px;
        overflow-wrap: anywhere;
    }

    .stage .icon {
        font-size: 20px;
    }
}
</style>
</head>

<body>

<div class="network" id="network">
    <div class="track"></div>
    <div class="marker" id="marker"></div>

    <div class="stage active" id="stage0">
        <span class="icon">🖼️</span>
        Original<br>Digit
    </div>

    <div class="stage" id="stage1">
        <span class="icon">🧠</span>
        Encoder<br>128 features
    </div>

    <div class="stage" id="stage2">
        <span class="icon">🌌</span>
        Latent Space<br>2 dimensions
    </div>

    <div class="stage" id="stage3">
        <span class="icon">⚙️</span>
        Decoder<br>784 pixels
    </div>

    <div class="stage" id="stage4">
        <span class="icon">✨</span>
        Reconstruction<br>28 × 28
    </div>
</div>

<div class="status" id="status">
    Stage 1 of 5 — Original Digit
</div>

<div class="controls">
    <button onclick="previousStage()">⬅ Previous</button>
    <button onclick="toggleAnimation()" id="playButton">
        ▶ Play Animation
    </button>
    <button onclick="nextStage()">Next ➡</button>
    <button onclick="resetAnimation()">↻ Reset</button>
</div>

<div class="note">
    Visualization of the trained VAE's processing pipeline.
    The model computation occurs in Python; this animation illustrates its stages.
</div>

<script>
const names = [
    "Original Digit",
    "Encoder",
    "Latent Space",
    "Decoder",
    "Reconstruction"
];

const positions = [8, 29, 50, 71, 92];
let current = 0;
let timer = null;

function showStage(index) {
    current = (index + names.length) % names.length;

    for (let i = 0; i < names.length; i++) {
        document.getElementById("stage" + i)
            .classList.toggle("active", i === current);
    }

    document.getElementById("marker").style.left =
        positions[current] + "%";

    document.getElementById("status").textContent =
        "Stage " + (current + 1) + " of 5 — " + names[current];
}

function nextStage() {
    showStage(current + 1);
}

function previousStage() {
    showStage(current - 1);
}

function resetAnimation() {
    stopAnimation();
    showStage(0);
}

function toggleAnimation() {
    if (timer !== null) {
        stopAnimation();
        return;
    }

    document.getElementById("playButton").textContent =
        "⏸ Pause Animation";

    timer = setInterval(function() {
        if (current === names.length - 1) {
            stopAnimation();
            return;
        }

        nextStage();
    }, 1300);
}

function stopAnimation() {
    if (timer !== null) {
        clearInterval(timer);
        timer = null;
    }

    document.getElementById("playButton").textContent =
        "▶ Play Animation";
}

showStage(0);
</script>

</body>
</html>
"""

components.html(
    animation_html,
    height=350,
    scrolling=False
)

st.caption(
    f"Selected MNIST digit: {int(label)} | "
    f"Latent coordinates: "
    f"({z[0, 0].item():.3f}, {z[0, 1].item():.3f}) | "
    f"Reconstruction loss: {reconstruction_loss.item():.2f}"
)

# --------------------------------------------------
# 10. INTERACTIVE LATENT SPACE
# --------------------------------------------------
import numpy as np
import plotly.express as px
import pandas as pd

st.divider()
st.header("6. Interactive 2D Latent Space")

st.write(
    "Explore how the trained encoder maps handwritten digits "
    "into a two-dimensional latent representation."
)

@st.cache_data
def calculate_latent_space():
    # Use a subset to keep the visualization responsive.
    sample_count = min(2000, len(dataset))
    indices = np.random.default_rng(42).choice(
        len(dataset),
        size=sample_count,
        replace=False
    )

    images = torch.stack([
        dataset[int(i)][0] for i in indices
    ]).to(DEVICE)

    labels = [int(dataset[int(i)][1]) for i in indices]

    with torch.no_grad():
        _, means, _ = model.encode(images)

    return pd.DataFrame({
        "z1": means[:, 0].cpu().numpy(),
        "z2": means[:, 1].cpu().numpy(),
        "Digit": [str(x) for x in labels]
    })


latent_df = calculate_latent_space()

# Add the currently selected digit's actual coordinates.
selected_point = pd.DataFrame({
    "z1": [mu[0, 0].item()],
    "z2": [mu[0, 1].item()],
    "Digit": [str(int(label))]
})

fig = px.scatter(
    latent_df,
    x="z1",
    y="z2",
    color="Digit",
    title="MNIST Images in the VAE Latent Space",
    opacity=0.65,
    hover_data=["Digit"],
    color_discrete_sequence=px.colors.qualitative.G10
)

fig.add_scatter(
    x=selected_point["z1"],
    y=selected_point["z2"],
    mode="markers",
    name="Selected image",
    marker=dict(
        size=18,
        symbol="star",
        color="white",
        line=dict(color="#00D4FF", width=3)
    ),
    hovertemplate=(
        "Selected digit: " + str(int(label)) +
        "<br>z₁: %{x:.3f}" +
        "<br>z₂: %{y:.3f}<extra></extra>"
    )
)

fig.update_layout(
    template="plotly_dark",
    height=600,
    xaxis_title="Latent Coordinate z₁",
    yaxis_title="Latent Coordinate z₂",
    legend_title="Digit Class"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "The scatter plot uses encoder means (μ) for a reproducible "
    "representation. The star marks the selected image. "
    "Only 2,000 MNIST images are plotted for responsiveness."
)

# --------------------------------------------------
# 11. INTERACTIVE LATENT-SPACE DIGIT GENERATOR
# --------------------------------------------------
st.divider()
st.header("7. Interactive Digit Generator")

st.write(
    "Adjust the latent coordinates and observe how the "
    "trained VAE decoder generates a handwritten-style digit."
)

col1, col2 = st.columns(2)

with col1:
    z1 = st.slider(
        "Latent coordinate z₁",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1
    )

with col2:
    z2 = st.slider(
        "Latent coordinate z₂",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1
    )

# Construct a latent vector for the selected coordinates.
# The first two dimensions are controlled by the sliders.
latent_values = [z1, z2] + [0.0] * (model.latent_dim - 2)

latent_vector = torch.tensor(
    [latent_values],
    dtype=torch.float32,
    device=DEVICE
)
# Generate an image using the trained decoder.
with torch.no_grad():
    generated_image = model.decode(latent_vector)

generated_array = generated_image[0, 0].cpu().numpy()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Generated Digit")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(
        generated_array,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Generation Information")

    st.metric("Latent coordinate z₁", f"{z1:.1f}")
    st.metric("Latent coordinate z₂", f"{z2:.1f}")

    st.write("**Decoder output:** 1 × 28 × 28 pixels")
    st.write("**Pixel intensity range:** 0 to 1")

    st.info(
        "Move either slider to generate a different image. "
        "The decoder uses your trained VAE weights."
    )

# --------------------------------------------------
# 12. LIVE VAE TRAINING-LOSS DASHBOARD
# --------------------------------------------------
import os
import pandas as pd
import plotly.express as px

st.divider()
st.header("8. Live VAE Training Progress")

st.write(
    "Monitor total loss, reconstruction loss, and KL divergence "
    "as the training history is updated."
)

history_file = "vae_loss_history.csv"
model_file = "vae_mnist.pth"

# Configure automatic refresh.
auto_refresh = st.toggle(
    "Automatically refresh training graph",
    value=True,
    key="vae_auto_refresh"
)

refresh_seconds = st.selectbox(
    "Refresh interval",
    [2, 5, 10],
    index=1,
    key="vae_refresh_interval"
)

if auto_refresh:
    st.caption(
        f"Only this training dashboard refreshes every "
        f"{refresh_seconds} seconds."
    )
else:
    st.caption("Automatic refresh is disabled.")


# Refresh only this section, not the entire page.
@st.fragment(run_every=refresh_seconds if auto_refresh else None)
def render_training_dashboard():

    if not os.path.exists(history_file):
        st.info(
            "Training history not found. Run train_vae.py "
            "from your project folder to create it."
        )
        return

    try:
        history_df = pd.read_csv(history_file)

        if history_df.empty:
            st.info(
                "The training history file is empty. "
                "Start training to populate the graph."
            )
            return

        required_columns = [
            "Epoch",
            "Total Loss",
            "Reconstruction Loss",
            "KL Loss"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in history_df.columns
        ]

        if missing_columns:
            st.warning(
                f"Missing columns in training history: "
                f"{', '.join(missing_columns)}"
            )
            return

        latest_epoch = int(history_df["Epoch"].max())

        # Latest training status.
        col1, col2 = st.columns(2)

        col1.metric("Latest Epoch", f"{latest_epoch}/5")

        if os.path.exists(model_file):
            col2.success("A saved model file exists.")
        else:
            col2.info("Waiting for the trained model file.")

        # Plot training curves.
        fig_loss = px.line(
            history_df,
            x="Epoch",
            y=[
                "Total Loss",
                "Reconstruction Loss",
                "KL Loss"
            ],
            markers=True,
            title="VAE Training Loss Across Epochs",
            labels={
                "value": "Average Loss per Image",
                "variable": "Loss Component",
                "Epoch": "Training Epoch"
            }
        )

        fig_loss.update_layout(
            template="plotly_dark",
            height=500,
            hovermode="x unified",
            legend_title="Loss Component"
        )

        st.plotly_chart(
            fig_loss,
            use_container_width=True,
            key="vae_live_loss_chart"
        )

        # Training history table.
        st.subheader("Latest Training History")

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # Loss metrics.
        initial_loss = float(history_df["Total Loss"].iloc[0])
        final_loss = float(history_df["Total Loss"].iloc[-1])

        reduction = (
            (initial_loss - final_loss) / initial_loss * 100
            if initial_loss != 0
            else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Initial Total Loss",
            f"{initial_loss:.2f}"
        )

        col2.metric(
            "Latest Total Loss",
            f"{final_loss:.2f}"
        )

        col3.metric(
            "Loss Reduction",
            f"{reduction:.2f}%"
        )

        # Training status.
        if latest_epoch < 5:
            st.info(
                f"Training history currently contains "
                f"{latest_epoch} of 5 completed epochs."
            )
        else:
            st.success(
                f"Training history contains {latest_epoch} "
                f"recorded epochs."
            )

    except (
        pd.errors.EmptyDataError,
        KeyError,
        ValueError,
        OSError
    ) as error:
        st.warning(
            "The training history could not be read. "
            f"Please wait for the next refresh. Details: {error}"
        )


render_training_dashboard()
# --------------------------------------------------
# 13. VAE ARCHITECTURE VISUALIZATION
# --------------------------------------------------
st.divider()
st.success("DEBUG: Architecture section reached!")
st.header("9. VAE Neural Network Architecture")

st.write(
    "Explore the architecture of the trained Variational "
    "Autoencoder, from the input image to reconstruction."
)

architecture_html = """
<div style="
    background:#101827;
    color:#f8fafc;
    padding:24px;
    border-radius:16px;
    font-family:Arial,sans-serif;
    overflow-x:auto;
">
    <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
        min-width:850px;
        text-align:center;
    ">
        <div style="flex:1">
            <div style="background:#334155;padding:16px;border-radius:12px">
                <div style="font-size:28px">🖼️</div>
                <b>Input</b>
                <div>784</div>
            </div>
        </div>

        <div style="font-size:24px;color:#38bdf8">→</div>

        <div style="flex:1">
            <div style="background:#164e63;padding:16px;border-radius:12px">
                <div style="font-size:28px">🧠</div>
                <b>Encoder</b>
                <div>256 → 128</div>
            </div>
        </div>

        <div style="font-size:24px;color:#38bdf8">→</div>

        <div style="flex:1">
            <div style="background:#5b21b6;padding:16px;border-radius:12px">
                <div style="font-size:28px">🌌</div>
                <b>Latent Space</b>
                <div>μ, logvar</div>
                <div>z: 2 dimensions</div>
            </div>
        </div>

        <div style="font-size:24px;color:#38bdf8">→</div>

        <div style="flex:1">
            <div style="background:#14532d;padding:16px;border-radius:12px">
                <div style="font-size:28px">⚙️</div>
                <b>Decoder</b>
                <div>128 → 256</div>
            </div>
        </div>

        <div style="font-size:24px;color:#38bdf8">→</div>

        <div style="flex:1">
            <div style="background:#7c2d12;padding:16px;border-radius:12px">
                <div style="font-size:28px">✨</div>
                <b>Output</b>
                <div>784</div>
                <div>28 × 28</div>
            </div>
        </div>
    </div>

    <div style="
        margin-top:28px;
        padding:16px;
        background:#1e293b;
        border-radius:12px;
        text-align:center;
        line-height:1.8;
    ">
        <b>Training objective</b><br>
        Total Loss = Reconstruction Loss + KL Divergence
    </div>
</div>
"""

import streamlit.components.v1 as components

components.html(
    architecture_html,
    height=240,
    scrolling=True
)

st.caption(
    "Architecture shown: 784 → 256 → 128 → 2 → 128 → 256 → 784. "
    "The encoder also calculates μ and log-variance; "
    "the latent vector is sampled using reparameterization during training."
)

# --------------------------------------------------
# 14. VAE RECONSTRUCTION QUALITY ANALYZER
# --------------------------------------------------
import numpy as np
import plotly.express as px

st.divider()
st.header("10. VAE Reconstruction Quality Analyzer")

st.write(
    "Compare the original MNIST image with its reconstruction "
    "and visualize pixel-wise reconstruction errors."
)

# Convert tensors to NumPy arrays.
original_array = (
    input_image.detach().cpu().numpy()[0, 0]
)

reconstructed_array = (
    reconstruction.detach().cpu().numpy()[0, 0]
)

# Calculate absolute pixel-wise error.
absolute_error = np.abs(
    original_array - reconstructed_array
)

# Calculate mean squared error.
mse = float(
    np.mean((original_array - reconstructed_array) ** 2)
)

# Calculate mean absolute error.
mae = float(np.mean(absolute_error))

# Display metrics.
col1, col2, col3 = st.columns(3)

col1.metric("Mean Squared Error", f"{mse:.6f}")
col2.metric("Mean Absolute Error", f"{mae:.6f}")
col3.metric("Maximum Pixel Error", f"{absolute_error.max():.4f}")

# Display the three images.
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original")
    st.image(
        original_array,
        clamp=True,
        use_container_width=True
    )

with col2:
    st.subheader("Reconstructed")
    st.image(
        reconstructed_array,
        clamp=True,
        use_container_width=True
    )

with col3:
    st.subheader("Absolute Error")
    error_fig = px.imshow(
        absolute_error,
        color_continuous_scale="Inferno",
        aspect="equal",
        title="Pixel-wise Error Heatmap"
    )
    error_fig.update_layout(height=300)
    st.plotly_chart(
        error_fig,
        use_container_width=True,
        key="vae_reconstruction_error_heatmap"
    )

st.caption(
    "Lower MSE and MAE indicate smaller pixel-wise reconstruction "
    "errors. The heatmap highlights locations where the reconstruction "
    "differs from the original."
)

# --------------------------------------------------
# 14. DIGIT-RECOGNITION ACCURACY ANALYZER
# --------------------------------------------------
st.divider()
st.header("10. Digit Recognition Accuracy Analyzer")

st.write(
    "Evaluate how recognizable the VAE-reconstructed digits are "
    "using a separately trained neural-network classifier."
)


@st.cache_resource
def load_digit_classifier():
    classifier = DigitClassifier().to(DEVICE)

    classifier.load_state_dict(
        torch.load(
            "digit_classifier.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    classifier.eval()
    return classifier


@st.cache_data
def evaluate_reconstructions():
    classifier = load_digit_classifier()

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # Evaluate a manageable subset of the test dataset.
    evaluation_count = 1000

    original_images = []
    reconstructed_images = []
    true_labels = []

    with torch.no_grad():
        for index in range(evaluation_count):
            image, label = test_dataset[index]

            image_batch = image.unsqueeze(0).to(DEVICE)

            # Reconstruct using the trained VAE.
            vae_result = model(image_batch)
            reconstruction = vae_result["reconstruction"]

            original_images.append(image)
            reconstructed_images.append(
                reconstruction.squeeze(0).cpu()
            )
            true_labels.append(label)

    original_batch = torch.stack(original_images).to(DEVICE)
    reconstructed_batch = torch.stack(
        reconstructed_images
    ).to(DEVICE)

    labels_tensor = torch.tensor(true_labels)

    with torch.no_grad():
        original_predictions = classifier(
            original_batch
        ).argmax(dim=1).cpu()

        reconstructed_predictions = classifier(
            reconstructed_batch
        ).argmax(dim=1).cpu()

    original_accuracy = accuracy_score(
        labels_tensor,
        original_predictions
    )

    reconstructed_accuracy = accuracy_score(
        labels_tensor,
        reconstructed_predictions
    )

    return (
        labels_tensor.numpy(),
        original_predictions.numpy(),
        reconstructed_predictions.numpy(),
        original_accuracy,
        reconstructed_accuracy
    )


if st.button("Run Digit Recognition Analysis"):
    with st.spinner(
        "Evaluating 1,000 MNIST test images..."
    ):
        (
            true_labels,
            original_predictions,
            reconstructed_predictions,
            original_accuracy,
            reconstructed_accuracy
        ) = evaluate_reconstructions()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Original Image Accuracy",
        f"{original_accuracy * 100:.2f}%"
    )

    col2.metric(
        "Reconstructed Image Accuracy",
        f"{reconstructed_accuracy * 100:.2f}%"
    )

    col3.metric(
        "Recognition Accuracy Drop",
        f"{(original_accuracy - reconstructed_accuracy) * 100:.2f}%",
        help="Difference between original and reconstructed image accuracy."
    )

    st.subheader("Reconstruction Recognition Confusion Matrix")

    matrix = confusion_matrix(
        true_labels,
        reconstructed_predictions,
        labels=list(range(10))
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display = ax.imshow(
        matrix,
        cmap="Blues"
    )

    ax.set_title("True Digit vs. Predicted Digit")
    ax.set_xlabel("Predicted Digit")
    ax.set_ylabel("True Digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    for row in range(10):
        for col in range(10):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                fontsize=8
            )

    fig.colorbar(display, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")

    report = classification_report(
        true_labels,
        reconstructed_predictions,
        labels=list(range(10)),
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.caption(
        "Evaluation uses the first 1,000 images of the MNIST test set. "
        "The classifier is trained separately from the VAE. "
        "Results depend on the trained models and selected evaluation subset."
    )