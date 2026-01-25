import torch
import streamlit as st
import torchvision.transforms.v2 as T
from PIL import Image
import torch.nn.functional as F
from src.model import ResNetTransfer

st.set_page_config(page_title="Cat & Dog Classifier", layout="centered")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetTransfer(n_classes=2)
    checkpoint_path = "models/resnet_finetuned.pth"

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        st.error(f"Erro: Arquivo do modelo não encontrado em {checkpoint_path}")
        return None

    model.to(device)
    model.eval()

    return model, device


def process_image(image):
    transform = T.Compose([T.Resize((224, 225)), T.ToTensor()])

    return transform(image).unsqueeze(0)


model_data = load_model()

with st.container(border=True):

    st.markdown("## 🐶 Classificador de Cães e Gatos 🐱", text_alignment="center")

    if model_data is not None:
        model, device = model_data
        uploaded_file = st.file_uploader(
            "Escolha uma imagem...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB").resize((650, 650))

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.image(image, caption="Imagem Carregada", width="content")

                col1, col2_button, col3 = st.columns([1, 5, 1])
                
                with col2_button:
                    if st.button("Classifier", width=250):
                        with st.spinner("Processando..."):
                            input_tensor = process_image(image).to(device)

                            with torch.no_grad():
                                output = model(input_tensor)
                                probabilities = F.softmax(output, dim=1)

                                top_p, top_class = probabilities.topk(1, dim=1)

                                score = top_p.item()
                                class_idx = top_class.item()

                                labels = {0: "Gato", 1: "Cachorro"}
                                prediction = labels.get(class_idx, "Desconhecido")

                        st.markdown("## Resultado", text_alignment="center")
                        st.success(body= f"**{prediction}**")

