# handwritten-digit-generator

This project is a handwritten digit generator based on generative neural networks (GANs), trained with the MNIST dataset. It allows you to generate synthetic images of digits (0-9) similar to real handwritten digits.

## Access the Web App
You can access the web application directly at:
https://handwritten-digit-talesoliveira.streamlit.app

## Local Usage Instructions (in Portuguese)

### Como rodar localmente
1. Crie um ambiente virtual Python:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate
   ```
2. Instale as dependências:
   ```powershell
   pip install -r requirements.txt
   ```
3. Execute o app:
   ```powershell
   streamlit run streamlit_app.py
   ```

O app permite escolher um dígito e gerar 5 imagens sintéticas desse dígito, exibidas em uma grade.
