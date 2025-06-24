# Handwritten Figit Generator

# Projeto organizado por pastas:
#
# /src         - Código-fonte principal (modelos, lógica de geração)
# /notebooks   - Notebooks de experimentos e documentação interativa
# /models      - Modelos treinados e checkpoints

# /app         - Scripts principais de execução
# /tests       - Testes automatizados
# /docs        - Documentação
#
# Arquivos principais:
# - streamlit_app.py: Interface principal do app (pode ser movido para /app)
# - src/digit_generator.py: Backend de geração de dígitos
# - models/generator_final.pth: Modelo treinado
# - notebooks/Handwritten_Digit_Generator.ipynb: Notebook de treinamento
# - requirements.txt: Dependências do projeto
# - README.md: Documentação principal

This project is a handwritten digit generator based on generative neural networks (GANs), trained with the MNIST dataset. It allows you to generate synthetic images of digits (0-9) similar to real handwritten digits.

## Access the Web App
You can access the web application directly at:
https://handwritten-digit-talesoliveira.streamlit.app

## Local Usage Instructions
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate
   ```

   ```powershell
   pip install -r requirements.txt
   ```

   ```powershell
   streamlit run streamlit_app.py
   ```
