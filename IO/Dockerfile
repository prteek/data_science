FROM python:3.8
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python3 -m pip install -Ur requirements.txt
COPY . .
CMD streamlit run app.py --theme.base light
