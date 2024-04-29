FROM python:3.12.1

WORKDIR /app

COPY poetry.lock pyproject.toml /app/

RUN pip install poetry

RUN poetry install --no-dev

COPY . /app

EXPOSE 8501

CMD ["poetry", "run", "python","-m" ,"streamlit", "run", "src/streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
