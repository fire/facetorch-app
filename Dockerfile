FROM python:3.9.12-slim

WORKDIR /code

COPY requirements.txt $WORKDIR/requirements.txt

RUN pip install gradio --no-cache-dir
RUN pip install --no-cache-dir --upgrade -r $WORKDIR/requirements.txt

COPY . .

EXPOSE 7860

ENTRYPOINT ["python", "app.py"]
