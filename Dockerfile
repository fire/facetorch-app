FROM python:3.9.12-slim

WORKDIR /code

COPY ./requirements.txt $WORKDIR/requirements.txt
COPY ./config.merged.yml $WORKDIR/config.merged.yml

RUN pip install gradio --no-cache-dir
RUN pip install --no-cache-dir --upgrade -r $WORKDIR/requirements.txt

COPY ./app.py $WORKDIR/app.py

EXPOSE 7860

CMD ["python", "app.py"]
