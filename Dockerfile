FROM python:3.9.12-slim

RUN useradd -ms /bin/bash admin

WORKDIR /code
RUN chown -R admin:admin $WORKDIR
RUN chmod 755 $WORKDIR

COPY requirements.txt $WORKDIR/requirements.txt

RUN pip install gradio --no-cache-dir
RUN pip install --no-cache-dir --upgrade -r $WORKDIR/requirements.txt

COPY . .

USER admin

EXPOSE 7860

ENTRYPOINT ["python", "app.py"]
