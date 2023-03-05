FROM python:3.9.12-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/code
#RUN chown -R user:user $WORKDIR
#RUN chmod 777 $WORKDIR

COPY --chown=user requirements.txt $WORKDIR/requirements.txt

RUN pip install gradio --no-cache-dir
RUN pip install --no-cache-dir --upgrade -r $WORKDIR/requirements.txt

COPY --chown=user . .

EXPOSE 7860

ENTRYPOINT ["python", "app.py"]
