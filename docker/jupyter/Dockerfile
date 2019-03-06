FROM python:3.6

WORKDIR /usr/src/jupyter

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -ms /bin/bash jupyter

USER jupyter

CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--notebook-dir=/jupyter"]

# docker run -d -v $(pwd)/:/jupyter -p 8888:8888 --name jupyter jupyter