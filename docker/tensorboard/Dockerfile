FROM python:3.6

WORKDIR /usr/src/tensorboard

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# docker run -d -v $(pwd)/log:/tensorboard -p 6006:6006 --name tensorboard tensorboard tensorboard --logdir=/tensorboard/<log_folder>