FROM python:3.6

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "./main.py"]

# docker run -v $(pwd)/data:/usr/src/app/data -v $(pwd)/log:/usr/src/app/log -v $(pwd)/model:/usr/src/app/model --name cnn-app traffic-sign-recognition python main.py <args>
