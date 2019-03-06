FROM python:3.6

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ENTRYPOINT ["python", "-u", "./main.py", "-m", "1", "-a", "0,1,2,3"]

CMD ["python", "-u", "./main.py", "-m", "1", "-a", "0,1,2,3", "&&", "python", "./main.py"]
