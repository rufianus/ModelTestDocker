FROM python:3.7

WORKDIR /app

COPY . /app

RUN pip3 install pillow
RUN pip3 install numpy
RUN pip3 install tensorflow
RUN pip3 install flask 
RUN pip3 install keras
RUN pip3 install flask_cors

EXPOSE 5000

CMD ["python3", "imageregserver.py"]
