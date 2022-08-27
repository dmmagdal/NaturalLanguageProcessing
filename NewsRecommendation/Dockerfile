# Docker file to run a container that will run the bert_faiss.py
# in Python 3 for PyTorch 1.10 (no GPU).

# Load pytorch image for pytorch 1.10 and Python 3.
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /bert-faiss
WORKDIR /bert-faiss
ADD . /bert-faiss

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y unzip

# Download and extract the dataset.
RUN curl --output "mindlarge_train.zip" https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip
RUN unzip mindlarge_train.zip -d ./
#RUN curl --output "mindlarge_valid.zip" https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip
#RUN unzip mindlarge_valid.zip -d ./
#RUN curl --output "mindlarge_test.zip" https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
#RUN unzip mindlarge_test.zip -d ./

#RUN curl --output "mindsmall_train.zip" https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip
#RUN unzip mindsmall_train.zip -d ./
#RUN curl --output "mindsmall_valid.zip" https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip
#RUN unzip mindsmall_valid.zip -d ./

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Run the bert_faiss.py program.
CMD ["python3", "bert_faiss.py"]