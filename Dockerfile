FROM ubuntu:16.04
MAINTAINER Anastasia Haswani <https://github.com/nastazya/final_project>

RUN apt-get update

RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib plotly sklearn

RUN cat /etc/lsb-release

ENTRYPOINT ["python3","./analyse.py"]
CMD ["breast_cancer"]




