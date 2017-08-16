FROM tensorflow/tensorflow:1.3.0-rc2-py3
USER root

COPY classifier /usr/local/classifier/

ADD startService.sh /usr/local/

RUN chmod 0777 /usr/local/startService.sh

EXPOSE 8081 

CMD ["/bin/bash", "-c", "/usr/local/startService.sh"]
