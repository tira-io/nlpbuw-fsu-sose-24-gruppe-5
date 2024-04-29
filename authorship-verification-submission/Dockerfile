FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD authorship_verification_submission.py /code/authorship_verification_submission.py
ADD requirements.txt /code/requirements.txt

RUN pip3 install -r /code/requirements.txt

ENTRYPOINT [ "python3", "/code/authorship_verification_submission.py" ]
