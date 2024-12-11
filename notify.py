import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
import sys
import json

def send(arg1, arg2):
    with open("./certification.json", "r") as f:
        json_data = json.load(f)
    recipients = json_data["target_email"]
    message = MIMEMultipart()
    message['Subject'] = "실험 종료"
    message['From'] = json_data["sender_email"]
    email_id = json_data["email_id"]
    email_pw = json_data["email_pw"]

    content = """
        <html>
        <body>
            <h2>{title}</h2>
            <p>{etc}</p>
        </body>
        </html>
    """.format(
    title = f"{arg1} 실험 종료",
    etc = f"실험 끝났다."
    )

    mimetext = MIMEText(content,'html')
    message.attach(mimetext)
    server = smtplib.SMTP('smtp.naver.com',587)
    server.ehlo()
    server.starttls()
    server.login(email_id,email_pw)
    server.sendmail(message['From'],recipients,message.as_string())
    server.quit()

if __name__ == '__main__':
    send(sys.argv[1], sys.argv[2])