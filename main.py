from flask import Flask
from app import views
app = Flask(__name__)

# url
app.add_url_rule('/base','base',views.base)
app.add_url_rule('/','index',views.index)
app.add_url_rule('/stressapp','stressapp',views.stressapp)
app.add_url_rule('/stressapp/stress','stress',views.stress,methods=['GET','POST'])
app.add_url_rule('/stressapp/stress/<filename>/<status>','video',views.video,methods=['GET','POST'])

# run
if __name__ == "__main__":
    app.run()