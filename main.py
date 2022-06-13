from flask import Flask
import app.views as views
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# url
app.add_url_rule('/base','base',views.base)
app.add_url_rule('/','index',views.index)
app.add_url_rule('/stressapp','stressapp',views.stressapp)
app.add_url_rule('/stressapp/stress','stress',views.stress,methods=['GET','POST'])
app.add_url_rule('/stressapp/stress/<filename>/<status>','video',views.video,methods=['GET','POST'])

# run
if __name__ == "__main__":
    app.run()