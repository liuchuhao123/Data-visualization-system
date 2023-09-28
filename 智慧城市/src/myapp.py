import sqlite3
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
infomation = []


# 查询地铁线路
def search_line(sstr):
    global infomation
    infomation.clear()  # 清空结果列表
    result = sstr.split(",")

    sql = "select city,line,name from info where city=? and line=?"
    cursor = conn.cursor()
    cursor.execute(sql, (result[0], result[1]))

    for row in cursor.fetchall():
        infomation.append(f"{row[0]}  {row[1]}  {row[2]}")


# 查询站点
def search_station(sstr):
    global infomation
    infomation.clear()  # 清空结果列表
    sql = "select city,line,name from info where name=?"
    cursor = conn.cursor()
    cursor.execute(sql, (sstr,))
    for row in cursor.fetchall():
        infomation.append(f"{row[0]}  {row[1]}  {row[2]}")


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        search_text = request.form['search_text']
        if "search_station" in request.form:
            search_station(search_text)
        else:
            search_line(search_text)
    return render_template('index.html', infomation=infomation)


@app.route('/search_station', methods=['POST'])
def search_station_route():
    if request.method == 'POST':
        search_text = request.form['search_text']
        search_station(search_text)
    return render_template('index.html', infomation=infomation)


@app.route('/show_images')
def show_images():
    return render_template('images.html')


@app.route('/back_to_home')
def back_to_home():
    return redirect(url_for('home'))


# 添加路由函数，用于打开其他HTML页面
@app.route('/open_html/中国地铁站最爱用的字')
def open_中国地铁站最爱用的字():
    return render_template('中国地铁站最爱用的字.html')


@app.route('/open_html/各城市地铁数量分布')
def open_各地铁城市数量分布():
    return render_template('各城市地铁数量分布.html')


@app.route('/open_html/地铁站最爱用门命名的城市')
def open_地铁站最爱用门命名的城市():
    return render_template('地铁站最爱用门命名的城市.html')


@app.route('/open_html/已开通地铁城市分布情况')
def open_已开通地铁城市分布情况():
    return render_template('已开通地铁城市分布情况.html')


@app.route('/images')
def images():
    return render_template('images.html')


if __name__ == '__main__':
    conn = sqlite3.connect('C:/Users/72466/Desktop/智慧城市/city_line_new.db', check_same_thread=False)
    app.run()
