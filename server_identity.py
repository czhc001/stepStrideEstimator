from http.server import HTTPServer, BaseHTTPRequestHandler
import io, shutil, json, time, socketserver, threading, socket

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import random
import sys

sess = tf.Session()
with gfile.FastGFile('Models/model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图

# 需要有一个初始化的过程
sess.run(tf.global_variables_initializer())

# 输入
X = sess.graph.get_tensor_by_name('X:0')
P = sess.graph.get_tensor_by_name('P:0')
output = sess.graph.get_tensor_by_name('output/BiasAdd:0')


user_list = ['zc', 'wjw', 'wz']
passwords_list = ['19941020', '12345678', '87654321']
dict_lastT = {}
dict_challenge = {}
dict_condition = {} # 0:外部 1:认证1 2:认证2 3:内部 -1:不通过
current_user_enterT = 0
current_user_passT = 0
current_user_leaveT = 0
current_user = ''
current_challenge = 1
busy = False
last_t = 0

reg_result_c = False
step_stride_c = 1
leave_allow_c = True
location_c = ''


class MyThreadingHTTPServer1(socketserver.ThreadingMixIn, HTTPServer):
    address_family = socket.AF_INET6
    pass


class HTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    pass


class MyHttpHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global current_user
        global reg_result_c
        global step_stride_c
        global leave_allow_c
        global location_c
        # print(threading.current_thread().name)
        length = int(self.headers['Content-Length'])
        readdata = self.rfile.read(length).decode('utf-8')
        if readdata == 'B':

            sys.stderr.write("%s - - [%s]\n" %
                             ('aaaa:bbbb:cccc:dddd:b6e6:2dff:fe34:a56d',
                              self.log_date_time_string(),))
            # print('BOARD' + current_user)
            if current_user != '' and dict_challenge.__contains__(current_user):
                challenge = dict_challenge[current_user]
                # print(challenge)
                if challenge == 1:
                    self.send_response(205)
                    # print(205)
                elif challenge == 2:
                    self.send_response(206)
                    # print(206)
                else:
                    self.send_response(200)
                    # print(200)
            else:
                self.send_response(100)
                # print(100)
            self.end_headers()
        else:
            post_data = json.loads(readdata)
            op = post_data['Operation']
            # print(op)

            if op == 'C':
                content = post_data['content']
                # print(content)
                if content == 'id1':
                    reg_result_c = True
                elif content == 'id0':
                    reg_result_c = False
                elif content == 'st0':
                    step_stride_c = 1
                elif content == 'st1':
                    step_stride_c = 2
                elif content == 'le0':
                    leave_allow_c = False
                elif content == 'le1':
                    leave_allow_c = True
                elif content == 'lc1':
                    location_c = '测试点1'
                elif content == 'lc0':
                    location_c = ''
                self.send_response(250)
                self.end_headers()
            elif op == 'login':
                sys.stderr.write("%s - - [%s]\n" %
                                 (self.address_string(),
                                  self.log_date_time_string(),))
                username = post_data['Username']
                passwords = post_data['Passwords']
                print('login: ' + username + ' ' + passwords)
                uid = -1
                for i in range(user_list.__len__()):
                    if user_list[i] == username:
                        uid = i
                login_result = False
                if uid >= 0 and passwords_list[uid] == passwords:
                    login_result = True
                else:
                    login_result = False
                data = json.dumps({'loginresult': str(login_result)})
                enc = "UTF-8"
                encoded = ''.join(data).encode(enc)
                print(encoded)
                f0 = io.BytesIO()
                f0.write(encoded)
                f0.seek(0)
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=%s" % enc)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                shutil.copyfileobj(f0, self.wfile)

            else:
                sys.stderr.write("%s - - [%s]\n" %
                                 (self.address_string(),
                                  self.log_date_time_string(),))
                username = post_data['User']
                print(username)
                if current_user == '':
                    current_user = username
                # print(op + ' ' + username + ' ' + current_user)
                if op == 'wifi':
                    wifi_list_str = post_data['Content']
                    print(wifi_list_str)
                    data = json.dumps({'inloc': leave_allow_c})
                    enc = "UTF-8"
                    encoded = ''.join(data).encode(enc)
                    print(encoded)
                    f0 = io.BytesIO()
                    f0.write(encoded)
                    f0.seek(0)
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=%s" % enc)
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    shutil.copyfileobj(f0, self.wfile)
                elif op == 'ble':
                    ble_list_str = post_data['Content']
                    ble_list = ble_list_str.split(';')
                    ble_count = ble_list.__len__()
                    print(ble_count)
                    print(ble_list_str)
                    data = json.dumps({'location':location_c})
                    enc = "UTF-8"
                    encoded = ''.join(data).encode(enc)
                    # print(encoded)
                    f0 = io.BytesIO()
                    f0.write(encoded)
                    f0.seek(0)
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=%s" % enc)
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    shutil.copyfileobj(f0, self.wfile)
                elif op == 'reg_start':
                    condition = 0  # 0:外部  1:认证1  2:认证2  3:内部  -1:不通过
                    challenge = random.randint(1, 2)
                    dict_challenge[username] = challenge
                    data = json.dumps({'challenge': str(dict_challenge[username])})
                    enc = "UTF-8"
                    encoded = ''.join(data).encode(enc)
                    f0 = io.BytesIO()
                    f0.write(encoded)
                    f0.seek(0)
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=%s" % enc)
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    shutil.copyfileobj(f0, self.wfile)
                elif op == 'reg':
                    condition = 0  # 0:外部  1:身份认证  2:步长认证    -1:不通过
                    if dict_condition.__contains__(username):   #查找此用户condition
                        condition = dict_condition[username]
                    content = post_data['Content'].split(',')
                    print(content)

                    if reg_result_c:
                        condition = 1
                    else:
                        condition = -1
                        print(username + ' ' + '身份验证失败')
                    # print('stride ' + str(step_stride_c) + " " + str(dict_challenge[username]))
                    if condition == 1 and step_stride_c == dict_challenge[username]:
                        condition = 2
                        print(username + ' ' + '验证成功')
                    elif condition == 1:
                        condition = -1
                        print(username + ' ' + '步长验证失败 ' + str(step_stride_c) + ' ' + str(dict_challenge[username]))
                    data = json.dumps({'result': str(condition)})
                    enc = "UTF-8"
                    encoded = ''.join(data).encode(enc)
                    f0 = io.BytesIO()
                    f0.write(encoded)
                    f0.seek(0)
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=%s" % enc)
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    shutil.copyfileobj(f0, self.wfile)
                    condition = 0
                    dict_challenge[username] = 0
                    dict_condition[username] = condition

                elif op == 'leave':
                    condition = 0  # 0 外部 2 内部 -1 失败
                    if dict_condition.__contains__(username):
                        condition = dict_condition[username]
                    data = json.dumps({'result': 'leaved'})
                    enc = "UTF-8"
                    encoded = ''.join(data).encode(enc)
                    f0 = io.BytesIO()
                    f0.write(encoded)
                    f0.seek(0)
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=%s" % enc)
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    shutil.copyfileobj(f0, self.wfile)
                    dict_condition[username] = 0
                    if current_user == username:
                        current_user = ''


def check_current_user():
    global dict_lastT
    global dict_challenge
    global lock
    global current_user
    global current_challenge
    while True:
        if int(round(time.time() * 1000)) - last_t > 2000:
            current_user = ''
        time.sleep(0.1)


# httpd = HTTPServer(('', 9601), MyHttpHandler)
httpd = MyThreadingHTTPServer1(('aaaa:bbbb:cccc:dddd:389f:3443:f457:aa5', 9601), MyHttpHandler)
httpd0 = HTTPServer(('', 9600), MyHttpHandler)
print("Server started on 127.0.0.1,port 9601......")


def s1():
    httpd.serve_forever()


def s0():
    httpd0.serve_forever()


def user_check():

    pass


t1 = threading.Thread(target=s1)
t1.setDaemon(True)
t0 = threading.Thread(target=s0)
t0.setDaemon(True)
t1.start()
t0.start()
while True:
    pass

