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


user_list = ['zc', 'wjw']
dict_lastT = {}
dict_challenge = {}
dict_condition = {} # 0:外部 1:认证1 2:认证2 3:内部 -1:不通过
lock = threading.Lock()
current_user_enterT = 0
current_user_passT = 0
current_user_leaveT = 0
current_user = ''
current_challenge = 1
busy = False
last_t = 0

pass_condition = 1


class MyThreadingHTTPServer1(socketserver.ThreadingMixIn, HTTPServer):
    address_family = socket.AF_INET6
    pass


class HTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    pass


class MyHttpHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global current_user
        # print(threading.current_thread().name)
        length = int(self.headers['Content-Length'])
        readdata = self.rfile.read(length).decode('utf-8')
        if readdata == 'B':

            sys.stderr.write("%s - - [%s]\n" %
                             ('aaaa:bbbb:cccc:dddd:b6e6:2dff:fe34:a56d',
                              self.log_date_time_string(),))
            print('BOARD' + current_user)
            if current_user != '' and dict_challenge.__contains__(current_user):
                challenge = dict_challenge[current_user]
                # print(challenge)
                if challenge == 1:
                    self.send_response(205)
                    print(205)
                elif challenge == 2:
                    self.send_response(206)
                    print(206)
                else:
                    self.send_response(200)
                    print(200)
            else:
                self.send_response(100)
                print(100)
            self.end_headers()
        else:
            post_data = json.loads(readdata)
            op = post_data['Operation']

            if op == 'C':
                content = post_data['content']
                print(content)
                self.send_response(205)
                self.end_headers()
            else:
                sys.stderr.write("%s - - [%s]\n" %
                                 (self.address_string(),
                                  self.log_date_time_string(),))
                username = post_data['User']
                print(username)
                if current_user == '':
                    current_user = username
                # print(op + ' ' + username + ' ' + current_user)
                if op == 'ble':
                    ble_list_str = post_data['Content']
                    ble_list = ble_list_str.split(';')
                    ble_count = ble_list.__len__()
                    print(ble_count)
                elif op == 'reg_start':
                    condition = 0  # 0:外部  1:认证1  2:认证2  3:内部  -1:不通过

                    challenge = random.randint(1, 2)
                    dict_challenge[username] = challenge
                elif op == 'reg':
                    condition = 0  # 0:外部  1:认证1  2:认证2  3:内部  -1:不通过
                    if dict_condition.__contains__(username):   #查找此用户condition
                        condition = dict_condition[username]
                    if pass_condition > condition >= 0:         #认证通过次数不够，继续
                        content = post_data['Content'].split(',')
                        feature = np.array(content, dtype=np.float)
                        lock.acquire()
                        r = sess.run(output, feed_dict={X: [feature], P: 1.0})
                        lock.release()
                        index = 0
                        if r[0][0] < r[0][1]:
                            index = 1
                        __name = user_list[index]
                        reg_state = condition + 1
                        if reg_state > pass_condition:
                            reg_state = pass_condition

                        data = json.dumps({'name': __name,
                                           'result': 'passed' + str(reg_state),
                                           'challenge': str(dict_challenge[username])})
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
                        dict_condition[username] = condition + 1
                    elif condition == pass_condition:
                        data = json.dumps({'name': username, 'result': 'complete'})
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
                        current_user = ''
                elif op == 'leave':
                    condition = 0  # 0 外部 1 认证中 2 内部
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


t1 = threading.Thread(target=s1)
t1.setDaemon(True)
t0 = threading.Thread(target=s0)
t0.setDaemon(True)
t1.start()
t0.start()
while True:
    pass

