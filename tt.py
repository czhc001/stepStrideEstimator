from http.server import HTTPServer, BaseHTTPRequestHandler
import io, shutil, json, time, socketserver, threading

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np

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
op = sess.graph.get_tensor_by_name('output/BiasAdd:0')


class MyThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    pass


class MyHttpHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        readdata = self.rfile.read(length).decode('utf-8')
        post_data = json.loads(readdata)
        cur_thread = threading.current_thread()
        time.sleep(5)
        print(1)
        print(cur_thread.name)
        print(post_data['Content'])
        content = post_data['Content'].split(',')
        feature = np.array(content, dtype=np.float)
        print(feature.shape)

        r = sess.run(op, feed_dict={
            X: [feature],
            P: 1.0})
        print(r)
        __name = 'zc'
        if r[0][0] < r[0][1]:
            __name = 'wjw'
        data = json.dumps({'name': __name})
        enc = "UTF-8"
        encoded = ''.join(data).encode(enc)
        f = io.BytesIO()
        f.write(encoded)
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=%s" % enc)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        shutil.copyfileobj(f, self.wfile)


httpd = MyThreadingHTTPServer(('', 9601), MyHttpHandler)
print("Server started on 127.0.0.1,port 9601......")
httpd.serve_forever() 
