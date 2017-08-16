from bottle import route, run, template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn
import json
from io import StringIO
import data_helpers


tf.flags.DEFINE_string("checkpoint_dir", ".", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#tf.flags.DEFINE_string('gpudev','','Select a GPU Device.')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
#os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpudev
print("\nParameters:    ")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

@route('/classify', method="POST")
def index():
    postdata = request.body.read()
    print(postdata)
    x_raw=[]
    x_raw.append(str(postdata))
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
 #   with tf.device('/cpu:0'):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            #device_count={'GPU': 0},
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            batch_scores = sess.run(scores, {input_x: x_test, dropout_keep_prob: 1.0})
            batch_top3 = (-batch_scores).argsort(1)[:, :3]

    scoresdict = dict()
    i = 0
    for score in batch_scores[0]:
        scoresdict.update({str(data_helpers.inv_wipoareas[i]) : str(score)})
        i = i+1

    #return "length fo batch_scores "+str(len(batch_scores))+"\n"+str(batch_top3[0])+str(list(data_helpers.inv_wipoareas[k] for k in batch_top3[0]))+"\n"+strscores
    result = {}
    result['topClass'] = data_helpers.inv_wipoareas[batch_top3[0][0]]
    result['top3Classes'] = list(data_helpers.inv_wipoareas[k] for k in batch_top3[0])
    result['scores'] = scoresdict
    io = StringIO()
    json.dump(result,io)
    return io.getvalue()

run(host='0.0.0.0', port=8081,debug=True)

