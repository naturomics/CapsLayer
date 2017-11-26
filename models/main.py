import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append('..')
sys.path.append('.')

from config import cfg
from capslayer.utils import load_data


def save_to():
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        loss = cfg.results + '/loss.csv'
        train_acc = cfg.results + '/train_acc.csv'
        val_acc = cfg.results + '/val_acc.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return(fd_train_acc, fd_loss, fd_val_acc)
    else:
        test_acc = cfg.results + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('step,test_acc\n')


def train(model, supervisor):
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(cfg.dataset, cfg.batch_size, is_training=True)
    Y = valY[:num_val_batch * cfg.batch_size].reshape((-1, 1))

    if cfg.dataset == 'mnist' or cfg.dataset == 'fashion-mnist':
        num_label = 10
    elif cfg.dataset == 'smallNORB':
        num_label = 5
    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with supervisor.managed_session(config=config) as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print('Training for epoch ' + str(epoch) + '/' + str(cfg.epoch) + ':')
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.loss, model.accuracy, model.train_summary])
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op)

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_acc = 0
                    prob = np.zeros((num_val_batch * cfg.batch_size, num_label))
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        acc, prob[start:end, :] = sess.run([model.accuracy, model.activation], {model.X: valX[start:end], model.labels: valY[start:end]})
                        val_acc += acc
                    val_acc = val_acc / (cfg.batch_size * num_val_batch)
                    np.savetxt(cfg.results + '/activations_step_' + str(global_step) + '.txt', np.hstack((prob, Y)), fmt='%1.2f')
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()


def evaluation(model, supervisor):
    pass


def main(_):
    if cfg.dataset == 'mnist' or cfg.dataset == 'fashion-mnist':
        tf.logging.info(' Loading Graph...')
        model = CapsNet(height=28, width=28, channels=1, num_label=10)
    elif cfg.dataset == 'smallNORB':
        model = CapsNet(height=32, width=32, channaels=3, num_label=5)
    tf.logging.info(' Graph loaded')

    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

    if cfg.is_training:
        tf.logging.info(' Start trainging...')
        train(model, sv)
        tf.logging.info('Training done')
    else:
        evaluation(model, sv)

if __name__ == "__main__":
    model = 'vectorCapsNet'
    if model == 'vectorCapsNet':
        from vectorCapsNet import CapsNet
    elif model == 'matrixCapsNet':
        from matrixCapsNet import CapsNet
    else:
        raise Exception('Unsupported model, please check the name of model:', model)
    tf.app.run()
