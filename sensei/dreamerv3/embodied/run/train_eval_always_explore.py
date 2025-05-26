import re

import embodied
import numpy as np

from embodied.core.video_utils import nums2vid
import time

import os  # CLUSTER UTILS
received_timeout_signal = False # CLUSTER_UTILS
def timeout_signal_handler(sig, frame):  # CLUSTER_UTILS
  global received_timeout_signal  # CLUSTER_UTILS
  print("Received timeout signal")  # CLUSTER_UTILS
  received_timeout_signal = True  # CLUSTER_UTILS

def train_eval_always_explore(
    agent, train_env, eval_env, train_replay, eval_replay, logger, cleanup, cluster, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Every(args.log_every)
  should_log_trainpolicy = embodied.when.Every(args.log_every) if args.video_every < 0 else embodied.when.Every(args.video_every)
  should_report = embodied.when.Every(args.report_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  should_sync = embodied.when.Every(args.sync_every)

  if cluster == 'mpi':
    from cluster import exit_for_resume  # CLUSTER_UTILS
    should_restart = embodied.when.Every(args.restart_every, initial=False)  # CLUSTER_UTILS
  elif cluster == 'uni':
    from cluster_utils import exit_for_resume  # CLUSTER_UTILS
    import signal  # CLUSTER_UTILS
    signal.signal(signal.SIGUSR1, timeout_signal_handler)  # CLUSTER_UTILS
    should_restart = lambda _: received_timeout_signal # CLUSTER_UTILS
  else: # CLUSTER_UTILS
    assert cluster == 'local'
    import sys # CLUSTER_UTILS
    exit_for_resume = lambda : sys.exit() # CLUSTER_UTILS
    should_restart = embodied.when.Every(args.restart_every, initial=False)  # CLUSTER_UTILS

  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()

  def per_episode(ep, step, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({
      'length': length, 'score': score,
      'reward_rate': (ep['reward'] - ep['reward'].min() >= 0.1).mean(),
    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    log_video = (mode == 'eval')
    if mode == 'train':
      log_video = should_log_trainpolicy(step)
    for key in args.log_keys_video:
      if key in ep:
        if log_video:
          vid = ep[key]
          # we can visualize sequences of numbers and videos
          if len(vid.shape) == 2 and vid.shape[-1] == 1:
            vid = nums2vid(vid)
          elif len(vid.shape) == 1:
            vid = nums2vid(np.expand_dims(vid, -1))
          else:
            assert len(vid.shape) == 4, f"Unexpected video shape {vid.shape}"
          stats[f'policy_{key}'] = vid
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix=f'{mode}_stats')

  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, step, mode='train'))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(train_replay.add)
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, step, mode='eval'))

  random_agent = embodied.RandomAgent(train_env.act_space)
  print('Prefill train dataset.')
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    driver_train(random_agent.policy, steps=100)
  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset_train = agent.dataset(train_replay.dataset)
  dataset_eval = agent.dataset(eval_replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        train_replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      logger.add(metrics.result())
      if should_report(step):
        logger.add(agent.report(batch[0]), prefix='report')
        with timer.scope('dataset_eval'):
          eval_batch = next(dataset_eval)
        logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(eval_replay.stats, prefix='eval_replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
    if should_restart(step):  # CLUSTER_UTILS
      checkpoint.save()  # CLUSTER_UTILS
      logger.write()  # CLUSTER_UTILS
      logger.write()   # CLUSTER_UTILS
      for obj in cleanup:  # CLUSTER_UTILS
        obj.close()  # CLUSTER_UTILS
      exit_for_resume()  # CLUSTER_UTILS

  driver_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  if args.pretrain_only_wm:
    time.sleep(30)  # wait for checkpoint to be written
    prev_agent = checkpoint.load()['agent']
    replace_agnt_state = {k: v for k, v in prev_agent.items() if 'expl_behavior' in k}
    for k,v in replace_agnt_state.items():  # debugging
      print(f"Replacing agent module {k}", v)
  should_save(step)  # Register that we just saved.

  if args.pretrain and not os.path.exists(logdir / 'pretrain.log'):  # CLUSTER_UTILS
    print("Start pretraining")
    for i in range(args.pretrain):
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train)
        outs, state[0], mets = agent.train(batch[0], state[0])
    agent.sync()
    if args.pretrain_only_wm: # overwrite weights of policy in agent
      checkpoint.save()
      should_save(step)  # register that we just saved.
      time.sleep(30)  # wait for checkpoint to be written
      for k, v in replace_agnt_state.items():  # debugging
        print(f"Replacing agent module {k}", v)
      checkpoint.load(replace_key='agent', replace_state=replace_agnt_state)
      checkpoint.save()
      should_save(step)  # register that we just saved.
      time.sleep(30)  # wait for checkpoint to be written
      agent.sync()

    dummyfile = open(os.path.join(logdir, 'pretrain.log'), 'w') # CLUSTER_UTILS
    dummyfile.close() # CLUSTER_UTILS

  print('Start training loop.')
  policy_train = lambda *args: agent.policy(*args, mode='explore')
  policy_eval = lambda *args: agent.policy(*args, mode='explore')
  while step < args.steps:
    if should_eval(step):
      print('Starting evaluation at step', int(step))
      driver_eval.reset()
      driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
    driver_train(policy_train, steps=100)
    if should_save(step):
      checkpoint.save()
  checkpoint.save()
  logger.write()
  logger.write()
