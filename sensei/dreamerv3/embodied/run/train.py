import re

import embodied
import numpy as np

received_timeout_signal = False # CLUSTER_UTILS
def timeout_signal_handler(sig, frame):  # CLUSTER_UTILS
  global received_timeout_signal  # CLUSTER_UTILS
  print("Received timeout signal")  # CLUSTER_UTILS
  received_timeout_signal = True  # CLUSTER_UTILS

def train(agent, env, replay, logger, cleanup, cluster, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Every(args.log_every)
  should_log_trainpolicy = embodied.when.Every(args.log_every) # TODO Hardcoded, log trainpolicy every XYZ
  should_report = embodied.when.Every(args.report_every)
  should_save = embodied.when.Clock(args.save_every)
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
  print('Observation space:', embodied.format(env.obs_space), sep='\n')
  print('Action space:', embodied.format(env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])
  timer.wrap('replay', replay, ['add', 'save'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  def per_episode(ep, step):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    sum_abs_reward = float(np.abs(ep['reward']).astype(np.float64).sum())
    logger.add({
        'length': length,
        'score': score,
        'sum_abs_reward': sum_abs_reward,
        'reward_rate': (np.abs(ep['reward']) >= 0.5).mean(),
    }, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        if should_log_trainpolicy(step):
          stats[f'policy_{key}'] = ep[key]
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
    metrics.add(stats, prefix='stats')

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep, step))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(replay.add)

  print('Prefill train dataset.')
  random_agent = embodied.RandomAgent(env.act_space)
  while len(replay) < max(args.batch_steps, args.train_fill):
    driver(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset = agent.dataset(replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset'):
        batch[0] = next(dataset)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      agg = metrics.result()
      report = agent.report(batch[0])
      report = {k: v for k, v in report.items() if 'train/' + k not in agg}
      logger.add(agg)
      if should_report(step):
        logger.add(report, prefix='report')
        logger.add(replay.stats, prefix='replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
    if should_restart(step):  # CLUSTER_UTILS
      checkpoint.save()  # CLUSTER_UTILS
      logger.write()  # CLUSTER_UTILS
      logger.write()   # CLUSTER_UTILS
      for obj in cleanup:  # CLUSTER_UTILS
        obj.close()  # CLUSTER_UTILS
      exit_for_resume()  # CLUSTER_UTILS
  driver.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  timer.wrap('checkpoint', checkpoint, ['save', 'load'])
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.replay = replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  while step < args.steps:
    driver(policy, steps=100)
    if should_save(step):
      checkpoint.save()
  checkpoint.save()
  logger.write()
