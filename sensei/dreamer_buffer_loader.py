import os

# configs = Agent.configs
import ruamel.yaml as yaml

from dreamerv3.embodied import replay as dreamer_replay
from dreamerv3.embodied.core.basics import unpack
from dreamerv3.embodied.core.checkpoint import Checkpoint
from dreamerv3.embodied.core.config import Config
from dreamerv3.embodied.core.path import Path
from dreamerv3.embodied.core.uuid import uuid
from dreamerv3.embodied.replay.chunk import Chunk
from dreamerv3.embodied.replay.saver import Saver

import imageio

configs = yaml.YAML(typ='safe').load(
    (Path("dreamerv3/configs.yaml")).read())

def scan(directory, capacity=None, shorten=0):
    directory = Path(directory)
    filenames, total = [], 0
    for filename in reversed(sorted(directory.glob('*.npz'))):
        if capacity and total >= capacity:
            break
        filenames.append(filename)
        total += max(0, int(filename.stem.split('-')[3]) - shorten)
    print(total)
    return sorted(filenames)

def make_replay(
    config, directory=None, is_eval=False, rate_limit=False, **kwargs):
  assert config.replay == 'uniform' or not rate_limit
  length = config.batch_length
  size = config.replay_size // 10 if is_eval else config.replay_size
  if config.replay == 'uniform' or is_eval:
    kw = {'online': config.replay_online}
    if rate_limit and config.run.train_ratio > 0:
      kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
      kw['tolerance'] = 10 * config.batch_size
      kw['min_size'] = config.batch_size
    replay = dreamer_replay.Uniform(length, size, directory, **kw)
  elif config.replay == 'reverb':
    replay = dreamer_replay.Reverb(length, size, directory)
  elif config.replay == 'chunks':
    replay = dreamer_replay.NaiveChunks(length, size, directory)
  else:
    raise NotImplementedError(config.replay)
  return replay

def setup_video(output_path, name_suffix, name_prefix, fps):
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{name_prefix}rollout{name_suffix}.mp4")
    i = 0
    while os.path.isfile(file_path):
        i += 1
        file_path = os.path.join(output_path, f"{name_prefix}rollout{name_suffix}_{i}.mp4")
    print("Record video in {}".format(file_path))
    return (
        imageio.get_writer(file_path, fps=fps, codec="h264", quality=10, pixelformat="yuv420p"), #yuv420p, yuvj422p
        file_path,
    )


if __name__ == "__main__":
    import numpy as np

    basedir = "/home/gilian/Documents/Uni/UVA/master/Deep Learning 2/OpenSourceSensei/logdir"

    logdir = f"{basedir}/gpt/seed10/replay"

    chunk_uuids = []
    filenames = scan(logdir, capacity=None, shorten=0)
    for filename in sorted(filenames):
        chunk_uuids.append(filename.stem.split('-')[1])

    # See configs.yaml for all options.
    config = Config(configs['defaults'])
    config = config.update(configs['robodesk_sensei'])
    # print(config)

    replay = make_replay(config, directory=logdir, is_eval=False)
    print("Made replay!")
    print(len(replay.table))
    # print(replay._sample)

    for _  in range(40):
        seq = replay._sample()

        output_path = f"{basedir}/replays"

        video, video_path = setup_video(output_path, f'',"", 40)
        for t, frame in enumerate(seq["image"]):
            video.append_data(frame)
        video.close()

        print("made video at: ", video_path)

    total_samples = 0
    for key in replay.table.keys():
       total_samples += len(replay.table[key])

    print(total_samples)