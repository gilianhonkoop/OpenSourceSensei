import numpy as np

def rasterise_n(n):
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    mpl.use('Agg')
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.text(0.5, 0.5, n, va='center', ha='center', fontsize=16)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.canvas.draw()
    return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))

def format_num(x, round_float=2, fill=6):
    return format(round(x, round_float), f'.{round_float}f').zfill(fill)

def nums2vid(ep):
    assert len(ep.shape) == 2 and ep.shape[1] == 1, f"Got unexpected shape {ep.shape}"
    nums = []
    for i in range(ep.shape[0]):
        num_i = rasterise_n(format_num(ep[i, 0]))
        nums.append(num_i)
    return np.stack(nums)