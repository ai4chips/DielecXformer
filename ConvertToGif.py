import imageio.v2 as imageio
import moviepy as mp

# 读取视频
clip = mp.VideoFileClip("./docs/figs/BayesianResult.mp4")

# 设定目标帧率
fps = 20
duration = int(1000 / fps)  # 每帧持续时间，毫秒

# 抽取每一帧
frames = [clip.get_frame(t) for t in [i / fps for i in range(int(clip.duration * fps))]]

# 保存为 GIF
imageio.mimsave(
    "./docs/figs/BayesianResult.gif",
    frames,
    duration=duration,     # 代替 fps
    quantizer='nq',        # 可选：使用 NeuQuant 算法压缩颜色
    loop=0
)