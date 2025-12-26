import cv2
import queue
from .structures import VideoData

class VideoReader:
    """
    改进的VideoReader类，用于分段读取长视频并均匀采样，结果通过阻塞队列输出。
    """

    def __init__(self, video_path: str, output_queue: queue.Queue, target_width: int = 640, target_height: int = 640):
        """
        初始化VideoReader。
        
        参数:
            video_path (str): 视频文件路径。
            output_queue (queue.Queue): 用于传递VideoData的阻塞队列。
            target_width (int): 输出帧的目标宽度，默认640。
            target_height (int): 输出帧的目标高度，默认640。
        """
        self.video_path = video_path
        self.output_queue = output_queue
        self.cap = None
        self.fps = 0.0
        self.total_frames = 0
        self.duration = 0.0

    def process_video(self, frames_per_segment: int = 16) -> None:
        """
        处理整个视频：分段、采样，并将每个段的VideoData放入阻塞队列。
        
        参数:
            frames_per_segment (int): 每个3秒段内目标采集的帧数，默认为16。
        """
        # 打开视频文件
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        # 获取视频基本信息[4](@ref)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0:
            raise ValueError("无法获取有效的视频帧率。")
        self.duration = self.total_frames / self.fps

        print(f"视频信息: 帧率={self.fps:.2f} FPS, 总帧数={self.total_frames}, 时长={self.duration:.2f}秒")

        # 计算固定的采样时间间隔（秒）[3](@ref)
        segment_duration = 3.0
        sample_interval = segment_duration / frames_per_segment  # 例如 3/16 = 0.1875秒

        current_time = 0.0
        segment_index = 0

        while current_time < self.duration:
            # 计算当前段的结束时间
            segment_end_time = min(current_time + segment_duration, self.duration)
            actual_segment_duration = segment_end_time - current_time

            # 计算当前段内所有的采样时间点[4](@ref)
            num_samples = int(actual_segment_duration / sample_interval) 
            sample_times = [current_time + i * sample_interval for i in range(num_samples)]
            # 确保最后一个采样点不超过段结束时间
            sample_times = [t for t in sample_times if t < segment_end_time + 1e-6]  # 添加容差

            frames_in_segment = []
            frames_index = []
            for sample_time in sample_times:
                # 将时间转换为近似的帧索引[4](@ref)
                frame_index = int(sample_time * self.fps)
                frame_index = min(frame_index, self.total_frames - 1)  # 确保不越界

                # 跳转到指定帧并读取
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = self.cap.read()
                if ret:
                    # 调整帧尺寸
                    frames_in_segment.append(frame)
                    frames_index.append(frame_index)
                else:
                    print(f"警告: 在时间点 {sample_time:.2f}s (帧索引 {frame_index}) 处读取帧失败。")

            # 创建VideoData对象并放入队列[6,8](@ref)
            video_data = VideoData(
                frame_rate=self.fps,
                frames_list=frames_in_segment,
                segment_start_time=current_time,
                segment_end_time=segment_end_time,
                frame_index = frames_index,
                video_path=self.video_path
            )
            print(f"生产段 {segment_index}: 时间 [{current_time:.2f}s, {segment_end_time:.2f}s], 采样帧数 {len(frames_in_segment)}")
            self.output_queue.put(video_data)  # 可能在此阻塞，直到队列有空位

            segment_index += 1
            current_time = segment_end_time  # 移动到下一段

        # 发送结束信号
        self.output_queue.put(None)
        print("视频处理完成，已发送结束信号。")
        self.cap.release()

    def __del__(self):
        """资源清理。"""
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()